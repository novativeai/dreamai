# server.py  (full file - all endpoints included)
import base64
import json
import os
import mimetypes
import uuid
from typing import Optional, List, Any, Dict
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request, Header, Response, Query
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import shutil

# --- NEW FAL AI IMPORT ---
import fal_client

# --- PADDLE & FIREBASE INTEGRATION (UPDATED TO paddle_billing) ---
# NOTE: ensure the installed package matches these imports (paddle_billing)
try:
    from paddle_billing import Client, Environment, Options
    from paddle_billing.Exceptions.ApiError import ApiError
    from paddle_billing.Notifications import Verifier, Secret
    from paddle_billing.Resources.Transactions.Operations import CreateTransaction, CreateTransactionItem
except Exception as e:
    # Provide a clearer runtime error if paddle_billing is missing
    raise ImportError(
        "paddle_billing import failed. Ensure you installed the correct package (paddle-python-sdk / paddle_billing) "
        "and are running Python >= 3.11 if required. Underlying error: " + str(e)
    )

import firebase_admin
from firebase_admin import credentials, firestore
# --- END INTEGRATION IMPORTS ---


load_dotenv()

app = FastAPI(title="DreamAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://dreamai-checkpoint.netlify.app",
        "http://localhost:8081",
        "http://localhost:3000",
        "https://vision-ai-tester.netlify.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- REQUIRED ENV CHECK (includes Paddle, Firebase, Fal) ---
required_envs = [
    "PADDLE_API_KEY",
    "PADDLE_WEBHOOK_SECRET",
    "FIREBASE_DATABASE_URL",
    "FIREBASE_SERVICE_ACCOUNT_BASE64",
    "FAL_AI_KEY",
]
missing = [k for k in required_envs if not os.environ.get(k)]
if missing:
    raise ValueError(f"All required environment variables must be set. Missing: {missing}")

# Configure Fal AI client with the key
fal_client.api_key = os.environ.get("FAL_AI_KEY")

# --- PADDLE CLIENT INIT ---
PADDLE_API_KEY = os.environ.get("PADDLE_API_KEY")
# Use Options(Environment.SANDBOX) for sandbox, Options(Environment.PRODUCTION) for live
paddle = Client(PADDLE_API_KEY, options=Options(Environment.SANDBOX))


# --- FIREBASE INIT FROM BASE64 ENV VAR (unchanged) ---
try:
    firebase_sa_base64 = os.environ.get("FIREBASE_SERVICE_ACCOUNT_BASE64")
    firebase_sa_decoded = base64.b64decode(firebase_sa_base64).decode('utf-8')
    firebase_sa_dict = json.loads(firebase_sa_decoded)
    cred = credentials.Certificate(firebase_sa_dict)
    firebase_admin.initialize_app(cred, {'databaseURL': os.environ.get("FIREBASE_DATABASE_URL")})
    db = firestore.client()
    print("Firebase Admin SDK initialized successfully from Base64 environment variable.")
except (ValueError, TypeError, json.JSONDecodeError) as e:
    raise ValueError(f"Error decoding or parsing Firebase service account from environment variable: {e}")
# --- END INITIALIZATION ---


# =================================================================================
# === REFACTORED /generate/ ENDPOINT FOR FAL AI ===================================
# =================================================================================
@app.post("/generate/")
async def generate_image(
    image1: UploadFile = File(...),
    # The payload can still contain these, but we will ignore them
    image2: Optional[UploadFile] = File(None),
    prompt: str = Form(...),
    temperature: float = Form(1.0), # temperature is not a direct param for FLUX
    top_p: float = Form(0.95),       # top_p is not a direct param for FLUX
    top_k: int = Form(40),         # top_k is not a direct param for FLUX
):
    """
    Generate an image using Fal AI's FLUX model based on ONE input image and a prompt.
    This endpoint maintains the original payload structure for frontend compatibility.
    """
    # Ensure an image was actually sent
    if not image1 or not image1.file:
        raise HTTPException(status_code=400, detail="An image file is required.")

    try:
        # Read the image bytes directly from the upload stream into memory.
        image_bytes = await image1.read()
        
        # Ensure the file type is one Fal AI can handle
        allowed_types = ["image/jpeg", "image/png", "image/webp"]
        if image1.content_type not in allowed_types:
            raise HTTPException(
                status_code=415, 
                detail=f"Unsupported image type: {image1.content_type}. Please use JPEG, PNG, or WebP."
            )

        # Call the Fal AI model using fal_client.run for a direct response
        result = fal_client.run(
            "fal-ai/flux-pro/kontext",
            arguments={
                "prompt": prompt,
                "image": fal_client.Image.from_bytes(image_bytes, format=image1.content_type.split('/')[1]),
                # Default values for FLUX can be set here
                "guidance_scale": 3.5,
                "num_images": 1,
                "output_format": "jpeg",
            },
        )

        # Process the response from Fal AI
        if not result or "images" not in result or len(result["images"]) == 0:
            raise HTTPException(status_code=500, detail="No image was generated by the model.")
        
        generated_image = result["images"][0]
        image_content_bytes = generated_image["content"]
        image_media_type = generated_image["content_type"]

        return Response(content=image_content_bytes, media_type=image_media_type)

    except fal_client.FALServerException as e:
        print(f"Fal AI Server Error: {e}")
        raise HTTPException(status_code=503, detail=f"The image generation service failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
# =================================================================================
# === END OF REFACTORED ENDPOINT ==================================================
# =================================================================================


# ---------------------------
# Helper: serialize product object from SDK or dict
# ---------------------------
def serialize_product(p: Any) -> Dict[str, Any]:
    """
    Convert a product SDK object or dict into a plain JSON-serializable dict.
    Includes prices (if present).
    """
    # product id / name / description / status / custom_data
    if isinstance(p, dict):
        prod_id = p.get("id")
        name = p.get("name")
        description = p.get("description")
        status_attr = p.get("status")
        custom_data = p.get("custom_data")
        raw_prices = p.get("prices", [])
    else:
        prod_id = getattr(p, "id", None) or getattr(p, "product_id", None)
        name = getattr(p, "name", None)
        description = getattr(p, "description", None)
        status_attr = getattr(p, "status", None)
        custom_data = getattr(p, "custom_data", None)
        raw_prices = getattr(p, "prices", None) or []

    prices = []
    for pr in raw_prices or []:
        if isinstance(pr, dict):
            price_id = pr.get("id")
            amount = pr.get("price") or pr.get("amount")
            currency = pr.get("currency")
            billing_cycle = pr.get("billing_cycle")
            interval = pr.get("interval")
        else:
            price_id = getattr(pr, "id", None)
            amount = getattr(pr, "price", None) or getattr(pr, "amount", None)
            currency = getattr(pr, "currency", None)
            billing_cycle = getattr(pr, "billing_cycle", None)
            interval = getattr(pr, "interval", None)
        prices.append({
            "id": price_id,
            "amount": amount,
            "currency": currency,
            "billing_cycle": billing_cycle,
            "interval": interval,
        })

    return {
        "id": prod_id,
        "name": name,
        "description": description,
        "status": status_attr,
        "custom_data": custom_data,
        "prices": prices,
    }


# --- ALL OTHER ENDPOINTS (products, checkouts, webhooks, health) ---

@app.get("/products")
async def get_products(status: str = Query("active", description="Filter by product status: active/archived")):
    """
    Fetch products from Paddle (includes prices). Returns JSON-serializable list.
    Uses SDK's products.list(...) which yields paginated results.
    """
    try:
        params = {"status": status, "include": "prices"}
        product_iter = paddle.products.list(params=params)

        serialized_products = []
        for p in product_iter:
            serialized_products.append(serialize_product(p))

        return {"data": serialized_products}

    except ApiError as e:
        detail = getattr(e, "message", str(e))
        raise HTTPException(status_code=502, detail=f"Paddle API error: {detail}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@app.post("/create-checkout")
async def create_checkout(request: Request):
    body = await request.json()
    price_id = body.get('priceId')
    user_id = body.get('userId')
    if not price_id or not user_id:
        raise HTTPException(status_code=400, detail="priceId and userId are required")
    try:
        user_doc = db.collection('users').document(user_id).get()
        paddle_customer_id = user_doc.to_dict().get('paddle_customer_id') if user_doc.exists else None

        # Build transaction items using the SDK's operation objects
        items = [
            CreateTransactionItem(
                price_id=price_id,
                quantity=1,
            )
        ]

        # Construct CreateTransaction operation. Add custom_data to tie to firebase UID.
        create_txn_op = CreateTransaction(
            items=items,
            customer_id=paddle_customer_id,
            custom_data={"firebase_uid": user_id},
            success_url="https://dreamai-checkpoint.netlify.app/payment-success",
        )

        transaction = paddle.transactions.create(create_txn_op)

        # Extract checkout URL - SDK shapes vary so be defensive
        checkout_obj = getattr(transaction, "checkout", None)
        url = None
        if checkout_obj:
            url = getattr(checkout_obj, "url", None) or getattr(checkout_obj, "checkout_url", None) or checkout_obj
        else:
            url = getattr(transaction, "checkout_url", None) or getattr(transaction, "url", None)

        if not url:
            raise HTTPException(status_code=500, detail="Failed to create checkout URL")

        return {"checkout_url": url}
    except ApiError as e:
        raise HTTPException(status_code=500, detail=f"Paddle API error: {getattr(e, 'message', str(e))}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")


@app.post("/create-customer-portal")
async def create_customer_portal(request: Request):
    body = await request.json()
    user_id = body.get('userId')
    if not user_id:
        raise HTTPException(status_code=400, detail="userId is required")
    user_doc = db.collection('users').document(user_id).get()
    if not user_doc.exists or not user_doc.to_dict().get('paddle_customer_id'):
        raise HTTPException(status_code=404, detail="User has no subscription to manage.")
    paddle_customer_id = user_doc.to_dict().get('paddle_customer_id')
    try:
        portal = paddle.customer_portal.create(paddle_customer_id)
        portal_url = getattr(portal, "url", None) or getattr(portal, "portal_url", None)
        if not portal_url:
            raise HTTPException(status_code=500, detail="Failed to create customer portal link")
        return {"portal_url": portal_url}
    except ApiError as e:
        raise HTTPException(status_code=500, detail=f"Paddle API error: {getattr(e, 'message', str(e))}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")


@app.post("/paddle-webhook")
async def paddle_webhook(request: Request, paddle_signature: str = Header(None)):
    if not paddle_signature:
        raise HTTPException(status_code=400, detail="Missing Paddle-Signature header")
    try:
        body_bytes = await request.body()
        webhook_secret = os.environ.get("PADDLE_WEBHOOK_SECRET")
        if not webhook_secret:
            raise HTTPException(status_code=500, detail="PADDLE_WEBHOOK_SECRET not configured")

        verifier = Verifier()
        secret = Secret(webhook_secret)

        # The SDK's verifier tries to work with framework request objects. If compatibility issues appear
        # you can pass a minimal object with `.body` and `.headers`. Here we attempt the simple approach first.
        integrity_ok = verifier.verify(request, secret)
        if not integrity_ok:
            raise HTTPException(status_code=401, detail="Invalid webhook signature")

        event_data = await request.json()
        event_type = event_data.get("event_type")
        data = event_data.get("data")

        if event_type == "transaction.completed":
            firebase_uid = data.get("custom_data", {}).get("firebase_uid")
            if not firebase_uid:
                return JSONResponse(status_code=400, content={"error": "Missing firebase_uid"})
            user_ref = db.collection('users').document(firebase_uid)
            paddle_customer_id = data.get("customer_id")
            if paddle_customer_id:
                user_ref.set({"paddle_customer_id": paddle_customer_id}, merge=True)
            for item in data.get("items", []):
                product_id = item.get("product", {}).get("id")
                # Defensive product retrieval
                try:
                    product_details = paddle.products.get(product_id)
                except Exception as e:
                    print(f"Warning: unable to fetch product {product_id}: {e}")
                    product_details = {}
                product_type = (getattr(product_details, "custom_data", None) or {}).get("type") if product_details else None
                if product_type == "credits":
                    credits_to_add = (getattr(product_details, "custom_data", None) or {}).get("credits", 0)
                    user_ref.update({"credits": firestore.Increment(credits_to_add)})
                    print(f"Added {credits_to_add} credits to user {firebase_uid}")
                elif product_type == "subscription":
                    user_ref.set({
                        "premium_status": getattr(product_details, "name", None),
                        "subscription_id": data.get("subscription_id"),
                        "subscription_status": "active"
                    }, merge=True)
                    print(f"ACTIVATED subscription for user {firebase_uid}")

        elif event_type == "subscription.canceled":
            customer_id = data.get("customer_id")
            users_query = db.collection('users').where('paddle_customer_id', '==', customer_id).limit(1).stream()
            for user_doc in users_query:
                user_ref = user_doc.reference
                user_ref.update({"premium_status": None, "subscription_status": "canceled"})
                print(f"CANCELED subscription for user {user_doc.id}")

        elif event_type in ("subscription.updated", "subscription.past_due"):
            customer_id = data.get("customer_id")
            subscription_status = data.get("status")
            users_query = db.collection('users').where('paddle_customer_id', '==', customer_id).limit(1).stream()
            for user_doc in users_query:
                user_ref = user_doc.reference
                if subscription_status != 'active':
                    user_ref.update({"premium_status": None, "subscription_status": subscription_status})
                    print(f"DEACTIVATED subscription for user {user_doc.id} due to status: {subscription_status}")
                else:
                    user_ref.update({"subscription_status": "active"})

        return {"status": "received"}
    except ApiError as e:
        raise HTTPException(status_code=500, detail=f"Paddle API error: {getattr(e, 'message', str(e))}")
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid signature")
    except Exception as e:
        print(f"Error processing webhook: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# Optionally run with `python server.py` for local dev
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
