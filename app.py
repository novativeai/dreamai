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
        "https://dreamai-mvp.netlify.app",
        "http://localhost:19006",
        # Expo web default
        "https://vision-ai-tester.netlify.app",
        "exp://localhost:8081",   # Expo native
        "*"  # Allow all origins for development - restrict in production
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
    "FIREBASE_SERVICE_ACCOUNT_BASE64",
    "FAL_AI_KEY",
]
# Optional but recommended
optional_envs = ["PADDLE_ENVIRONMENT"]  # Defaults to 'sandbox' if not set

missing = [k for k in required_envs if not os.environ.get(k)]
if missing:
    raise ValueError(f"All required environment variables must be set. Missing: {missing}")

print("✓ All required environment variables loaded")
if not os.environ.get("PADDLE_ENVIRONMENT"):
    print("⚠ PADDLE_ENVIRONMENT not set, defaulting to 'sandbox'")

# Configure Fal AI client with the key
fal_client.api_key = os.environ.get("FAL_AI_KEY")

# --- PADDLE CLIENT INIT ---
PADDLE_API_KEY = os.environ.get("PADDLE_API_KEY")
PADDLE_ENVIRONMENT = os.environ.get("PADDLE_ENVIRONMENT", "sandbox").lower()

# Initialize Paddle client with appropriate environment
if PADDLE_ENVIRONMENT == "production":
    paddle = Client(PADDLE_API_KEY, options=Options(Environment.PRODUCTION))
    print("✓ Paddle initialized in PRODUCTION mode")
else:
    paddle = Client(PADDLE_API_KEY, options=Options(Environment.SANDBOX))
    print("✓ Paddle initialized in SANDBOX mode")


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

@app.get("/products")
async def get_products():
    """
    Fetch products from Paddle and transform them for frontend consumption.
    DEBUG VERSION - with extensive logging
    """
    try:
        print("=" * 60)
        print("🔍 FETCHING PRODUCTS FROM PADDLE")
        print("=" * 60)
        
        product_iter = paddle.products.list()
        subscription_plans = []
        
        product_count = 0
        for product in product_iter:
            product_count += 1
            prod_id = getattr(product, "id", None)
            prod_name = getattr(product, "name", "")
            prod_status = getattr(product, "status", "active")
            
            print(f"\n📦 Product #{product_count}:")
            print(f"   ID: {prod_id}")
            print(f"   Name: {prod_name}")
            print(f"   Status: {prod_status}")
            
            # Check custom_data
            def safe_dict(obj):
                if obj is None:
                    return {}
                if isinstance(obj, dict):
                    return obj
                try:
                    return vars(obj)
                except TypeError:
                    return getattr(obj, '__dict__', {})
            
            custom_data = safe_dict(getattr(product, "custom_data", None))
            print(f"   Custom Data: {custom_data}")
            
            # Skip archived products
            if prod_status != "active":
                print(f"   ❌ SKIPPED: Product not active (status={prod_status})")
                continue
            
            if not prod_id:
                print(f"   ❌ SKIPPED: No product ID")
                continue
            
            # Fetch prices
            print(f"   🔍 Fetching prices for product {prod_id}...")
            try:
                price_iter = paddle.prices.list(product_id=prod_id)
                price_count = 0
                
                for price in price_iter:
                    price_count += 1
                    price_id = getattr(price, "id", None)
                    price_status = getattr(price, "status", "active")
                    price_name = getattr(price, "name", "Unnamed")
                    
                    # Get unit price info
                    unit_price = getattr(price, "unit_price", None)
                    amount = getattr(unit_price, "amount", "0") if unit_price else "0"
                    
                    print(f"      💰 Price #{price_count}:")
                    print(f"         ID: {price_id}")
                    print(f"         Name: {price_name}")
                    print(f"         Status: {price_status}")
                    print(f"         Amount: {amount}")
                    
                    if price_status != "active":
                        print(f"         ❌ SKIPPED: Price not active")
                        continue
                    
                    if not price_id:
                        print(f"         ❌ SKIPPED: No price ID")
                        continue
                    
                    # If we got here, we'll add it
                    print(f"         ✅ ADDING TO RESULTS")
                    
                    # Extract all the data (simplified for debugging)
                    currency = getattr(unit_price, "currency_code", "USD") if unit_price else "USD"
                    billing_cycle = getattr(price, "billing_cycle", None)
                    interval = getattr(billing_cycle, "interval", None) if billing_cycle else None
                    
                    price_amount = float(amount) / 100
                    formatted_price = f"${price_amount:.2f}"
                    
                    subscription_plans.append({
                        "id": price_id,
                        "productId": prod_id,
                        "name": prod_name,
                        "price": formatted_price,
                        "interval": interval,
                        "description": getattr(product, "description", ""),
                        "isRecommended": custom_data.get("isRecommended", False),
                    })
                
                print(f"   📊 Found {price_count} prices for this product")
                
                if price_count == 0:
                    print(f"   ⚠️  WARNING: No prices found for product {prod_id}")
                    
            except Exception as e:
                print(f"   ❌ ERROR fetching prices: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n" + "=" * 60)
        print(f"📊 SUMMARY:")
        print(f"   Total products found: {product_count}")
        print(f"   Total plans to return: {len(subscription_plans)}")
        print("=" * 60)
        
        return {"success": True, "data": subscription_plans}
        
    except ApiError as e:
        print(f"❌ Paddle API error: {e}")
        raise HTTPException(status_code=502, detail=f"Paddle API error: {str(e)}")
    except Exception as e:
        print(f"❌ Internal error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
        

@app.post("/create-checkout")
async def create_checkout(request: Request):
    """
    Create a Paddle checkout session for subscription or credit purchase.
    Enhanced to handle email parameter and provide better error messages.
    """
    body = await request.json()
    price_id = body.get('priceId')
    user_id = body.get('userId')
    email = body.get('email')  # Optional: pre-fill customer email
    
    if not price_id or not user_id:
        raise HTTPException(
            status_code=400, 
            detail="priceId and userId are required"
        )
    
    try:
        # Get user document from Firebase
        user_doc = db.collection('users').document(user_id).get()
        user_data = user_doc.to_dict() if user_doc.exists else {}
        paddle_customer_id = user_data.get('paddle_customer_id')
        
        # If no email provided, try to get from Firebase
        if not email and user_data:
            email = user_data.get('email')

        # Build transaction payload
        txn_payload = {
            "items": [
                {"price_id": price_id, "quantity": 1}
            ],
            "custom_data": {"firebase_uid": user_id},
            "success_url": "https://dreamai-checkpoint.netlify.app/payment-success",
        }
        
        # Add customer_id if exists (for returning customers)
        if paddle_customer_id:
            txn_payload["customer_id"] = paddle_customer_id
        
        # Add email if available (for new customers)
        if email and not paddle_customer_id:
            txn_payload["customer_email"] = email

        print(f"Creating checkout for user {user_id} with price {price_id}")
        transaction = paddle.transactions.create(txn_payload)

        # Extract checkout URL intelligently
        url = None
        if hasattr(transaction, "checkout"):
            checkout_obj = transaction.checkout
            if hasattr(checkout_obj, "url"):
                url = checkout_obj.url
        
        if not url:
            url = getattr(transaction, "checkout_url", None) or getattr(transaction, "url", None)

        if not url:
            print(f"Failed to extract checkout URL from transaction: {transaction}")
            raise HTTPException(
                status_code=500, 
                detail="Failed to create checkout URL. Please try again."
            )

        print(f"Successfully created checkout: {url}")
        return {
            "success": True,
            "checkout_url": url,
            "transaction_id": getattr(transaction, "id", None)
        }

    except ApiError as e:
        error_msg = getattr(e, 'message', str(e))
        print(f"Paddle API error in /create-checkout: {error_msg}")
        raise HTTPException(
            status_code=500, 
            detail=f"Paddle API error: {error_msg}"
        )
    except Exception as e:
        print(f"Internal error in /create-checkout: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"An internal error occurred: {str(e)}"
        )



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
    """
    Handle Paddle webhook events.
    CRITICAL: Must respond within 5 seconds to prevent retries.
    Enhanced with better logging and error handling.
    """
    if not paddle_signature:
        print("ERROR: Missing Paddle-Signature header")
        raise HTTPException(status_code=400, detail="Missing Paddle-Signature header")
    
    try:
        # Get raw body for signature verification
        body_bytes = await request.body()
        webhook_secret = os.environ.get("PADDLE_WEBHOOK_SECRET")
        
        if not webhook_secret:
            print("ERROR: PADDLE_WEBHOOK_SECRET not configured")
            raise HTTPException(status_code=500, detail="PADDLE_WEBHOOK_SECRET not configured")

        # Verify webhook signature
        verifier = Verifier()
        secret = Secret(webhook_secret)

        try:
            integrity_ok = verifier.verify(request, secret)
        except Exception as verify_error:
            print(f"ERROR: Signature verification failed: {verify_error}")
            raise HTTPException(status_code=401, detail="Invalid webhook signature")
        
        if not integrity_ok:
            print("ERROR: Webhook signature verification failed")
            raise HTTPException(status_code=401, detail="Invalid webhook signature")

        # Parse event data
        event_data = await request.json()
        event_type = event_data.get("event_type")
        event_id = event_data.get("event_id")
        data = event_data.get("data")

        print(f"✓ Received webhook: {event_type} (ID: {event_id})")

        # Process different event types
        if event_type == "transaction.completed":
            await handle_transaction_completed(data)
        
        elif event_type == "subscription.created":
            await handle_subscription_created(data)
        
        elif event_type == "subscription.updated":
            await handle_subscription_updated(data)
        
        elif event_type == "subscription.canceled":
            await handle_subscription_canceled(data)
        
        elif event_type == "subscription.past_due":
            await handle_subscription_past_due(data)
        
        else:
            print(f"INFO: Unhandled event type: {event_type}")

        # IMPORTANT: Respond quickly (within 5 seconds)
        return {"status": "received", "event_id": event_id}

    except HTTPException:
        raise
    except ApiError as e:
        error_msg = getattr(e, 'message', str(e))
        print(f"ERROR: Paddle API error in webhook: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Paddle API error: {error_msg}")
    except Exception as e:
        print(f"ERROR: Unexpected error in webhook processing: {e}")
        import traceback
        traceback.print_exc()
        # Still return 200 to prevent Paddle retries for non-signature errors
        return {"status": "error", "message": "Internal processing error"}


# --- Webhook Event Handlers ---

async def handle_transaction_completed(data: dict):
    """Handle transaction.completed event - provision access or credits"""
    firebase_uid = data.get("custom_data", {}).get("firebase_uid")
    
    if not firebase_uid:
        print("WARNING: transaction.completed missing firebase_uid")
        return
    
    try:
        user_ref = db.collection('users').document(firebase_uid)
        paddle_customer_id = data.get("customer_id")
        
        # Update paddle_customer_id if available
        if paddle_customer_id:
            user_ref.set({"paddle_customer_id": paddle_customer_id}, merge=True)
            print(f"✓ Updated paddle_customer_id for user {firebase_uid}")
        
        # Process each item in the transaction
        for item in data.get("items", []):
            product_id = item.get("product", {}).get("id")
            
            if not product_id:
                continue
            
            try:
                # Fetch product details
                product_details = paddle.products.get(product_id)
                custom_data = getattr(product_details, "custom_data", {}) or {}
                product_type = custom_data.get("type")
                product_name = getattr(product_details, "name", "Unknown")
                
                if product_type == "credits":
                    # Add credits to user account
                    credits_to_add = custom_data.get("credits", 0)
                    if credits_to_add > 0:
                        user_ref.update({"credits": firestore.Increment(credits_to_add)})
                        print(f"✓ Added {credits_to_add} credits to user {firebase_uid}")
                
                elif product_type == "subscription":
                    # Activate subscription
                    subscription_id = data.get("subscription_id")
                    user_ref.set({
                        "premium_status": product_name,
                        "subscription_id": subscription_id,
                        "subscription_status": "active"
                    }, merge=True)
                    print(f"✓ Activated subscription '{product_name}' for user {firebase_uid}")
            
            except Exception as e:
                print(f"WARNING: Unable to fetch product {product_id}: {e}")
                continue
    
    except Exception as e:
        print(f"ERROR: Failed to handle transaction.completed: {e}")
        raise


async def handle_subscription_created(data: dict):
    """Handle subscription.created event"""
    customer_id = data.get("customer_id")
    subscription_id = data.get("id")
    status = data.get("status")
    custom_data = data.get("custom_data", {})
    firebase_uid = custom_data.get("firebase_uid")
    
    print(f"✓ Subscription created: {subscription_id} (status: {status})")
    
    if firebase_uid:
        try:
            user_ref = db.collection('users').document(firebase_uid)
            user_ref.set({
                "paddle_customer_id": customer_id,
                "subscription_id": subscription_id,
                "subscription_status": status
            }, merge=True)
            print(f"✓ Updated subscription for user {firebase_uid}")
        except Exception as e:
            print(f"ERROR: Failed to update user on subscription.created: {e}")


async def handle_subscription_updated(data: dict):
    """Handle subscription.updated event"""
    customer_id = data.get("customer_id")
    subscription_id = data.get("id")
    subscription_status = data.get("status")
    
    try:
        # Find user by paddle_customer_id
        users_query = db.collection('users').where(
            'paddle_customer_id', '==', customer_id
        ).limit(1).stream()
        
        for user_doc in users_query:
            user_ref = user_doc.reference
            
            if subscription_status != 'active':
                # Deactivate premium if not active
                user_ref.update({
                    "premium_status": None,
                    "subscription_status": subscription_status
                })
                print(f"✓ Deactivated subscription for user {user_doc.id} (status: {subscription_status})")
            else:
                # Keep subscription active
                user_ref.update({"subscription_status": "active"})
                print(f"✓ Subscription remains active for user {user_doc.id}")
    
    except Exception as e:
        print(f"ERROR: Failed to handle subscription.updated: {e}")


async def handle_subscription_canceled(data: dict):
    """Handle subscription.canceled event"""
    customer_id = data.get("customer_id")
    subscription_id = data.get("id")
    
    try:
        # Find user by paddle_customer_id
        users_query = db.collection('users').where(
            'paddle_customer_id', '==', customer_id
        ).limit(1).stream()
        
        for user_doc in users_query:
            user_ref = user_doc.reference
            user_ref.update({
                "premium_status": None,
                "subscription_status": "canceled"
            })
            print(f"✓ Canceled subscription for user {user_doc.id}")
    
    except Exception as e:
        print(f"ERROR: Failed to handle subscription.canceled: {e}")


async def handle_subscription_past_due(data: dict):
    """Handle subscription.past_due event"""
    customer_id = data.get("customer_id")
    subscription_status = data.get("status")
    
    try:
        # Find user by paddle_customer_id
        users_query = db.collection('users').where(
            'paddle_customer_id', '==', customer_id
        ).limit(1).stream()
        
        for user_doc in users_query:
            user_ref = user_doc.reference
            user_ref.update({
                "subscription_status": subscription_status
            })
            print(f"✓ Marked subscription as past_due for user {user_doc.id}")
    
    except Exception as e:
        print(f"ERROR: Failed to handle subscription.past_due: {e}")


@app.get("/subscription-status/{user_id}")
async def get_subscription_status(user_id: str):
    """
    Get current subscription status for a user.
    Useful for frontend to check subscription state.
    """
    try:
        user_doc = db.collection('users').document(user_id).get()
        
        if not user_doc.exists:
            raise HTTPException(status_code=404, detail="User not found")
        
        user_data = user_doc.to_dict()
        
        return {
            "success": True,
            "user_id": user_id,
            "premium_status": user_data.get("premium_status"),
            "subscription_status": user_data.get("subscription_status"),
            "subscription_id": user_data.get("subscription_id"),
            "paddle_customer_id": user_data.get("paddle_customer_id"),
            "credits": user_data.get("credits", 0),
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching subscription status: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch subscription status")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# Optionally run with `python server.py` for local dev
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
