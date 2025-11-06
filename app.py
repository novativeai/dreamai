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
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

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

# Load PADDLE_ENVIRONMENT early for CORS configuration
PADDLE_ENVIRONMENT = os.environ.get("PADDLE_ENVIRONMENT", "sandbox").lower()

app = FastAPI(title="DreamAI API")

# Configure CORS based on environment
ALLOWED_ORIGINS = []

if PADDLE_ENVIRONMENT == "production":
    # Production: Only allow specific domains
    ALLOWED_ORIGINS = [
        "https://dreamai-checkpoint.netlify.app",
        "https://vision-ai-tester.netlify.app",
    ]
else:
    # Development/Sandbox: Allow localhost for testing
    ALLOWED_ORIGINS = [
        "https://dreamai-checkpoint.netlify.app",
        "http://localhost:8081",
        "http://localhost:3000",
        "http://localhost:19006",  # Expo web default
        "https://vision-ai-tester.netlify.app",
        "exp://localhost:8081",   # Expo native
    ]

print(f"✓ CORS configured for {len(ALLOWED_ORIGINS)} allowed origins")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Only allow necessary methods
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
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
# PADDLE_ENVIRONMENT already loaded at the top for CORS configuration

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
# === FILE VALIDATION UTILITY ======================================================
# =================================================================================
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

async def validate_image_file(file: UploadFile) -> bytes:
    """
    Validate uploaded image file for security and size constraints.

    Args:
        file: The uploaded file to validate

    Returns:
        bytes: The validated file content

    Raises:
        HTTPException: If validation fails
    """
    # Read file content
    content = await file.read()

    # Validate file size
    file_size_mb = len(content) / (1024 * 1024)
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({file_size_mb:.1f}MB). Maximum size is {MAX_FILE_SIZE_MB}MB."
        )

    # Validate content type
    allowed_types = ["image/jpeg", "image/png", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Allowed types: JPEG, PNG, WebP"
        )

    # Validate the file can be opened as an image (prevents malformed files)
    try:
        img = Image.open(BytesIO(content))
        img.verify()  # Verify it's a valid image

        # Validate dimensions
        width, height = img.size
        if width < 100 or height < 100:
            raise HTTPException(
                status_code=400,
                detail=f"Image too small ({width}x{height}px). Minimum size is 100x100px."
            )
        if width > 4096 or height > 4096:
            raise HTTPException(
                status_code=400,
                detail=f"Image too large ({width}x{height}px). Maximum size is 4096x4096px."
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid or corrupted image file: {str(e)}"
        )

    print(f"✓ Image validated: {file_size_mb:.2f}MB, {width}x{height}px, {file.content_type}")
    return content


# =================================================================================
# === WATERMARK UTILITY FUNCTION ==================================================
# =================================================================================
def add_watermark(image_bytes: bytes, watermark_text: str = "DreamAI") -> bytes:
    """
    Add a watermark to an image in the bottom right corner.

    Args:
        image_bytes: The original image as bytes
        watermark_text: The text to use as watermark (default: "DreamAI")

    Returns:
        The watermarked image as bytes (JPEG format)
    """
    try:
        # Open the image from bytes
        img = Image.open(BytesIO(image_bytes))

        # Calculate font size based on image dimensions (2% of image height)
        img_width, img_height = img.size
        font_size = max(int(img_height * 0.02), 12)  # Minimum 12px

        # Try to use a nice font, fall back to default if not available
        try:
            # Common font paths across different systems
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
                "/System/Library/Fonts/Helvetica.ttc",  # macOS
                "C:\\Windows\\Fonts\\arial.ttf",  # Windows
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",  # Linux alternative
            ]
            font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                    break

            if font is None:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        # Create a semi-transparent overlay for the watermark background
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        # Get text bounding box
        bbox = overlay_draw.textbbox((0, 0), watermark_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Calculate position (bottom right with padding)
        padding = int(img_height * 0.015)  # 1.5% padding
        bg_padding = 5
        x = img_width - text_width - padding
        y = img_height - text_height - padding

        # Draw semi-transparent black rectangle on overlay
        bg_bbox = (
            x - bg_padding,
            y - bg_padding,
            x + text_width + bg_padding,
            y + text_height + bg_padding
        )
        overlay_draw.rectangle(bg_bbox, fill=(0, 0, 0, 180))  # Semi-transparent black

        # Draw white text on overlay
        overlay_draw.text((x, y), watermark_text, fill=(255, 255, 255, 255), font=font)

        # Convert base image to RGBA for compositing
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        # Composite the overlay onto the image
        img = Image.alpha_composite(img, overlay)

        # Convert back to RGB for JPEG output
        img = img.convert('RGB')

        # Save to bytes
        output_buffer = BytesIO()
        img.save(output_buffer, format='JPEG', quality=95)
        output_buffer.seek(0)

        return output_buffer.read()

    except Exception as e:
        print(f"Error adding watermark: {e}")
        # Return original image if watermarking fails
        return image_bytes


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
        # Validate and read the uploaded image
        image_bytes = await validate_image_file(image1)

        # Upload the image to fal.ai storage and get a URL
        print(f"Uploading image to fal.ai storage...")
        uploaded_image_url = fal_client.upload(image_bytes, "image/jpeg")
        print(f"Image uploaded: {uploaded_image_url}")

        # Call the Fal AI model using fal_client.run for a direct response
        print(f"Calling fal.ai flux-pro/v1.1-ultra/kontext with prompt: {prompt[:50]}...")
        result = fal_client.run(
            "fal-ai/flux-pro/v1.1-ultra/kontext",
            arguments={
                "prompt": prompt,
                "image_url": uploaded_image_url,
                "guidance_scale": 3.5,
                "num_inference_steps": 28,
                "num_images": 1,
                "output_format": "jpeg",
                "image_prompt_strength": 0.1,
            },
        )

        print(f"Fal.ai response received: {type(result)}")

        # Process the response from Fal AI
        # Fal.ai typically returns: {"images": [{"url": "https://..."}]}
        if not result or "images" not in result or len(result["images"]) == 0:
            raise HTTPException(status_code=500, detail="No image was generated by the model.")

        generated_image = result["images"][0]

        # Check if we got a URL (typical fal.ai response) or direct content
        if "url" in generated_image:
            image_url = generated_image["url"]
            print(f"Downloading image from fal.ai: {image_url}")

            # Download the image from the URL
            img_response = requests.get(image_url, timeout=30)
            img_response.raise_for_status()
            image_content_bytes = img_response.content
        elif "content" in generated_image:
            # Direct content (less common)
            image_content_bytes = generated_image["content"]
        else:
            raise HTTPException(
                status_code=500,
                detail="Unexpected response format from image generation service."
            )

        # Add watermark to the image
        print("Adding watermark to image...")
        watermarked_image_bytes = add_watermark(image_content_bytes, "DreamAI")

        print(f"Returning watermarked image ({len(watermarked_image_bytes)} bytes)")
        return Response(content=watermarked_image_bytes, media_type="image/jpeg")

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
    Enhanced for frontend compatibility with proper sorting and filtering.
    """
    try:
        # Pass parameters directly as kwargs in paddle-python-sdk 1.11.0+
        # Use include_prices=True for including price data
        product_iter = paddle.products.list(status=status, include_prices=True)

        serialized_products = []
        for p in product_iter:
            serialized = serialize_product(p)
            # Ensure custom_data exists for frontend filtering
            if not serialized.get("custom_data"):
                serialized["custom_data"] = {}

            # Transform to frontend-compatible format
            custom_data = serialized.get("custom_data", {})

            # Extract type from custom_data (required by frontend)
            product_type = custom_data.get("type")
            if not product_type:
                # Skip products without a type
                print(f"WARNING: Product {serialized.get('id')} missing type in custom_data, skipping")
                continue

            # Format price from prices array
            prices = serialized.get("prices", [])
            formatted_price = "$0.00"
            interval = None
            price_id = None

            if prices and len(prices) > 0:
                first_price = prices[0]
                amount = first_price.get("amount")
                currency = first_price.get("currency", "USD")
                price_id = first_price.get("id")

                # Format price (amount is in cents for USD)
                if amount:
                    if currency == "USD":
                        formatted_price = f"${float(amount) / 100:.2f}"
                    else:
                        formatted_price = f"{float(amount) / 100:.2f} {currency}"

                # Extract interval for subscriptions
                billing_cycle = first_price.get("billing_cycle")
                if billing_cycle:
                    interval_value = billing_cycle.get("interval")
                    if interval_value:
                        interval = interval_value

            # Build frontend-compatible object
            frontend_item = {
                "type": product_type,
                "id": price_id or serialized.get("id"),  # Use price_id for checkout
                "name": serialized.get("name"),
                "description": serialized.get("description", ""),
                "price": formatted_price,
            }

            # Add type-specific fields
            if product_type == "subscription":
                frontend_item["interval"] = interval
                frontend_item["isRecommended"] = custom_data.get("isRecommended", False)
            elif product_type == "credits":
                frontend_item["credits"] = custom_data.get("credits", 0)
                frontend_item["isPopular"] = custom_data.get("isPopular", False)

            serialized_products.append(frontend_item)

        # Sort by isRecommended/isPopular for frontend display
        serialized_products.sort(
            key=lambda x: (
                x.get("isRecommended") is not True and x.get("isPopular") is not True,
                x.get("name", "")
            )
        )

        print(f"✓ Returning {len(serialized_products)} products for frontend")
        return {"success": True, "data": serialized_products}

    except ApiError as e:
        detail = getattr(e, "message", str(e))
        print(f"Paddle API error in /products: {detail}")
        raise HTTPException(status_code=502, detail=f"Paddle API error: {detail}")
    except Exception as e:
        print(f"Internal error in /products: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


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
