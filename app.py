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
    Generate an image using Fal AI's FLUX Kontext model based on ONE input image and a prompt.
    This endpoint maintains the original payload structure for frontend compatibility.
    Downloads the generated image from fal.ai URL and adds "DreamAI" watermark.
    """
    # Ensure an image was actually sent
    if not image1 or not image1.file:
        raise HTTPException(status_code=400, detail="An image file is required.")

    try:
        # Read the image bytes
        image_bytes = await image1.read()

        # Validate file type
        allowed_types = ["image/jpeg", "image/png", "image/webp"]
        if image1.content_type not in allowed_types:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported image type: {image1.content_type}. Please use JPEG, PNG, or WebP."
            )

        print(f"📤 Uploading image to fal.ai storage...")
        # Upload the image to fal.ai storage and get a URL
        uploaded_image_url = fal_client.upload(image_bytes, "image/jpeg")
        print(f"✅ Image uploaded: {uploaded_image_url}")

        print(f"🎨 Generating image with prompt: {prompt[:50]}...")
        # Call the Fal AI FLUX Kontext model with image_url parameter
        result = fal_client.run(
            "fal-ai/flux-pro/kontext",
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

        print(f"✅ Fal.ai response received: {type(result)}")

        # Process the response from Fal AI
        # Fal.ai returns: {"images": [{"url": "https://..."}]}
        if not result or "images" not in result or len(result["images"]) == 0:
            raise HTTPException(status_code=500, detail="No image was generated by the model.")

        generated_image = result["images"][0]

        # Check if we got a URL (typical fal.ai response) or direct content
        if "url" in generated_image:
            image_url = generated_image["url"]
            print(f"📥 Downloading generated image from: {image_url}")

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

        # Add watermark to the generated image
        print("🏷️  Adding DreamAI watermark...")
        watermarked_image_bytes = add_watermark(image_content_bytes, "DreamAI")

        print(f"✅ Returning watermarked image ({len(watermarked_image_bytes)} bytes)")
        return Response(content=watermarked_image_bytes, media_type="image/jpeg")

    except fal_client.FALServerException as e:
        print(f"❌ Fal AI Server Error: {e}")
        raise HTTPException(status_code=503, detail=f"The image generation service failed: {e}")
    except requests.RequestException as e:
        print(f"❌ Error downloading image: {e}")
        raise HTTPException(status_code=500, detail="Failed to download generated image.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
# =================================================================================
# === END OF REFACTORED ENDPOINT ==================================================
# =================================================================================


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

@app.get("/products")
async def get_products():
    """
    Fetch products from Paddle and transform them for frontend consumption.
    """
    try:
        print("🔍 Fetching all products and prices...")
        
        # Fetch all products and prices once
        all_products = list(paddle.products.list())
        all_prices = list(paddle.prices.list())
        
        print(f"✅ Found {len(all_products)} products and {len(all_prices)} prices")
        
        subscription_plans = []
        
        for product in all_products:
            prod_id = getattr(product, "id", None)
            prod_name = getattr(product, "name", "")
            prod_description = getattr(product, "description", "")
            prod_status = getattr(product, "status", "active")
            
            # Skip archived products
            if prod_status != "active" or not prod_id:
                continue
            
            # ✅ FIX: Handle nested custom_data
            def safe_dict(obj):
                if obj is None:
                    return {}
                if isinstance(obj, dict):
                    return obj
                try:
                    return vars(obj)
                except TypeError:
                    return getattr(obj, '__dict__', {})
            
            custom_data_raw = safe_dict(getattr(product, "custom_data", None))
            
            # ✅ CRITICAL: Extract nested 'data' key
            custom_data = custom_data_raw.get('data', custom_data_raw)
            
            print(f"📦 {prod_name}: custom_data = {custom_data}")
            
            # Determine product type
            product_type = custom_data.get("type", custom_data.get("planType", "subscription"))
            
            # ✅ Filter prices for this product
            product_prices = [p for p in all_prices if getattr(p, "product_id", None) == prod_id]
            print(f"   Found {len(product_prices)} prices")
            
            if len(product_prices) == 0:
                print(f"   ⚠️ No prices for {prod_name}, skipping")
                continue
            
            for price in product_prices:
                price_id = getattr(price, "id", None)
                price_status = getattr(price, "status", "active")
                
                if price_status != "active" or not price_id:
                    continue
                
                # Extract unit price
                unit_price = getattr(price, "unit_price", None)
                amount = getattr(unit_price, "amount", "0") if unit_price else "0"
                currency = getattr(unit_price, "currency_code", "USD") if unit_price else "USD"
                
                # Extract billing cycle
                billing_cycle = getattr(price, "billing_cycle", None)
                interval = None
                frequency = None
                if billing_cycle:
                    interval = getattr(billing_cycle, "interval", None)
                    frequency = getattr(billing_cycle, "frequency", 1)
                
                # Format price
                price_amount = float(amount) / 100
                currency_symbol = "$" if currency == "USD" else currency
                formatted_price = f"{currency_symbol}{price_amount:.2f}"
                
                # Get names
                price_name = getattr(price, "name", None) or prod_name
                price_description = getattr(price, "description", None) or prod_description
                
                # Handle price custom_data (also might be nested)
                price_custom_data_raw = safe_dict(getattr(price, "custom_data", None))
                price_custom_data = price_custom_data_raw.get('data', price_custom_data_raw)
                
                # Check if recommended
                is_recommended = (
                    custom_data.get("isRecommended", False) or 
                    price_custom_data.get("isRecommended", False)
                )
                
                # Detect if subscription (has billing interval)
                is_subscription = interval is not None
                
                # Build response
                plan_data = {
                    "id": price_id,
                    "productId": prod_id,
                    "name": prod_name,
                    "priceName": price_name,
                    "price": formatted_price,
                    "interval": interval,
                    "frequency": frequency,
                    "description": price_description,
                    "isRecommended": is_recommended,
                    "currency": currency,
                    "rawAmount": amount,
                }
                
                # Add type-specific fields
                if not is_subscription:
                    # Credit package
                    plan_data["type"] = "credits"
                    plan_data["credits"] = int(custom_data.get("amount", 0))
                    plan_data["isPopular"] = custom_data.get("isPopular", False)
                else:
                    # Subscription
                    plan_data["type"] = "subscription"
                
                subscription_plans.append(plan_data)
                print(f"   ✅ Added: {price_name} ({price_id})")
        
        # Sort: recommended first, then by name
        subscription_plans.sort(
            key=lambda x: (
                x.get("isRecommended") is not True,
                x.get("name", ""),
                x.get("interval", "") or "",
            )
        )
        
        print(f"✅ Returning {len(subscription_plans)} plans")
        return {"success": True, "data": subscription_plans}
        
    except ApiError as e:
        print(f"❌ Paddle API error: {e}")
        raise HTTPException(status_code=502, detail=f"Paddle API error: {str(e)}")
    except Exception as e:
        print(f"❌ Internal error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


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
