# server.py  (full file - all endpoints included)
import base64
import json
import os
import mimetypes
import uuid
import hmac
import hashlib
import time
from typing import Optional, List, Any, Dict
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request, Header, Response, Query
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import shutil
import requests
from io import BytesIO
from PIL import Image, ImageFilter

# --- NEW FAL AI IMPORT ---
import fal_client

# --- PADDLE & FIREBASE INTEGRATION (UPDATED TO paddle_billing) ---
# NOTE: ensure the installed package matches these imports (paddle_billing)
try:
    from paddle_billing import Client, Environment, Options
    from paddle_billing.Exceptions.ApiError import ApiError
    from paddle_billing.Notifications import Verifier, Secret
    from paddle_billing.Resources.Subscriptions.Operations import CancelSubscription
    from paddle_billing.Entities.Subscriptions import SubscriptionEffectiveFrom
except Exception as e:
    # Provide a clearer runtime error if paddle_billing is missing
    raise ImportError(
        "paddle_billing import failed. Ensure you installed the correct package (paddle-python-sdk / paddle_billing) "
        "and are running Python >= 3.11 if required. Underlying error: " + str(e)
    )

import firebase_admin
from firebase_admin import credentials, firestore, auth as firebase_auth
# --- END INTEGRATION IMPORTS ---


load_dotenv()

app = FastAPI(title="DreamAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://dreamai-web.vercel.app",  # Next.js 16 production deployment
        "https://dreamai-web-sooty.vercel.app",  # Next.js 16 alternate deployment
        "https://akmldsfmasdfmma.space",  # Custom domain deployment
        "https://dreamai-generator.vercel.app",  # Next.js production deployment
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
# === AUTHENTICATION HELPER FUNCTIONS =============================================
# =================================================================================
async def verify_firebase_token(authorization: Optional[str] = Header(None)) -> str:
    """
    Verify Firebase ID token from Authorization header.
    Returns the authenticated user's UID.
    Raises HTTPException 401 if token is invalid or missing.
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header. Please include 'Authorization: Bearer <token>'"
        )

    # Extract token from "Bearer <token>" format
    parts = authorization.split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header format. Expected 'Bearer <token>'"
        )

    id_token = parts[1]

    try:
        # Verify the ID token with Firebase Admin SDK
        decoded_token = firebase_auth.verify_id_token(id_token)
        uid = decoded_token.get("uid")

        if not uid:
            raise HTTPException(status_code=401, detail="Invalid token: missing UID")

        return uid

    except firebase_auth.ExpiredIdTokenError:
        raise HTTPException(status_code=401, detail="Token has expired. Please sign in again.")
    except firebase_auth.RevokedIdTokenError:
        raise HTTPException(status_code=401, detail="Token has been revoked. Please sign in again.")
    except firebase_auth.InvalidIdTokenError as e:
        print(f"❌ Invalid ID token: {e}")
        raise HTTPException(status_code=401, detail="Invalid authentication token.")
    except Exception as e:
        print(f"❌ Token verification error: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed.")


# =================================================================================
# === WATERMARK UTILITY FUNCTIONS =================================================
# =================================================================================
def add_stroke_to_watermark(watermark_rgba: Image.Image, stroke_width: int = 3) -> Image.Image:
    """
    Add a dark stroke/outline around a watermark to make it visible on any background.

    Args:
        watermark_rgba: The watermark image in RGBA mode
        stroke_width: Width of the stroke in pixels (default: 3)

    Returns:
        Watermark with dark stroke applied
    """
    # Extract alpha channel
    alpha = watermark_rgba.getchannel('A')

    # Dilate the alpha channel to create stroke
    stroke_alpha = alpha.copy()
    for _ in range(stroke_width):
        stroke_alpha = stroke_alpha.filter(ImageFilter.MaxFilter(3))

    # Create black stroke layer
    stroke = Image.new('RGBA', watermark_rgba.size, (0, 0, 0, 255))
    stroke.putalpha(stroke_alpha)

    # Composite: stroke (black background) behind watermark (white foreground)
    result = Image.alpha_composite(stroke, watermark_rgba)

    return result


def add_watermark(image_bytes: bytes, is_premium: bool = False, watermark_text: str = "DreamAI") -> bytes:
    """
    Add watermark(s) to an image based on user's premium status.

    Free users: Double watermark
      - watermark-1.png centered at 40% opacity
      - watermark-2.png in bottom right at 100% opacity

    Premium users: Single watermark
      - watermark-2.png in bottom right at 100% opacity

    Args:
        image_bytes: The original image as bytes
        is_premium: If True, only adds single watermark
        watermark_text: Not used (kept for backward compatibility)

    Returns:
        The watermarked image as bytes (JPEG format)
    """
    try:
        # Open the image from bytes
        img = Image.open(BytesIO(image_bytes))
        img_width, img_height = img.size

        # Convert base image to RGBA for compositing
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        print(f"[WATERMARK DEBUG] Image size: {img_width}x{img_height}")

        # Load watermark images
        watermark_dir = os.path.dirname(os.path.abspath(__file__))

        # FREE USERS: Add center watermark at 40% opacity
        if not is_premium:
            try:
                center_watermark_path = os.path.join(watermark_dir, "watermark-1.png")
                center_watermark = Image.open(center_watermark_path).convert('RGBA')
                print(f"[WATERMARK DEBUG] Center watermark loaded: {center_watermark.size}")

                # Scale watermark to be 56% of image width (reduced by 20% from 70%)
                scale_factor = (img_width * 0.56) / center_watermark.width
                new_width = int(center_watermark.width * scale_factor)
                new_height = int(center_watermark.height * scale_factor)
                center_watermark = center_watermark.resize((new_width, new_height), Image.Resampling.LANCZOS)

                print(f"[WATERMARK DEBUG] Center watermark resized to: {new_width}x{new_height}")

                # Apply 40% opacity while preserving transparency
                alpha = center_watermark.getchannel('A')
                alpha = alpha.point(lambda x: int(x * 0.4))
                center_watermark.putalpha(alpha)

                # Center position
                center_x = (img_width - new_width) // 2
                center_y = (img_height - new_height) // 2

                # Paste watermark
                img.paste(center_watermark, (center_x, center_y), center_watermark)
                print(f"[WATERMARK DEBUG] Center watermark applied at ({center_x}, {center_y})")

            except Exception as e:
                print(f"[WATERMARK ERROR] Failed to load center watermark: {e}")

        # BOTH FREE AND PREMIUM: Add bottom right watermark at 100% opacity
        try:
            bottom_watermark_path = os.path.join(watermark_dir, "watermark-2.png")
            bottom_watermark = Image.open(bottom_watermark_path).convert('RGBA')
            print(f"[WATERMARK DEBUG] Bottom watermark loaded: {bottom_watermark.size}")

            # Scale watermark to be 40% of image width (increased for visibility)
            scale_factor = (img_width * 0.4) / bottom_watermark.width
            new_width = int(bottom_watermark.width * scale_factor)
            new_height = int(bottom_watermark.height * scale_factor)
            bottom_watermark = bottom_watermark.resize((new_width, new_height), Image.Resampling.LANCZOS)

            print(f"[WATERMARK DEBUG] Bottom watermark resized to: {new_width}x{new_height}")

            # Bottom right position with padding
            padding = int(min(img_height, img_width) * 0.02)
            bottom_x = img_width - new_width - padding
            bottom_y = img_height - new_height - padding

            # Paste watermark
            img.paste(bottom_watermark, (bottom_x, bottom_y), bottom_watermark)
            print(f"[WATERMARK DEBUG] Bottom watermark applied at ({bottom_x}, {bottom_y})")

        except Exception as e:
            print(f"[WATERMARK ERROR] Failed to load bottom watermark: {e}")

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
    prompt: str = Form(...),
    authorization: Optional[str] = Header(None),
    # Legacy fields kept for backwards compatibility (ignored)
    userId: Optional[str] = Form(None),  # DEPRECATED: Use Authorization header
    image2: Optional[UploadFile] = File(None),
    temperature: float = Form(1.0),
    top_p: float = Form(0.95),
    top_k: int = Form(40),
):
    """
    Generate an image using Fal AI's FLUX Kontext model based on ONE input image and a prompt.
    AUTHENTICATION REQUIRED: Include 'Authorization: Bearer <firebase_id_token>' header.
    Downloads the generated image from fal.ai URL and adds "DreamAI" watermark.
    Deducts 1 credit from non-premium users using atomic transaction.
    """
    # Ensure an image was actually sent
    if not image1 or not image1.file:
        raise HTTPException(status_code=400, detail="An image file is required.")

    # SECURITY: Verify Firebase ID token and get authenticated user ID
    authenticated_user_id = await verify_firebase_token(authorization)
    print(f"🔐 Authenticated user: {authenticated_user_id}")

    # Check user's premium status and deduct credits atomically
    is_premium = False
    try:
        user_ref = db.collection('users').document(authenticated_user_id)

        # Use Firestore transaction to atomically check and deduct credits
        @firestore.transactional
        def deduct_credit_transaction(transaction, user_ref):
            user_doc = user_ref.get(transaction=transaction)

            if not user_doc.exists:
                raise HTTPException(status_code=404, detail="User not found")

            user_data = user_doc.to_dict()
            premium_status = user_data.get('premium_status')
            is_premium_user = premium_status == 'active'
            current_credits = user_data.get('credits', 0)

            # If not premium, check and deduct credits atomically
            if not is_premium_user:
                if current_credits < 1:
                    raise HTTPException(
                        status_code=402,  # Payment Required
                        detail="Insufficient credits. Please purchase more credits or upgrade to premium."
                    )

                # Atomically deduct 1 credit within the transaction
                transaction.update(user_ref, {
                    "credits": current_credits - 1
                })
                print(f"💳 Deducted 1 credit from user {authenticated_user_id}. Remaining: {current_credits - 1}")
            else:
                print(f"👑 Premium user {authenticated_user_id} - no credit deduction")

            return is_premium_user

        # Execute the transaction
        transaction = db.transaction()
        is_premium = deduct_credit_transaction(transaction, user_ref)

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error checking user credits: {e}")
        raise HTTPException(status_code=500, detail="Failed to verify user credits")

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
        # Call the Fal AI FLUX Kontext [pro] model with image_url parameter
        # Official docs: https://fal.ai/models/fal-ai/flux-pro/kontext/api
        result = fal_client.run(
            "fal-ai/flux-pro/kontext",
            arguments={
                "prompt": prompt,
                "image_url": uploaded_image_url,
                "guidance_scale": 3.5,  # CFG scale (1-20, default: 3.5)
                "num_inference_steps": 28,  # Generation iterations (1-50, default: 28)
                "num_images": 1,  # Number of images to generate (1-4)
                "output_format": "jpeg",  # Output format: "jpeg" or "png"
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
        # Premium users: single watermark ("AI Generated by DreamAI" bottom right)
        # Free users: double watermark (center "DreamAI" + bottom right text)
        if is_premium:
            print("👑 Premium user - adding single watermark")
        else:
            print("🏷️  Free user - adding double watermark")

        watermarked_image_bytes = add_watermark(image_content_bytes, is_premium, "DreamAI")

        print(f"✅ Returning image ({len(watermarked_image_bytes)} bytes)")
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
async def create_checkout(
    request: Request,
    authorization: Optional[str] = Header(None)
):
    """
    Create a Paddle checkout session for subscription or credit purchase.
    AUTHENTICATION REQUIRED: Include 'Authorization: Bearer <firebase_id_token>' header.

    Trial Prevention:
    - Checks user's hasUsedTrial flag
    - Checks deleted_accounts collection by email
    - Checks trial_blocked_devices collection by device_id
    """
    # SECURITY: Verify Firebase ID token and get authenticated user ID
    authenticated_user_id = await verify_firebase_token(authorization)
    print(f"🔐 Authenticated user for checkout: {authenticated_user_id}")

    body = await request.json()
    price_id = body.get('priceId')
    email = body.get('email')  # Optional: pre-fill customer email
    device_id = body.get('deviceId')  # Device fingerprint for trial abuse prevention

    # userId from body is now ignored - we use the authenticated user ID
    # This prevents users from creating checkouts for other users

    if not price_id:
        raise HTTPException(
            status_code=400,
            detail="priceId is required"
        )

    try:
        # Get user document from Firebase using authenticated user ID
        user_doc = db.collection('users').document(authenticated_user_id).get()
        user_data = user_doc.to_dict() if user_doc.exists else {}
        paddle_customer_id = user_data.get('paddle_customer_id')
        has_used_trial = user_data.get('hasUsedTrial', False)

        # If no email provided, try to get from Firebase
        if not email and user_data:
            email = user_data.get('email')

        # Check if price has a trial period and if user is eligible
        try:
            price = paddle.prices.get(price_id)
            trial_period = getattr(price, 'trial_period', None)

            if trial_period:
                # Build detailed block reason for better error messages
                block_reason = None

                # Check 1: User's hasUsedTrial flag
                if has_used_trial:
                    block_reason = "email"
                    print(f"❌ User {authenticated_user_id} already used free trial (hasUsedTrial flag)")

                # Check 2: Deleted accounts by email
                if not block_reason and email:
                    email_normalized = email.lower().strip()
                    archive_doc = db.collection('deleted_accounts').document(email_normalized).get()
                    if archive_doc.exists:
                        archive_data = archive_doc.to_dict()
                        if archive_data.get('hasUsedTrial', False):
                            block_reason = "email"
                            print(f"❌ Email {email_normalized} found in deleted_accounts with trial history")
                            # Update current user's hasUsedTrial flag
                            db.collection('users').document(authenticated_user_id).update({
                                "hasUsedTrial": True
                            })

                # Check 3: Device ID (if provided)
                if not block_reason and device_id:
                    device_doc = db.collection('trial_blocked_devices').document(device_id).get()
                    if device_doc.exists:
                        device_data = device_doc.to_dict()
                        if device_data.get('blocked', False):
                            block_reason = "device"
                            print(f"❌ Device {device_id} found in blocked devices")
                            # Also mark user as having used trial
                            db.collection('users').document(authenticated_user_id).update({
                                "hasUsedTrial": True,
                                "trialBlockedByDevice": device_id,
                            })

                # If blocked, return appropriate error
                if block_reason:
                    if block_reason == "device":
                        error_message = (
                            "Free trial not available. This device has already been used for a free trial. "
                            "Please select a paid plan to continue."
                        )
                    else:  # email
                        error_message = (
                            "Free trial not available. This email address has already been used for a free trial. "
                            "Please select a paid plan or use a different account."
                        )

                    raise HTTPException(
                        status_code=403,
                        detail={
                            "message": error_message,
                            "code": "TRIAL_NOT_AVAILABLE",
                            "reason": block_reason,
                        }
                    )

        except HTTPException:
            raise
        except Exception as price_check_error:
            # Don't block checkout if price check fails - just log it
            print(f"⚠️ Could not check price trial period: {price_check_error}")

        # Build transaction payload
        txn_payload = {
            "items": [
                {"price_id": price_id, "quantity": 1}
            ],
            "custom_data": {"firebase_uid": authenticated_user_id},
            "success_url": "https://dreamai-checkpoint.netlify.app/payment-success",
        }

        # Add customer_id if exists (for returning customers)
        if paddle_customer_id:
            txn_payload["customer_id"] = paddle_customer_id

        # Add email if available (for new customers)
        if email and not paddle_customer_id:
            txn_payload["customer_email"] = email

        print(f"Creating checkout for user {authenticated_user_id} with price {price_id}")
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
async def create_customer_portal(
    request: Request,
    authorization: Optional[str] = Header(None)
):
    """
    Create a Paddle customer portal link for subscription management.
    AUTHENTICATION REQUIRED: Include 'Authorization: Bearer <firebase_id_token>' header.
    """
    # SECURITY: Verify Firebase ID token and get authenticated user ID
    authenticated_user_id = await verify_firebase_token(authorization)
    print(f"🔐 Authenticated user for customer portal: {authenticated_user_id}")

    # userId from body is now ignored - we use the authenticated user ID
    user_doc = db.collection('users').document(authenticated_user_id).get()
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

@app.post("/cancel-subscription")
async def cancel_subscription(
    request: Request,
    authorization: Optional[str] = Header(None)
):
    """
    Cancel a user's Paddle subscription.
    Called when user deletes their account.
    AUTHENTICATION REQUIRED: Include 'Authorization: Bearer <firebase_id_token>' header.
    """
    # SECURITY: Verify Firebase ID token and get authenticated user ID
    authenticated_user_id = await verify_firebase_token(authorization)
    print(f"🔐 Authenticated user for subscription cancellation: {authenticated_user_id}")

    # Get user's subscription from Firestore (don't trust client-provided subscription_id)
    user_doc = db.collection('users').document(authenticated_user_id).get()
    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="User not found")

    user_data = user_doc.to_dict()
    subscription_id = user_data.get('subscription_id')

    if not subscription_id:
        # No subscription to cancel - this is fine
        return {"success": True, "message": "No subscription to cancel"}

    print(f"🔄 Canceling subscription {subscription_id} for user {authenticated_user_id}")

    user_ref = db.collection('users').document(authenticated_user_id)

    try:
        # First, fetch the subscription from Paddle to check its current status
        subscription = paddle.subscriptions.get(subscription_id=subscription_id)
        current_status = subscription.status if subscription else None
        scheduled_change = getattr(subscription, 'scheduled_change', None)

        print(f"📋 Current subscription status: {current_status}, scheduled_change: {scheduled_change}")

        # Check if already cancelled - these states don't need/allow cancellation
        # Note: past_due and paused CAN be cancelled, so we don't include them here
        already_cancelled_states = ["canceled", "cancelled"]
        if current_status and str(current_status).lower() in already_cancelled_states:
            print(f"ℹ️ Subscription {subscription_id} is already {current_status}, treating as success")

            # Update Firestore to ensure it reflects the cancelled state
            user_ref.update({
                "subscription_status": "canceled",
                "isPremium": False,
                "premium_status": None,
                "subscription_canceled_at": firestore.SERVER_TIMESTAMP,
            })

            return {"success": True, "message": f"Subscription already {current_status}"}

        # Check if subscription is already scheduled to cancel
        # In this case, we want to cancel immediately instead of waiting
        if scheduled_change:
            scheduled_action = getattr(scheduled_change, 'action', None)
            if scheduled_action and str(scheduled_action).lower() == "cancel":
                print(f"ℹ️ Subscription {subscription_id} is scheduled to cancel, proceeding with immediate cancellation")

        # Cancel the subscription immediately via Paddle API
        result = paddle.subscriptions.cancel(
            subscription_id,
            CancelSubscription(effective_from=SubscriptionEffectiveFrom.Immediately)
        )

        print(f"✅ Subscription {subscription_id} cancelled successfully")

        # Update Firestore to reflect cancellation (webhook will also do this, but do it now for immediate feedback)
        user_ref.update({
            "subscription_status": "canceled",
            "isPremium": False,
            "premium_status": None,
            "subscription_canceled_at": firestore.SERVER_TIMESTAMP,
        })

        return {"success": True, "message": "Subscription cancelled successfully"}

    except ApiError as e:
        error_msg = getattr(e, 'message', str(e))
        error_code = getattr(e, 'code', None)
        print(f"❌ Paddle API error cancelling subscription: {error_msg} (code: {error_code})")

        # Handle specific error cases
        # If subscription not found or already cancelled, treat as success for account deletion flow
        error_lower = str(error_msg).lower()
        if "not found" in error_lower or "canceled" in error_lower or "cancelled" in error_lower:
            print(f"ℹ️ Treating error as success for account deletion: {error_msg}")
            user_ref.update({
                "subscription_status": "canceled",
                "isPremium": False,
                "premium_status": None,
                "subscription_id": None,
            })
            return {"success": True, "message": "Subscription cleared"}

        return {"success": False, "error": f"Failed to cancel subscription: {error_msg}"}

    except Exception as e:
        print(f"❌ Error cancelling subscription: {e}")
        error_str = str(e).lower()

        # Handle edge cases where subscription might already be gone
        if "not found" in error_str:
            user_ref.update({
                "subscription_status": "canceled",
                "isPremium": False,
                "premium_status": None,
                "subscription_id": None,
            })
            return {"success": True, "message": "Subscription cleared"}

        return {"success": False, "error": f"An error occurred: {str(e)}"}


@app.post("/archive-deleted-user")
async def archive_deleted_user(
    request: Request,
    authorization: Optional[str] = Header(None)
):
    """
    Archive user data before account deletion.
    Saves credits and trial usage to deleted_accounts collection for future restoration.
    AUTHENTICATION REQUIRED: Include 'Authorization: Bearer <firebase_id_token>' header.
    """
    # SECURITY: Verify Firebase ID token and get authenticated user ID
    authenticated_user_id = await verify_firebase_token(authorization)
    print(f"🔐 Archiving data for user: {authenticated_user_id}")

    try:
        # Get user data from Firestore
        user_ref = db.collection('users').document(authenticated_user_id)
        user_doc = user_ref.get()

        if not user_doc.exists:
            return {"success": True, "message": "No user data to archive"}

        user_data = user_doc.to_dict()
        email = user_data.get('email')

        if not email:
            print(f"⚠️ User {authenticated_user_id} has no email, skipping archive")
            return {"success": True, "message": "No email to archive"}

        # Normalize email for consistent lookups
        email_normalized = email.lower().strip()

        # Check if user ever had a trial - be precise to avoid blocking credit-only purchasers
        subscription_status = user_data.get('subscription_status', '')
        has_used_trial = (
            user_data.get('hasUsedTrial', False) or  # Explicit flag from webhook
            user_data.get('trialStartedAt') is not None or  # Explicit trial marker
            subscription_status == 'trialing' or  # Currently in trial
            subscription_status in ['active', 'canceled', 'paused']  # Had an active subscription (not just credits)
        )

        # Get current credits and device ID
        credits = user_data.get('credits', 0)
        device_id = user_data.get('deviceId')

        # Archive data to deleted_accounts collection (keyed by normalized email)
        archive_ref = db.collection('deleted_accounts').document(email_normalized)
        existing_archive = archive_ref.get()

        if existing_archive.exists:
            # User has deleted account before - update with new data
            existing_data = existing_archive.to_dict()
            # Keep the highest credit count (in case they had more before)
            existing_credits = existing_data.get('credits', 0)
            # Always mark as hasUsedTrial if they ever had one
            existing_trial = existing_data.get('hasUsedTrial', False)

            update_data = {
                "credits": max(credits, existing_credits),
                "hasUsedTrial": has_used_trial or existing_trial,
                "lastDeletedAt": firestore.SERVER_TIMESTAMP,
                "lastDeletedUid": authenticated_user_id,
                "paddle_customer_id": user_data.get('paddle_customer_id') or existing_data.get('paddle_customer_id'),
                "deletionCount": firestore.Increment(1),
            }

            # Add device ID to the list of associated devices
            if device_id:
                update_data["deviceIds"] = firestore.ArrayUnion([device_id])

            archive_ref.update(update_data)
            print(f"📦 Updated archive for {email_normalized}: credits={max(credits, existing_credits)}, hasUsedTrial={has_used_trial or existing_trial}")
        else:
            # First time deletion - create archive
            archive_data = {
                "email": email_normalized,
                "credits": credits,
                "hasUsedTrial": has_used_trial,
                "firstDeletedAt": firestore.SERVER_TIMESTAMP,
                "lastDeletedAt": firestore.SERVER_TIMESTAMP,
                "lastDeletedUid": authenticated_user_id,
                "paddle_customer_id": user_data.get('paddle_customer_id'),
                "deletionCount": 1,
            }

            # Store device ID
            if device_id:
                archive_data["deviceIds"] = [device_id]

            archive_ref.set(archive_data)
            print(f"📦 Created archive for {email_normalized}: credits={credits}, hasUsedTrial={has_used_trial}")

        # If user had used trial, also block their device
        if has_used_trial and device_id:
            db.collection('trial_blocked_devices').document(device_id).set({
                "blocked": True,
                "blockedAt": firestore.SERVER_TIMESTAMP,
                "blockedByEmail": email_normalized,
                "blockedReason": "account_deletion_with_trial",
            }, merge=True)
            print(f"🚫 Blocked device {device_id} due to account deletion with trial history")

        return {
            "success": True,
            "message": "User data archived successfully",
            "archived": {
                "credits": credits,
                "hasUsedTrial": has_used_trial
            }
        }

    except Exception as e:
        print(f"❌ Error archiving user data: {e}")
        import traceback
        traceback.print_exc()
        # Don't block deletion if archive fails - just log it
        return {"success": False, "error": str(e)}


@app.post("/check-deleted-account")
async def check_deleted_account(
    request: Request,
    authorization: Optional[str] = Header(None)
):
    """
    Check if user's email has a deleted account with credits/trial history.
    Called after account creation to restore credits.
    AUTHENTICATION REQUIRED: Include 'Authorization: Bearer <firebase_id_token>' header.

    IMPORTANT: This endpoint REPLACES the default 5 credits with archived credits,
    NOT adds to them. This prevents credit duplication exploits.
    """
    # SECURITY: Verify Firebase ID token and get authenticated user ID
    authenticated_user_id = await verify_firebase_token(authorization)
    print(f"🔍 Checking deleted account for user: {authenticated_user_id}")

    try:
        # Get current user's email from Firestore
        user_ref = db.collection('users').document(authenticated_user_id)
        user_doc = user_ref.get()

        if not user_doc.exists:
            return {"success": False, "error": "User not found"}

        user_data = user_doc.to_dict()
        email = user_data.get('email')
        current_credits = user_data.get('credits', 0)

        # Check if restoration already happened (prevent multiple restorations)
        if user_data.get('restoredFromDeletedAccount'):
            print(f"⚠️ User {authenticated_user_id} already restored from deleted account, skipping")
            return {"success": True, "found": False, "message": "Already restored", "skipped": True}

        if not email:
            return {"success": True, "found": False, "message": "No email on account"}

        # Normalize email for lookup
        email_normalized = email.lower().strip()

        # Check deleted_accounts collection
        archive_ref = db.collection('deleted_accounts').document(email_normalized)
        archive_doc = archive_ref.get()

        if not archive_doc.exists:
            return {"success": True, "found": False, "message": "No previous account found"}

        archive_data = archive_doc.to_dict()
        archived_credits = archive_data.get('credits', 0)
        has_used_trial = archive_data.get('hasUsedTrial', False)

        # CRITICAL: Check if archive was already restored (prevents credit duplication)
        if archive_data.get('restoredToUid') and archive_data.get('credits', 0) == 0:
            print(f"⚠️ Archive for {email_normalized} already restored, only setting trial flag")
            # Only update the trial flag, don't give any credits
            user_ref.update({
                "hasUsedTrial": has_used_trial,
                "restoredFromDeletedAccount": True,
                "restoredAt": firestore.SERVER_TIMESTAMP,
            })
            return {
                "success": True,
                "found": True,
                "restored": {"credits": 0, "hasUsedTrial": has_used_trial},
                "message": "Trial flag restored, credits already claimed"
            }

        print(f"📦 Found archived account for {email_normalized}: credits={archived_credits}, hasUsedTrial={has_used_trial}")

        # Returning user: SET to exactly their archived credits (no bonus)
        # This prevents credit exploitation from delete/recreate cycles
        final_credits = archived_credits
        print(f"   Returning user. Setting to {final_credits} archived credits")

        # Restore credits (SET, not INCREMENT) and set trial flag on current user
        update_data = {
            "credits": final_credits,  # SET to final value, not INCREMENT
            "hasUsedTrial": has_used_trial,
            "restoredFromDeletedAccount": True,
            "restoredAt": firestore.SERVER_TIMESTAMP,
            "previousArchivedCredits": archived_credits,  # For audit trail
        }

        user_ref.update(update_data)

        # Mark archive as restored and zero out credits (prevents re-claiming)
        archive_ref.update({
            "restoredToUid": authenticated_user_id,
            "restoredAt": firestore.SERVER_TIMESTAMP,
            "credits": 0,  # Zero out credits after restoration - CRITICAL
        })

        credits_restored = final_credits - 5 if current_credits == 5 else 0
        print(f"✅ Set credits to {final_credits} and trial flag for {email_normalized} (net change: {credits_restored})")

        return {
            "success": True,
            "found": True,
            "restored": {
                "credits": credits_restored,  # Report net credits added (can be negative)
                "finalCredits": final_credits,
                "hasUsedTrial": has_used_trial
            }
        }

    except Exception as e:
        print(f"❌ Error checking deleted account: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


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
                
                # Get names - use price name if available, fallback to product name
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

                # Build response - use price_name as the main name to show different price names
                plan_data = {
                    "id": price_id,
                    "productId": prod_id,
                    "name": price_name,  # Use price name instead of product name
                    "productName": prod_name,  # Keep product name for reference
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

                    # Try multiple field names for flexibility
                    # Check both product-level and price-level custom_data
                    # Paddle custom data may use: amount, credit_amount, credits, or credit_count
                    credits = (
                        # Try product custom_data first
                        custom_data.get("amount") or
                        custom_data.get("credit_amount") or
                        custom_data.get("credits") or
                        custom_data.get("credit_count") or
                        # Then try price custom_data
                        price_custom_data.get("amount") or
                        price_custom_data.get("credit_amount") or
                        price_custom_data.get("credits") or
                        price_custom_data.get("credit_count") or
                        0
                    )
                    plan_data["credits"] = int(credits) if credits else 0

                    # Check isPopular from both sources
                    plan_data["isPopular"] = (
                        custom_data.get("isPopular", False) or
                        price_custom_data.get("isPopular", False)
                    )

                    # Debug logging
                    print(f"   💰 Credit package: {credits} credits")
                    print(f"      Product custom_data: {custom_data}")
                    print(f"      Price custom_data: {price_custom_data}")
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


def verify_paddle_webhook_signature(paddle_signature: str, raw_body: str, webhook_secret: str) -> bool:
    """
    Manually verify Paddle webhook signature using HMAC-SHA256.

    Paddle signature format: "ts=1234567890;h1=abc123..."

    Args:
        paddle_signature: Value from Paddle-Signature header
        raw_body: Raw request body as string
        webhook_secret: Webhook secret from Paddle dashboard

    Returns:
        True if signature is valid, False otherwise
    """
    try:
        # Parse signature header to extract timestamp and signature
        parts = paddle_signature.split(";")
        timestamp = None
        signature = None

        for part in parts:
            if part.startswith("ts="):
                timestamp = part.split("=", 1)[1]
            elif part.startswith("h1="):
                signature = part.split("=", 1)[1]

        if not timestamp or not signature:
            print("❌ Invalid signature format: missing timestamp or signature")
            return False

        # Verify timestamp to prevent replay attacks (5 minutes tolerance)
        current_time = int(time.time())
        sig_time = int(timestamp)
        time_diff = abs(current_time - sig_time)

        if time_diff > 300:  # 5 minutes = 300 seconds
            print(f"❌ Signature timestamp too old: {time_diff} seconds difference")
            return False

        # Debug logging
        print(f"🔍 Debug Info:")
        print(f"   Timestamp: {timestamp}")
        print(f"   Body length: {len(raw_body)} bytes")
        print(f"   Body preview: {raw_body[:200]}...")
        print(f"   Webhook secret length: {len(webhook_secret)}")

        # Create payload: timestamp:body
        payload = f"{timestamp}:{raw_body}"

        # Generate HMAC-SHA256 signature
        computed_signature = hmac.new(
            webhook_secret.encode('utf-8'),
            msg=payload.encode('utf-8'),
            digestmod=hashlib.sha256
        ).hexdigest()

        # Compare signatures using constant-time comparison
        is_valid = hmac.compare_digest(signature, computed_signature)

        if not is_valid:
            print("❌ Signature mismatch")
            print(f"   Expected: {computed_signature}")
            print(f"   Received: {signature}")
            print(f"   Payload format: ts={timestamp}:body({len(raw_body)} bytes)")

        return is_valid

    except Exception as e:
        print(f"❌ Signature verification error: {e}")
        import traceback
        traceback.print_exc()
        return False


@app.post("/paddle-webhook")
async def paddle_webhook(
    request: Request,
    paddle_signature: Optional[str] = Header(None, alias="Paddle-Signature")
):
    """
    Handle Paddle webhook notifications.
    Updates user subscription status and credits in Firestore.

    Events handled:
    - subscription.created: Activates premium subscription
    - subscription.updated: Updates subscription status
    - subscription.canceled: Removes premium status
    - transaction.completed: Adds credits for one-time purchases
    """
    try:
        # Get webhook secret from environment
        webhook_secret = os.environ.get("PADDLE_WEBHOOK_SECRET")

        if not webhook_secret:
            print("❌ PADDLE_WEBHOOK_SECRET not configured")
            raise HTTPException(status_code=500, detail="Webhook secret not configured")

        # Get raw body bytes BEFORE parsing JSON (required for signature verification)
        raw_body_bytes = await request.body()
        raw_body_str = raw_body_bytes.decode('utf-8')

        # Verify webhook signature using manual HMAC verification
        print("🔐 Verifying webhook signature...")

        if not paddle_signature:
            print("❌ Missing Paddle-Signature header")
            raise HTTPException(status_code=400, detail="Missing webhook signature")

        # Use our manual verification function (compatible with FastAPI)
        is_valid = verify_paddle_webhook_signature(paddle_signature, raw_body_str, webhook_secret)

        if not is_valid:
            print("❌ Webhook signature verification failed")
            raise HTTPException(status_code=400, detail="Invalid webhook signature")

        print("✅ Webhook signature verified")

        # Parse webhook body from raw string
        body = json.loads(raw_body_str)
        event_type = body.get("event_type")
        data = body.get("data", {})

        print(f"📥 Webhook received: {event_type}")
        print(f"📦 Data: {json.dumps(data, indent=2)}")

        # Extract custom_data to get Firebase UID (available in most events)
        custom_data = data.get("custom_data", {})
        firebase_uid = custom_data.get("firebase_uid")

        if not firebase_uid:
            print("⚠️ No firebase_uid in custom_data, checking transaction items...")
            # For some events, custom_data might be nested in items
            items = data.get("items", [])
            for item in items:
                item_custom_data = item.get("custom_data", {})
                if item_custom_data.get("firebase_uid"):
                    firebase_uid = item_custom_data.get("firebase_uid")
                    break

        if not firebase_uid:
            print("⚠️ Warning: No firebase_uid found in webhook data")
            return {"status": "ok", "message": "No firebase_uid to process"}

        # Handle subscription.created
        if event_type == "subscription.created":
            subscription_id = data.get("id")
            customer_id = data.get("customer_id")
            status = data.get("status")

            # Extract price_id from subscription items
            price_id = None
            items = data.get("items", [])
            if items and len(items) > 0:
                price_id = items[0].get("price", {}).get("id") if isinstance(items[0].get("price"), dict) else items[0].get("price_id")

            print(f"🎉 Creating subscription for user {firebase_uid} (price_id: {price_id})")

            # Determine if subscription grants premium access
            is_premium = status in ["active", "trialing"]
            premium_status = "active" if is_premium else None

            # Mark hasUsedTrial if they started a trial (prevents future trial abuse)
            is_trialing = status == "trialing"

            user_ref = db.collection('users').document(firebase_uid)
            update_data = {
                "subscription_id": subscription_id,
                "paddle_customer_id": customer_id,
                "subscription_status": status,
                "isPremium": is_premium,
                "premium_status": premium_status,
                "subscription_created_at": firestore.SERVER_TIMESTAMP,
            }

            # Mark trial usage to prevent future trial abuse
            if is_trialing:
                update_data["hasUsedTrial"] = True
                update_data["trialStartedAt"] = firestore.SERVER_TIMESTAMP
                print(f"🏷️ Marking user {firebase_uid} as hasUsedTrial=True")

                # Block the device ID if it was stored on the user
                user_doc = user_ref.get()
                if user_doc.exists:
                    user_data = user_doc.to_dict()
                    device_id = user_data.get('deviceId')
                    user_email = user_data.get('email')

                    if device_id:
                        # Block this device from future trials
                        db.collection('trial_blocked_devices').document(device_id).set({
                            "blocked": True,
                            "blockedAt": firestore.SERVER_TIMESTAMP,
                            "blockedByUid": firebase_uid,
                            "blockedByEmail": user_email,
                            "subscriptionId": subscription_id,
                        }, merge=True)
                        print(f"🚫 Blocked device {device_id} from future trials")

                    # Also update deleted_accounts if email exists
                    if user_email:
                        email_normalized = user_email.lower().strip()
                        db.collection('deleted_accounts').document(email_normalized).set({
                            "hasUsedTrial": True,
                            "trialUsedAt": firestore.SERVER_TIMESTAMP,
                            "email": email_normalized,
                        }, merge=True)
                        print(f"📧 Marked email {email_normalized} as trial used in deleted_accounts")

            # Add price_id if available
            if price_id:
                update_data["subscription_price_id"] = price_id

            user_ref.update(update_data)

            print(f"✅ Updated user {firebase_uid} with subscription {subscription_id} (isPremium: {is_premium}, price_id: {price_id}, trialing: {is_trialing})")

        # Handle subscription.updated
        elif event_type == "subscription.updated":
            subscription_id = data.get("id")
            status = data.get("status")
            scheduled_change = data.get("scheduled_change")

            # Extract price_id from subscription items
            price_id = None
            items = data.get("items", [])
            if items and len(items) > 0:
                price_id = items[0].get("price", {}).get("id") if isinstance(items[0].get("price"), dict) else items[0].get("price_id")

            print(f"🔄 Updating subscription for user {firebase_uid}: {status} (price_id: {price_id})")

            update_data = {
                "subscription_status": status,
                "subscription_updated_at": firestore.SERVER_TIMESTAMP,
            }

            # Add price_id if available
            if price_id:
                update_data["subscription_price_id"] = price_id

            # Set isPremium and premium_status based on subscription status
            if status in ["active", "trialing"]:
                update_data["isPremium"] = True
                update_data["premium_status"] = "active"
            elif status in ["paused", "past_due"]:
                update_data["isPremium"] = False
                update_data["premium_status"] = "paused"
            elif status in ["canceled", "deleted"]:
                update_data["isPremium"] = False
                update_data["premium_status"] = None

            # Store scheduled change info if present
            if scheduled_change:
                update_data["scheduled_change"] = scheduled_change

            user_ref = db.collection('users').document(firebase_uid)
            user_ref.update(update_data)

            print(f"✅ Updated subscription status for {firebase_uid}: {status} (isPremium: {update_data.get('isPremium', False)}, price_id: {price_id})")

        # Handle subscription.canceled
        elif event_type == "subscription.canceled":
            subscription_id = data.get("id")
            canceled_at = data.get("canceled_at")

            print(f"❌ Canceling subscription for user {firebase_uid}")

            user_ref = db.collection('users').document(firebase_uid)
            user_ref.update({
                "subscription_status": "canceled",
                "isPremium": False,
                "premium_status": None,
                "subscription_canceled_at": canceled_at or firestore.SERVER_TIMESTAMP,
            })

            print(f"✅ Canceled subscription for {firebase_uid} (isPremium: False)")

        # Handle subscription.paused
        elif event_type == "subscription.paused":
            subscription_id = data.get("id")
            paused_at = data.get("paused_at")

            print(f"⏸️ Pausing subscription for user {firebase_uid}")

            user_ref = db.collection('users').document(firebase_uid)
            user_ref.update({
                "subscription_status": "paused",
                "isPremium": False,
                "premium_status": "paused",
                "subscription_paused_at": paused_at or firestore.SERVER_TIMESTAMP,
            })

            print(f"✅ Paused subscription for {firebase_uid} (isPremium: False)")

        # Handle subscription.resumed
        elif event_type == "subscription.resumed":
            subscription_id = data.get("id")
            status = data.get("status")

            print(f"▶️ Resuming subscription for user {firebase_uid}")

            user_ref = db.collection('users').document(firebase_uid)
            user_ref.update({
                "subscription_status": status,
                "isPremium": True,
                "premium_status": "active",
                "subscription_resumed_at": firestore.SERVER_TIMESTAMP,
            })

            print(f"✅ Resumed subscription for {firebase_uid} (isPremium: True)")

        # Handle transaction.completed (for credit purchases and subscription payments)
        elif event_type == "transaction.completed":
            transaction_id = data.get("id")
            items = data.get("items", [])
            status = data.get("status")

            print(f"💳 Transaction completed for user {firebase_uid}: {transaction_id}")
            print(f"📦 Full transaction data: {json.dumps(data, indent=2, default=str)}")

            # Process each item in the transaction
            total_credits_added = 0

            for item in items:
                price = item.get("price", {})
                product = price.get("product", {})

                # Helper function to safely convert objects to dicts
                def safe_dict(obj):
                    if obj is None:
                        return {}
                    if isinstance(obj, dict):
                        return obj
                    try:
                        return vars(obj)
                    except TypeError:
                        return getattr(obj, '__dict__', {})

                # Extract and handle nested custom_data for BOTH product and price
                # Paddle may nest custom_data under a 'data' key
                product_custom_data_raw = safe_dict(product.get("custom_data", {}))
                product_custom_data = product_custom_data_raw.get('data', product_custom_data_raw)

                price_custom_data_raw = safe_dict(price.get("custom_data", {}))
                price_custom_data = price_custom_data_raw.get('data', price_custom_data_raw)

                # Debug: Log what we received from both sources
                print(f"   🔍 Product custom_data: {product_custom_data}")
                print(f"   🔍 Price custom_data: {price_custom_data}")

                # Extract credits amount - try BOTH product and price custom_data
                # Check multiple field names for flexibility
                credits = 0
                if isinstance(product_custom_data, dict) or isinstance(price_custom_data, dict):
                    credits_raw = (
                        # Try product custom_data first
                        product_custom_data.get("amount") or
                        product_custom_data.get("credit_amount") or
                        product_custom_data.get("credits") or
                        product_custom_data.get("credit_count") or
                        # Then try price custom_data (CRITICAL FIX - was missing)
                        price_custom_data.get("amount") or
                        price_custom_data.get("credit_amount") or
                        price_custom_data.get("credits") or
                        price_custom_data.get("credit_count") or
                        0
                    )

                    # Convert to int immediately with proper error handling
                    # This handles cases where Paddle returns "40" as string instead of 40
                    print(f"   🔍 Raw credits value: {credits_raw} (type: {type(credits_raw).__name__})")
                    try:
                        credits = int(credits_raw) if credits_raw else 0
                        print(f"   ✅ Converted to integer: {credits}")
                    except (ValueError, TypeError) as e:
                        print(f"   ⚠️ Failed to convert credits value '{credits_raw}': {e}")
                        credits = 0

                if credits > 0:
                    total_credits_added += credits
                    print(f"   💰 Found credit package: {credits} credits")

            # Add credits to user if any were purchased
            if total_credits_added > 0:
                print(f"📝 Preparing to add {total_credits_added} credits to user {firebase_uid}")
                user_ref = db.collection('users').document(firebase_uid)

                # Use Firestore increment to safely add credits
                print(f"   🔄 Executing Firestore update with Increment({total_credits_added})")
                user_ref.update({
                    "credits": firestore.Increment(total_credits_added)
                })

                print(f"✅ Successfully added {total_credits_added} credits to user {firebase_uid}")

                # Log transaction in user's history
                user_ref.collection('transactions').document(transaction_id).set({
                    "transaction_id": transaction_id,
                    "type": "credit_purchase",
                    "credits_added": total_credits_added,
                    "status": status,
                    "created_at": firestore.SERVER_TIMESTAMP,
                })
            else:
                print(f"   ℹ️ No credits found in transaction (likely subscription payment)")

        # Handle transaction.paid (alternative event name in some cases)
        elif event_type == "transaction.paid":
            print(f"✅ Transaction paid event received for user {firebase_uid}")
            # Similar logic to transaction.completed if needed
            # Usually transaction.completed is sufficient

        # Log unhandled events for monitoring
        else:
            print(f"ℹ️ Unhandled event type: {event_type}")
            print(f"   Data: {json.dumps(data, indent=2)}")

        return {
            "status": "ok",
            "event_type": event_type,
            "processed": True
        }

    except HTTPException:
        # Re-raise HTTP exceptions (signature failures, config errors)
        raise

    except ApiError as e:
        error_msg = getattr(e, 'message', str(e))
        print(f"❌ Paddle API error in webhook: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Paddle API error: {error_msg}")

    except Exception as e:
        print(f"❌ Webhook processing error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/register-device")
async def register_device(
    request: Request,
    authorization: Optional[str] = Header(None)
):
    """
    Register a device ID with the current user.
    Called on app launch to enable device-based trial prevention.
    AUTHENTICATION REQUIRED: Include 'Authorization: Bearer <firebase_id_token>' header.
    """
    authenticated_user_id = await verify_firebase_token(authorization)

    body = await request.json()
    device_id = body.get('deviceId')

    if not device_id:
        raise HTTPException(status_code=400, detail="deviceId is required")

    print(f"📱 Registering device {device_id} for user {authenticated_user_id}")

    try:
        user_ref = db.collection('users').document(authenticated_user_id)
        user_doc = user_ref.get()

        if not user_doc.exists:
            raise HTTPException(status_code=404, detail="User not found")

        user_data = user_doc.to_dict()
        email = user_data.get('email', '').lower().strip()

        # Update user with device ID
        user_ref.update({
            "deviceId": device_id,
            "deviceRegisteredAt": firestore.SERVER_TIMESTAMP,
        })

        # Check if this device is already blocked
        device_doc = db.collection('trial_blocked_devices').document(device_id).get()
        is_device_blocked = device_doc.exists and device_doc.to_dict().get('blocked', False)

        # Check if email is blocked (from deleted accounts)
        email_blocked = False
        if email:
            archive_doc = db.collection('deleted_accounts').document(email).get()
            if archive_doc.exists:
                email_blocked = archive_doc.to_dict().get('hasUsedTrial', False)

        # If either is blocked, mark user as having used trial
        if is_device_blocked or email_blocked:
            user_ref.update({
                "hasUsedTrial": True,
            })
            print(f"⚠️ User {authenticated_user_id} marked as trial-used (device_blocked={is_device_blocked}, email_blocked={email_blocked})")

        # Also register this device with the email for cross-reference
        db.collection('device_email_mappings').document(device_id).set({
            "deviceId": device_id,
            "emails": firestore.ArrayUnion([email]) if email else [],
            "lastSeenUid": authenticated_user_id,
            "lastSeenAt": firestore.SERVER_TIMESTAMP,
        }, merge=True)

        return {
            "success": True,
            "deviceId": device_id,
            "trialBlocked": is_device_blocked or email_blocked,
            "blockReason": "device" if is_device_blocked else ("email" if email_blocked else None)
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error registering device: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/admin/cleanup-auth")
async def cleanup_auth(request: Request):
    """
    TEMPORARY: Delete all Firebase Auth users for testing.
    Requires admin secret key.
    """
    body = await request.json()
    admin_key = body.get('adminKey')

    # Simple protection - require a key
    expected_key = os.environ.get("ADMIN_CLEANUP_KEY", "dreamai-cleanup-2024")
    if admin_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid admin key")

    try:
        deleted_users = []

        # List and delete all users
        page = firebase_auth.list_users()
        while page:
            for user in page.users:
                try:
                    firebase_auth.delete_user(user.uid)
                    deleted_users.append({
                        "uid": user.uid,
                        "email": user.email,
                    })
                    print(f"🗑️ Deleted user: {user.email or user.uid}")
                except Exception as e:
                    print(f"❌ Error deleting {user.uid}: {e}")

            page = page.get_next_page()

        print(f"✅ Deleted {len(deleted_users)} users from Firebase Auth")

        return {
            "success": True,
            "deleted_count": len(deleted_users),
            "deleted_users": deleted_users
        }

    except Exception as e:
        print(f"❌ Error in cleanup: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Optionally run with `python server.py` for local dev
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
