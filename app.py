import base64
import os
import mimetypes
import uuid
from typing import Optional  # Keep Optional for type hinting
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
from google import genai
from google.genai import types
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import shutil

load_dotenv()  # This loads the variables from .env

app = FastAPI(title="Gemini Image Generation API")

# Then add this after creating your FastAPI app instance
app.add_middleware(
    CORSMiddleware, # Your React app's URL
    allow_origins=["https://dreamai-checkpoint.netlify.app", "http://localhost:8081","http://localhost:3000", "https://vision-ai-tester.netlify.app" ],  # Your React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory to store uploaded and generated files
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check if API key is set
if not os.environ.get("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY environment variable must be set")

def save_file(upload_file: UploadFile) -> str:
    """Save an uploaded file to disk and return the path"""
    file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{upload_file.filename}")
    with open(file_path, "wb") as f:
        f.write(upload_file.file.read())
    return file_path

@app.post("/generate/")
async def generate_image(
    image1: UploadFile = File(...),
    image2: Optional[UploadFile] = File(None),  # Change: Make image2 optional
    prompt: str = Form(...),
    temperature: float = Form(1.0),
    top_p: float = Form(0.95),
    top_k: int = Form(40),
):
    """
    Generate an image using Gemini API based on one or two input images and a text prompt.

    - **image1**: First input image (required)
    - **image2**: Second input image (optional) # Change: Updated docstring
    - **prompt**: Text description of what to generate
    - **temperature**: Controls randomness (0.0-1.0)
    - **top_p**: Nucleus sampling parameter (0.0-1.0)
    - **top_k**: Top-k sampling parameter
    """
    # Change: Initialize paths and file objects to None
    image1_path = None
    image2_path = None
    file1 = None
    file2 = None

    try:
        # Initialize Gemini client
        client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
        )
        [os.remove(os.path.join(OUTPUT_DIR, f)) if os.path.isfile(os.path.join(OUTPUT_DIR, f)) else shutil.rmtree(os.path.join(OUTPUT_DIR, f)) for f in os.listdir(OUTPUT_DIR)]

        # Save and upload image1 (always required)
        image1_path = save_file(image1)
        file1 = client.files.upload(file=image1_path)

        # Change: Save and upload image2 only if provided
        if image2:
            image2_path = save_file(image2)
            file2 = client.files.upload(file=image2_path)

        # Set up the model request
        model = "gemini-2.0-flash-exp-image-generation"

        # Change: Dynamically build parts list
        parts = [
            types.Part.from_uri(
                file_uri=file1.uri,
                mime_type=file1.mime_type,
            ),
        ]
        if file2: # Add image2 part if it exists
            parts.append(
                types.Part.from_uri(
                    file_uri=file2.uri,
                    mime_type=file2.mime_type,
                )
            )
        parts.append(types.Part.from_text(text=prompt)) # Add text prompt last

        contents = [types.Content(role="user", parts=parts)]

        generate_content_config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=8192,
            response_modalities=["image", "text"],
        )

        # Generate content
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )

        # Process the response
        if not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
            raise HTTPException(status_code=500, detail="No content generated by the model")

        # Handle image in response
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                # Save the generated image
                output_filename = f"generated_{uuid.uuid4()}"
                file_extension = mimetypes.guess_extension(part.inline_data.mime_type) or ".png"
                output_file_path = os.path.join(OUTPUT_DIR, f"{output_filename}{file_extension}")

                with open(output_file_path, "wb") as f:
                    f.write(part.inline_data.data)

                # Change: Clean up temporary files conditionally
                if image1_path and os.path.exists(image1_path):
                    os.remove(image1_path)
                if image2_path and os.path.exists(image2_path):
                    os.remove(image2_path)

                return FileResponse(
                    path=output_file_path,
                    media_type=part.inline_data.mime_type,
                    filename=f"generated_image{file_extension}"
                )

        # Handle text-only response
        text_response = response.candidates[0].content.parts[0].text if hasattr(response.candidates[0].content.parts[0], 'text') else "No text response"

        # Change: Clean up temporary files conditionally
        if image1_path and os.path.exists(image1_path):
            os.remove(image1_path)
        if image2_path and os.path.exists(image2_path):
            os.remove(image2_path)

        return {"message": "Generation completed", "text": text_response}

    except Exception as e:
        # Change: Clean up any temporary files if they exist, conditionally
        if image1_path and os.path.exists(image1_path):
             os.remove(image1_path)
        if image2_path and os.path.exists(image2_path):
             os.remove(image2_path)

        # It's good practice to log the actual error for debugging
        print(f"Error during generation: {e}") # Optional: Log the error
        import traceback
        traceback.print_exc() # Optional: Print traceback for more details

        return JSONResponse(
            status_code=500,
            content={"error": f"An internal error occurred: {type(e).__name__}"}, # Avoid leaking too much detail
        )

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the API is running"""
    return {"status": "healthy"}

