"""
FastAPI backend for RunwAI - Runway Segmentation API

Endpoints:
- POST /api/segment: Upload image, return segmentation mask + overlay
- GET /api/health: Health check, verify model is loaded
"""

import os
import sys
import traceback
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from model import ResNetUNet, load_model
from utils import (
    preprocess_image,
    run_inference,
    create_overlay,
    create_mask_image,
    numpy_to_base64,
    calculate_stats,
    ImageProcessingError
)


# Global model and device
model = None
device = None
model_load_error = None

# Model configuration - hosted on Hugging Face
MODEL_URL = "https://huggingface.co/lazypanther/Runway-Segmentation/resolve/main/runway_0.9243.pth"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "runway_0.9243.pth")


def download_model():
    """Download model from Hugging Face if not present locally."""
    if os.path.exists(MODEL_PATH):
        print(f"Model already exists at: {MODEL_PATH}")
        return True
    
    print(f"Downloading model from: {MODEL_URL}")
    try:
        # Create a custom opener with proper headers
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'RunwAI/1.0')]
        urllib.request.install_opener(opener)
        
        # Download with progress
        def show_progress(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, block_num * block_size * 100 / total_size)
                print(f"\rDownloading: {percent:.1f}%", end="", flush=True)
        
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=show_progress)
        print("\nModel downloaded successfully!")
        return True
    except Exception as e:
        print(f"\nFailed to download model: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model, device, model_load_error
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Download model if needed
    if not download_model():
        model_load_error = f"Failed to download model from: {MODEL_URL}"
        print(f"ERROR: {model_load_error}")
    elif not os.path.exists(MODEL_PATH):
        model_load_error = f"Model file not found at: {MODEL_PATH}"
        print(f"ERROR: {model_load_error}")
    else:
        # Load model
        try:
            print(f"Loading model from: {MODEL_PATH}")
            model = load_model(MODEL_PATH, device)
            print("Model loaded successfully!")
        except Exception as e:
            model_load_error = f"Failed to load model: {str(e)}"
            print(f"ERROR: {model_load_error}")
            traceback.print_exc()
    
    yield
    
    # Cleanup
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="RunwAI - Runway Segmentation API",
    description="A high-precision semantic segmentation model for runway detection in aerial imagery",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve React static files in production
DIST_PATH = Path(__file__).parent.parent / "dist"
PUBLIC_PATH = Path(__file__).parent.parent / "public"

# Mount static assets
if DIST_PATH.exists():
    assets_path = DIST_PATH / "assets"
    if assets_path.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_path)), name="assets")

# Serve sample images from public folder
if PUBLIC_PATH.exists():
    for sample_file in PUBLIC_PATH.glob("sample*.png"):
        sample_name = sample_file.name
        sample_path = sample_file
        
        @app.get(f"/{sample_name}")
        async def serve_sample(filepath: Path = sample_path):
            if filepath.exists():
                return FileResponse(filepath)
            return {"error": "Sample image not found"}


@app.get("/api/health")
async def health_check():
    """
    Health check endpoint.
    Returns model status and device information.
    """
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_error": model_load_error,
        "device": str(device) if device else None
    }


@app.post("/api/segment")
async def segment_runway(file: UploadFile = File(...)):
    """
    Segment runway from uploaded aerial/satellite image.
    
    Args:
        file: Uploaded image file (JPEG, PNG, etc.)
        
    Returns:
        JSON response with:
        - success: Boolean indicating success
        - overlay: Base64 encoded image with runway overlay (green)
        - mask: Base64 encoded binary mask image
        - stats: Segmentation statistics (runway pixels, percentage)
    """
    global model, device, model_load_error
    
    # Check if model is loaded
    if model is None:
        error_detail = {
            "error": "Model not available",
            "message": model_load_error or "The segmentation model failed to load.",
            "suggestion": "Please check that the model file exists and restart the server."
        }
        return JSONResponse(status_code=503, content=error_detail)
    
    # Validate file
    if file is None:
        return JSONResponse(status_code=400, content={
            "error": "No file provided",
            "message": "Please upload an image file.",
            "suggestion": "Use the file upload button or drag and drop an image."
        })
    
    # Validate file type
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={
            "error": "Invalid file type",
            "message": f"Received file type: {content_type or 'unknown'}",
            "suggestion": "Please upload an image file (JPEG, PNG, BMP, GIF, WEBP, or TIFF)."
        })
    
    # Validate filename
    filename = file.filename or "unknown"
    
    try:
        # Read file contents
        contents = await file.read()
        
        if len(contents) == 0:
            return JSONResponse(status_code=400, content={
                "error": "Empty file",
                "message": f"The file '{filename}' appears to be empty.",
                "suggestion": "Please try uploading the image again."
            })
        
        # Preprocess image
        try:
            image_tensor, original_shape, original_img = preprocess_image(contents)
        except ImageProcessingError as e:
            return JSONResponse(status_code=400, content={
                "error": e.message,
                "message": e.details or e.message,
                "suggestion": "Please try a different image file."
            })
        
        # Run inference
        try:
            mask = run_inference(model, image_tensor, original_shape, device)
        except ImageProcessingError as e:
            return JSONResponse(status_code=500, content={
                "error": e.message,
                "message": e.details or e.message,
                "suggestion": "Try uploading a smaller image or try again later."
            })
        
        # Create visualizations
        try:
            overlay = create_overlay(original_img, mask, color=(0, 255, 0), alpha=0.5)
            mask_img = create_mask_image(mask)
        except ImageProcessingError as e:
            return JSONResponse(status_code=500, content={
                "error": e.message,
                "message": e.details or e.message,
                "suggestion": "Error creating visualizations. Please try again."
            })
        
        # Convert to base64
        try:
            overlay_b64 = numpy_to_base64(overlay, format='png')
            mask_b64 = numpy_to_base64(mask_img, format='png')
            original_b64 = numpy_to_base64(original_img, format='png')
        except ImageProcessingError as e:
            return JSONResponse(status_code=500, content={
                "error": e.message,
                "message": e.details or e.message,
                "suggestion": "Error encoding results. Please try again."
            })
        
        # Calculate statistics
        try:
            stats = calculate_stats(mask)
        except ImageProcessingError as e:
            return JSONResponse(status_code=500, content={
                "error": e.message,
                "message": e.details or e.message,
                "suggestion": "Error calculating statistics. Please try again."
            })
        
        return JSONResponse(content={
            "success": True,
            "original": f"data:image/png;base64,{original_b64}",
            "overlay": f"data:image/png;base64,{overlay_b64}",
            "mask": f"data:image/png;base64,{mask_b64}",
            "stats": stats,
            "image_size": {
                "width": int(original_shape[1]),
                "height": int(original_shape[0])
            }
        })
        
    except ImageProcessingError as e:
        print(f"Image processing error: {e.message} - {e.details}")
        return JSONResponse(status_code=400, content={
            "error": e.message,
            "message": e.details or e.message,
            "suggestion": "Please try a different image or check the file format."
        })
    except Exception as e:
        # Log the full error for debugging
        error_msg = str(e)
        print(f"Unexpected error processing '{filename}': {error_msg}")
        traceback.print_exc()
        
        return JSONResponse(status_code=500, content={
            "error": "Processing failed",
            "message": f"An unexpected error occurred while processing your image: {error_msg}",
            "suggestion": "Please try a different image or contact support if the problem persists."
        })


@app.get("/")
async def root():
    """Serve React app or API info."""
    index_path = DIST_PATH / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    
    # Fallback to API info if React build not found
    return {
        "name": "RunwAI - Runway Segmentation API",
        "version": "1.0.0",
        "status": "ready" if model is not None else "model not loaded",
        "endpoints": {
            "POST /api/segment": "Upload image for runway segmentation",
            "GET /api/health": "Health check"
        }
    }


@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    """Serve React SPA for all other routes."""
    # Check if it's a sample image
    if full_path.startswith("sample") and full_path.endswith(".png"):
        sample_path = PUBLIC_PATH / full_path
        if sample_path.exists():
            return FileResponse(sample_path)
    
    # Check if file exists in dist
    file_path = DIST_PATH / full_path
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)
    
    # Fallback to index.html for React routing
    index_path = DIST_PATH / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    
    return {"error": "Not found", "path": full_path}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
