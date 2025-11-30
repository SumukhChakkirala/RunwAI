"""
Image preprocessing and postprocessing utilities for runway segmentation.
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import io
import base64
import traceback
from typing import Tuple, Optional


# ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Supported image formats
SUPPORTED_FORMATS = {'JPEG', 'JPG', 'PNG', 'BMP', 'GIF', 'TIFF', 'WEBP'}


class ImageProcessingError(Exception):
    """Custom exception for image processing errors with user-friendly messages."""
    def __init__(self, message: str, details: str = None):
        self.message = message
        self.details = details
        super().__init__(self.message)


def validate_image_bytes(image_bytes: bytes) -> None:
    """Validate that the image bytes are not empty and have reasonable size."""
    if image_bytes is None:
        raise ImageProcessingError(
            "No image data received",
            "The uploaded file appears to be empty or corrupted."
        )
    
    if len(image_bytes) == 0:
        raise ImageProcessingError(
            "Empty image file",
            "The uploaded file contains no data. Please try uploading again."
        )
    
    if len(image_bytes) < 100:
        raise ImageProcessingError(
            "File too small to be a valid image",
            f"The file is only {len(image_bytes)} bytes. Please upload a valid image file."
        )
    
    # Check for reasonable max size (100MB)
    max_size = 100 * 1024 * 1024
    if len(image_bytes) > max_size:
        raise ImageProcessingError(
            "Image file too large",
            f"The file is {len(image_bytes) / (1024*1024):.1f}MB. Maximum allowed size is 100MB."
        )


def preprocess_image(image_bytes: bytes) -> Tuple[torch.Tensor, Tuple[int, int], np.ndarray]:
    """
    Preprocess an image for model inference.
    
    Args:
        image_bytes: Raw image bytes from uploaded file
        
    Returns:
        Tuple of:
        - Preprocessed tensor ready for model input (1, 3, H, W)
        - Original image dimensions (height, width)
        - Original image as numpy array (for overlay generation)
        
    Raises:
        ImageProcessingError: If image cannot be processed
    """
    # Validate input
    validate_image_bytes(image_bytes)
    
    # Try to load with PIL first (better format support)
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise ImageProcessingError(
            "Failed to open image file",
            f"The file could not be opened as an image. Error: {str(e)}. "
            "Please ensure you're uploading a valid image file (JPEG, PNG, BMP, etc.)."
        )
    
    # Check image format
    img_format = pil_image.format
    if img_format and img_format.upper() not in SUPPORTED_FORMATS:
        raise ImageProcessingError(
            f"Unsupported image format: {img_format}",
            f"Please upload an image in one of these formats: {', '.join(SUPPORTED_FORMATS)}"
        )
    
    # Check image dimensions
    width, height = pil_image.size
    if width < 10 or height < 10:
        raise ImageProcessingError(
            "Image too small",
            f"Image dimensions ({width}x{height}) are too small. Minimum size is 10x10 pixels."
        )
    
    if width > 10000 or height > 10000:
        raise ImageProcessingError(
            "Image too large",
            f"Image dimensions ({width}x{height}) are too large. Maximum dimension is 10000 pixels. "
            "Please resize your image before uploading."
        )
    
    # Convert to RGB, handling various image modes
    try:
        if pil_image.mode == 'RGBA':
            # Handle transparency by compositing on white background
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
            background.paste(pil_image, mask=pil_image.split()[3])
            pil_image = background
        elif pil_image.mode == 'LA':
            # Grayscale with alpha
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
            rgba = pil_image.convert('RGBA')
            background.paste(rgba, mask=rgba.split()[3])
            pil_image = background
        elif pil_image.mode == 'P':
            # Palette mode (GIF, etc.)
            pil_image = pil_image.convert('RGB')
        elif pil_image.mode == 'L':
            # Grayscale
            pil_image = pil_image.convert('RGB')
        elif pil_image.mode == 'CMYK':
            pil_image = pil_image.convert('RGB')
        elif pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
    except Exception as e:
        raise ImageProcessingError(
            "Failed to convert image to RGB",
            f"Could not process image color mode '{pil_image.mode}'. Error: {str(e)}"
        )
    
    # Convert to numpy array
    try:
        img_rgb = np.array(pil_image)
    except Exception as e:
        raise ImageProcessingError(
            "Failed to convert image to array",
            f"Internal error converting image data. Error: {str(e)}"
        )
    
    # Validate numpy array
    if img_rgb is None or img_rgb.size == 0:
        raise ImageProcessingError(
            "Image conversion failed",
            "The image could not be converted to a processable format."
        )
    
    if len(img_rgb.shape) != 3:
        raise ImageProcessingError(
            "Invalid image dimensions",
            f"Expected a color image with 3 dimensions, got shape: {img_rgb.shape}"
        )
    
    if img_rgb.shape[2] != 3:
        raise ImageProcessingError(
            "Invalid color channels",
            f"Expected 3 color channels (RGB), got {img_rgb.shape[2]} channels."
        )
    
    original_shape = img_rgb.shape[:2]  # (height, width)
    original_img = img_rgb.copy()
    
    # Pad to multiple of 32 for encoder/decoder compatibility
    h, w = original_shape
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    
    try:
        if pad_h > 0 or pad_w > 0:
            img_rgb = cv2.copyMakeBorder(
                img_rgb, 0, pad_h, 0, pad_w,
                borderType=cv2.BORDER_REFLECT,
                value=(0, 0, 0)
            )
    except Exception as e:
        raise ImageProcessingError(
            "Failed to pad image",
            f"Error during image padding: {str(e)}"
        )
    
    # Apply transforms
    try:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        img_tensor = transform(img_rgb).unsqueeze(0)  # Add batch dimension
    except Exception as e:
        raise ImageProcessingError(
            "Failed to transform image",
            f"Error during image normalization: {str(e)}"
        )
    
    return img_tensor, original_shape, original_img


def run_inference(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    original_shape: Tuple[int, int],
    device: torch.device
) -> np.ndarray:
    """
    Run model inference on preprocessed image.
    
    Args:
        model: Loaded PyTorch model
        image_tensor: Preprocessed image tensor
        original_shape: Original (height, width) before padding
        device: torch.device for inference
        
    Returns:
        Binary segmentation mask (0 = background, 1 = runway)
        
    Raises:
        ImageProcessingError: If inference fails
    """
    if model is None:
        raise ImageProcessingError(
            "Model not available",
            "The segmentation model is not loaded. Please restart the server."
        )
    
    try:
        model.eval()
        
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            output = model(image_tensor)
            
            # Get prediction mask (argmax over class dimension)
            pred_mask = torch.argmax(output, dim=1).cpu().numpy()[0]
            
            # Crop back to original size (remove padding)
            orig_h, orig_w = original_shape
            pred_mask = pred_mask[:orig_h, :orig_w]
        
        return pred_mask.astype(np.uint8)
        
    except torch.cuda.OutOfMemoryError:
        raise ImageProcessingError(
            "GPU memory exceeded",
            "The image is too large to process with available GPU memory. "
            "Try uploading a smaller image or the server will use CPU (slower)."
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise ImageProcessingError(
                "Memory exceeded",
                "Not enough memory to process this image. Please try a smaller image."
            )
        raise ImageProcessingError(
            "Model inference failed",
            f"Error during segmentation: {str(e)}"
        )
    except Exception as e:
        raise ImageProcessingError(
            "Segmentation failed",
            f"Unexpected error during inference: {str(e)}"
        )


def create_overlay(
    original_img: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),  # Green
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create an overlay visualization of the segmentation mask on the original image.
    
    Args:
        original_img: Original RGB image
        mask: Binary segmentation mask
        color: RGB color for the runway overlay
        alpha: Transparency of the overlay (0-1)
        
    Returns:
        RGB image with segmentation overlay
        
    Raises:
        ImageProcessingError: If overlay creation fails
    """
    try:
        if original_img is None or mask is None:
            raise ImageProcessingError(
                "Missing data for overlay",
                "Original image or mask is not available."
            )
        
        overlay = original_img.copy()
        
        # Create colored mask
        colored_mask = np.zeros_like(original_img)
        colored_mask[mask == 1] = color
        
        # Blend overlay with original
        mask_area = mask == 1
        if np.any(mask_area):
            overlay[mask_area] = cv2.addWeighted(
                original_img[mask_area], 1 - alpha,
                colored_mask[mask_area], alpha,
                0
            )
        
        return overlay
        
    except ImageProcessingError:
        raise
    except Exception as e:
        raise ImageProcessingError(
            "Failed to create overlay",
            f"Error creating visualization: {str(e)}"
        )


def create_mask_image(mask: np.ndarray) -> np.ndarray:
    """
    Convert binary mask to a viewable grayscale image.
    
    Args:
        mask: Binary segmentation mask (0 or 1)
        
    Returns:
        Grayscale image (0 or 255)
        
    Raises:
        ImageProcessingError: If mask conversion fails
    """
    try:
        if mask is None:
            raise ImageProcessingError(
                "No mask data",
                "Segmentation mask is not available."
            )
        return (mask * 255).astype(np.uint8)
    except ImageProcessingError:
        raise
    except Exception as e:
        raise ImageProcessingError(
            "Failed to create mask image",
            f"Error: {str(e)}"
        )


def numpy_to_base64(img: np.ndarray, format: str = 'png') -> str:
    """
    Convert numpy array image to base64 encoded string.
    
    Args:
        img: RGB or grayscale numpy array
        format: Image format ('png' or 'jpeg')
        
    Returns:
        Base64 encoded image string
        
    Raises:
        ImageProcessingError: If encoding fails
    """
    try:
        if img is None:
            raise ImageProcessingError(
                "No image data to encode",
                "Image array is empty."
            )
        
        # Handle grayscale images
        if len(img.shape) == 2:
            img_to_encode = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            # Convert RGB to BGR for OpenCV encoding
            img_to_encode = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            raise ImageProcessingError(
                "Invalid image format for encoding",
                f"Unexpected image shape: {img.shape}"
            )
        
        # Encode to bytes
        if format.lower() == 'jpeg':
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            success, buffer = cv2.imencode('.jpg', img_to_encode, encode_param)
        else:
            success, buffer = cv2.imencode('.png', img_to_encode)
        
        if not success:
            raise ImageProcessingError(
                "Image encoding failed",
                f"Failed to encode image as {format.upper()}."
            )
        
        return base64.b64encode(buffer).decode('utf-8')
        
    except ImageProcessingError:
        raise
    except Exception as e:
        raise ImageProcessingError(
            "Failed to encode image",
            f"Error during base64 encoding: {str(e)}"
        )


def calculate_stats(mask: np.ndarray) -> dict:
    """
    Calculate statistics about the segmentation.
    
    Args:
        mask: Binary segmentation mask
        
    Returns:
        Dictionary with runway pixel count, total pixels, and percentage
        
    Raises:
        ImageProcessingError: If calculation fails
    """
    try:
        if mask is None:
            raise ImageProcessingError(
                "No mask for statistics",
                "Segmentation mask is not available."
            )
        
        total_pixels = int(mask.size)
        runway_pixels = int(np.sum(mask == 1))
        
        if total_pixels == 0:
            raise ImageProcessingError(
                "Invalid mask dimensions",
                "Mask has no pixels."
            )
        
        percentage = (runway_pixels / total_pixels) * 100
        
        return {
            "runway_pixels": runway_pixels,
            "total_pixels": total_pixels,
            "runway_percentage": round(percentage, 2)
        }
        
    except ImageProcessingError:
        raise
    except Exception as e:
        raise ImageProcessingError(
            "Failed to calculate statistics",
            f"Error: {str(e)}"
        )
