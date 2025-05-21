"""Utilities for image handling in OpenVLM."""

import io
import logging
import aiohttp
import asyncio
from typing import Optional, Union, Tuple
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

async def load_image_from_url(url: str, timeout: int = 10) -> Image.Image:
    """Asynchronously download and load an image from a URL.
    
    Args:
        url: The URL of the image to download
        timeout: Timeout in seconds for the HTTP request
        
    Returns:
        PIL Image object
        
    Raises:
        Exception: If the image cannot be downloaded or loaded
    """
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, timeout=timeout) as response:
                if response.status != 200:
                    raise ValueError(f"HTTP error {response.status} for URL: {url}")
                
                image_data = await response.read()
                return Image.open(io.BytesIO(image_data))
        except asyncio.TimeoutError:
            raise TimeoutError(f"Timeout while downloading image from {url}")
        except Exception as e:
            logger.exception(f"Error downloading image from {url}")
            raise ValueError(f"Failed to download image: {str(e)}")

def resize_and_pad_image(
    image: Union[Image.Image, np.ndarray, str, Path],
    target_size: Tuple[int, int] = (224, 224),
    pad_color: Tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """Resize and pad an image to a target size while preserving aspect ratio.
    
    Args:
        image: Input image (PIL Image, numpy array, or path to image file)
        target_size: Target size as (width, height)
        pad_color: Padding color as RGB tuple
        
    Returns:
        Resized and padded image as a PIL Image
    """
    # Convert input to PIL Image if needed
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Get original dimensions
    orig_width, orig_height = image.size
    target_width, target_height = target_size
    
    # Calculate scaling factor to preserve aspect ratio
    scale = min(target_width / orig_width, target_height / orig_height)
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)
    
    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Create a new blank image with the target size
    padded_image = Image.new("RGB", target_size, pad_color)
    
    # Paste the resized image onto the padded image, centered
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    padded_image.paste(resized_image, (paste_x, paste_y))
    
    return padded_image

def normalize_image(image: np.ndarray, mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                   std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> np.ndarray:
    """Normalize an image with mean and standard deviation.
    
    Args:
        image: Input image as numpy array (H, W, C) with values in [0, 255]
        mean: Mean values for each channel
        std: Standard deviation values for each channel
        
    Returns:
        Normalized image as numpy array with values in float32
    """
    # Convert to float32 and scale to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Normalize with mean and std for each channel
    image[..., 0] = (image[..., 0] - mean[0]) / std[0]
    image[..., 1] = (image[..., 1] - mean[1]) / std[1]
    image[..., 2] = (image[..., 2] - mean[2]) / std[2]
    
    return image

def crop_diagram_region(
    image: Union[Image.Image, np.ndarray],
    threshold: float = 0.1,
    padding: int = 10
) -> Image.Image:
    """Automatically crop to the diagram region by removing whitespace/background.
    
    Args:
        image: Input image (PIL Image or numpy array)
        threshold: Threshold for determining background vs. content
        padding: Extra padding to add around the cropped region
        
    Returns:
        Cropped image as PIL Image
    """
    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        # Convert to RGB if not already
        if image.mode != "RGB":
            image = image.convert("RGB")
        img_array = np.array(image)
    else:
        img_array = image
    
    # Convert to grayscale
    if img_array.ndim == 3:
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array
    
    # Invert if background is dark
    if np.mean(gray) < 128:
        gray = 255 - gray
    
    # Find non-background pixels
    mask = gray < (255 * (1 - threshold))
    
    # Find the bounding box of non-background pixels
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        # No content found, return the original image
        if isinstance(image, Image.Image):
            return image
        else:
            return Image.fromarray(img_array)
    
    # Get the bounding box
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    # Add padding
    height, width = gray.shape[:2]
    y_min = max(0, y_min - padding)
    y_max = min(height, y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(width, x_max + padding)
    
    # Crop the image
    if isinstance(image, Image.Image):
        return image.crop((x_min, y_min, x_max, y_max))
    else:
        return Image.fromarray(img_array[y_min:y_max, x_min:x_max]) 