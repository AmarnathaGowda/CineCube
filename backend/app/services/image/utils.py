import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
from pathlib import Path
import logging
from PIL import Image
import io

from app.core.logger import get_logger
from app.core.errors import FileProcessingError

logger = get_logger(__name__)

def load_image(file_path: Union[str, Path, bytes], max_size: int = 1920) -> np.ndarray:
    """
    Load and preprocess image for analysis.
    
    Args:
        file_path: Path to image file or bytes data
        max_size: Maximum dimension size
        
    Returns:
        numpy.ndarray: Processed image
    """
    try:
        if isinstance(file_path, (str, Path)):
            image = cv2.imread(str(file_path))
        else:
            nparr = np.frombuffer(file_path, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
        if image is None:
            raise FileProcessingError("Failed to load image")
            
        # Resize if needed
        image = resize_image(image, max_size)
        
        return image
    except Exception as e:
        logger.error(f"Error loading image: {str(e)}")
        raise FileProcessingError(f"Failed to load image: {str(e)}")

def resize_image(image: np.ndarray, max_size: int = 1920) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image
        max_size: Maximum dimension size
        
    Returns:
        numpy.ndarray: Resized image
    """
    height, width = image.shape[:2]
    
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return image

def convert_color_space(image: np.ndarray, target_space: str) -> np.ndarray:
    """
    Convert image between color spaces.
    
    Args:
        image: Input image
        target_space: Target color space (BGR, RGB, HSV, LAB)
        
    Returns:
        numpy.ndarray: Converted image
    """
    color_spaces = {
        'BGR': lambda x: x,
        'RGB': lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB),
        'HSV': lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2HSV),
        'LAB': lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2LAB)
    }
    
    if target_space.upper() not in color_spaces:
        raise ValueError(f"Unsupported color space: {target_space}")
        
    return color_spaces[target_space.upper()](image)

def calculate_image_stats(image: np.ndarray) -> Dict[str, float]:
    """
    Calculate basic image statistics.
    
    Args:
        image: Input image
        
    Returns:
        dict: Image statistics
    """
    try:
        # Convert to different color spaces for analysis
        hsv = convert_color_space(image, 'HSV')
        lab = convert_color_space(image, 'LAB')
        
        # Calculate statistics
        stats = {
            'brightness_mean': float(np.mean(hsv[..., 2])),
            'brightness_std': float(np.std(hsv[..., 2])),
            'saturation_mean': float(np.mean(hsv[..., 1])),
            'saturation_std': float(np.std(hsv[..., 1])),
            'lightness_mean': float(np.mean(lab[..., 0])),
            'lightness_std': float(np.std(lab[..., 0])),
            'contrast': calculate_contrast(image)
        }
        
        return stats
    except Exception as e:
        logger.error(f"Error calculating image stats: {str(e)}")
        raise

def calculate_contrast(image: np.ndarray) -> float:
    """
    Calculate RMS contrast of the image.
    
    Args:
        image: Input image
        
    Returns:
        float: RMS contrast value
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray.astype(float)) / np.mean(gray.astype(float)))

def create_color_histogram(image: np.ndarray, bins: int = 256) -> Dict[str, List[int]]:
    """
    Create color histograms for each channel.
    
    Args:
        image: Input image
        bins: Number of histogram bins
        
    Returns:
        dict: Histogram data for each channel
    """
    histograms = {}
    for i, channel in enumerate(['blue', 'green', 'red']):
        hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
        histograms[channel] = hist.flatten().tolist()
    return histograms

def extract_dominant_colors(image: np.ndarray, n_colors: int = 5) -> List[Dict[str, Union[List[int], float]]]:
    """
    Extract dominant colors using K-means clustering.
    
    Args:
        image: Input image
        n_colors: Number of colors to extract
        
    Returns:
        list: Dominant colors with percentages
    """
    # Reshape image for clustering
    pixels = image.reshape(-1, 3)
    pixels = np.float32(pixels)
    
    # Perform K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Calculate color percentages
    _, counts = np.unique(labels, return_counts=True)
    percentages = counts / len(labels) * 100
    
    # Sort colors by percentage
    colors = []
    for center, percentage in sorted(zip(centers, percentages), key=lambda x: x[1], reverse=True):
        colors.append({
            'color': center.astype(int).tolist(),
            'percentage': float(percentage)
        })
    
    return colors

def apply_color_mask(image: np.ndarray, 
                    lower_bound: Tuple[int, int, int], 
                    upper_bound: Tuple[int, int, int]) -> np.ndarray:
    """
    Apply color mask to image.
    
    Args:
        image: Input image
        lower_bound: Lower HSV bounds
        upper_bound: Upper HSV bounds
        
    Returns:
        numpy.ndarray: Masked image
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_bound), np.array(upper_bound))
    return cv2.bitwise_and(image, image, mask=mask)

def calculate_image_quality(image: np.ndarray) -> Dict[str, float]:
    """
    Calculate various image quality metrics.
    
    Args:
        image: Input image
        
    Returns:
        dict: Quality metrics
    """
    # Convert to grayscale for some calculations
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate sharpness using Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = np.var(laplacian)
    
    # Calculate noise level
    noise = estimate_noise(gray)
    
    # Calculate dynamic range
    dynamic_range = np.max(gray) - np.min(gray)
    
    return {
        'sharpness': float(sharpness),
        'noise_level': float(noise),
        'dynamic_range': float(dynamic_range)
    }

def estimate_noise(image: np.ndarray) -> float:
    """
    Estimate image noise level.
    
    Args:
        image: Grayscale image
        
    Returns:
        float: Estimated noise level
    """
    H, W = image.shape
    M = [[1, -2, 1],
         [-2, 4, -2],
         [1, -2, 1]]
    sigma = np.sum(np.sum(np.absolute(cv2.filter2D(image, -1, np.array(M)))))
    sigma = sigma * sigma / (float(H) * float(W))
    return sigma

def create_preview(image: np.ndarray, max_size: int = 800) -> bytes:
    """
    Create preview image in bytes format.
    
    Args:
        image: Input image
        max_size: Maximum dimension size
        
    Returns:
        bytes: Preview image data
    """
    # Resize image for preview
    preview = resize_image(image, max_size)
    
    # Convert to RGB for PIL
    preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(preview_rgb)
    
    # Save to bytes
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    return img_byte_arr

def validate_image(image: np.ndarray) -> bool:
    """
    Validate image data.
    
    Args:
        image: Input image
        
    Returns:
        bool: Whether image is valid
    """
    if image is None or image.size == 0:
        return False
        
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        return False
        
    if len(image.shape) != 3 or image.shape[2] != 3:
        return False
        
    return True