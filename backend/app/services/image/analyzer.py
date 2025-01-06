import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from dataclasses import dataclass
from scipy import stats
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.core.config import settings
from app.core.errors import FileProcessingError
from app.core.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ColorAnalysis:
    """Container for color analysis results."""
    dominant_colors: List[Tuple[int, int, int]]
    color_histogram: Dict[str, List[int]]
    average_brightness: float
    contrast_level: float
    saturation_level: float
    temperature: float
    tint: float

class ImageAnalyzer:
    """Service for analyzing images and extracting color characteristics."""

    def __init__(self):
        """Initialize the image analyzer service."""
        self.executor = ThreadPoolExecutor(max_workers=settings.MAX_WORKERS)
        logger.info("ImageAnalyzer service initialized")

    async def analyze_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Analyze image and extract color characteristics.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict containing analysis results
        """
        try:
            # Read image using OpenCV
            image = await self._read_image(image_path)
            
            # Run analyses in parallel
            analyses = await asyncio.gather(
                self._analyze_color_distribution(image),
                self._analyze_contrast(image),
                self._analyze_brightness(image),
                self._analyze_temperature_tint(image),
                self._extract_dominant_colors(image)
            )
            
            # Combine results
            result = {
                "color_distribution": analyses[0],
                "contrast": analyses[1],
                "brightness": analyses[2],
                "temperature_tint": analyses[3],
                "dominant_colors": analyses[4]
            }
            
            logger.info(f"Successfully analyzed image: {image_path}")
            return result

        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            raise FileProcessingError(f"Failed to analyze image: {str(e)}")

    async def _read_image(self, image_path: Path) -> np.ndarray:
        """Read image file asynchronously."""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                self.executor,
                cv2.imread,
                str(image_path)
            )
        except Exception as e:
            raise FileProcessingError(f"Failed to read image: {str(e)}")

    async def _analyze_color_distribution(self, image: np.ndarray) -> Dict:
        """Analyze color distribution in the image."""
        async def calculate_histogram(channel: np.ndarray) -> List[int]:
            hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
            return hist.flatten().tolist()

        # Split channels and calculate histograms
        b, g, r = cv2.split(image)
        histograms = await asyncio.gather(
            calculate_histogram(b),
            calculate_histogram(g),
            calculate_histogram(r)
        )

        return {
            "red_histogram": histograms[2],
            "green_histogram": histograms[1],
            "blue_histogram": histograms[0]
        }

    async def _analyze_contrast(self, image: np.ndarray) -> Dict:
        """Analyze image contrast levels."""
        try:
            # Convert to grayscale for contrast analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate standard deviation as measure of contrast
            contrast = np.std(gray)
            
            # Calculate local contrast using Sobel operator
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            local_contrast = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))
            
            return {
                "global_contrast": float(contrast),
                "local_contrast": float(local_contrast),
                "contrast_range": float(np.max(gray) - np.min(gray))
            }

        except Exception as e:
            logger.error(f"Error analyzing contrast: {str(e)}")
            raise

    async def _analyze_brightness(self, image: np.ndarray) -> Dict:
        """Analyze image brightness levels."""
        try:
            # Convert to HSV for brightness analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            brightness = hsv[..., 2]
            
            return {
                "average_brightness": float(np.mean(brightness)),
                "brightness_std": float(np.std(brightness)),
                "brightness_histogram": cv2.calcHist([brightness], [0], None, [256], [0, 256]).flatten().tolist()
            }

        except Exception as e:
            logger.error(f"Error analyzing brightness: {str(e)}")
            raise

    async def _analyze_temperature_tint(self, image: np.ndarray) -> Dict:
        """Analyze color temperature and tint."""
        try:
            # Split channels
            b, g, r = cv2.split(image)
            
            # Calculate average values
            avg_r = np.mean(r)
            avg_g = np.mean(g)
            avg_b = np.mean(b)
            
            # Calculate temperature (red vs blue ratio)
            temperature = (avg_r - avg_b) / ((avg_r + avg_b) / 2) * 100
            
            # Calculate tint (green vs magenta ratio)
            tint = (avg_g - ((avg_r + avg_b) / 2)) / (avg_g + 1e-6) * 100
            
            return {
                "temperature": float(temperature),
                "tint": float(tint),
                "rgb_averages": {
                    "red": float(avg_r),
                    "green": float(avg_g),
                    "blue": float(avg_b)
                }
            }

        except Exception as e:
            logger.error(f"Error analyzing temperature/tint: {str(e)}")
            raise

    async def _extract_dominant_colors(self, image: np.ndarray, n_colors: int = 5) -> List:
        """Extract dominant colors using K-means clustering."""
        try:
            # Reshape image for K-means
            pixels = image.reshape(-1, 3)
            
            # Convert to float32 for K-means
            pixels = np.float32(pixels)
            
            # Define criteria and run K-means
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Convert centers to integers
            centers = np.uint8(centers)
            
            # Calculate color percentages
            unique_labels, counts = np.unique(labels, return_counts=True)
            percentages = counts / len(labels) * 100
            
            # Sort colors by occurrence
            sorted_indices = np.argsort(percentages)[::-1]
            
            return [
                {
                    "color": centers[idx].tolist(),
                    "percentage": float(percentages[idx])
                }
                for idx in sorted_indices
            ]

        except Exception as e:
            logger.error(f"Error extracting dominant colors: {str(e)}")
            raise

    async def analyze_image_style(self, image_path: Path) -> Dict:
        """Perform advanced style analysis on the image."""
        try:
            image = await self._read_image(image_path)
            
            # Basic analysis
            basic_analysis = await self.analyze_image(image_path)
            
            # Additional style metrics
            style_metrics = await asyncio.gather(
                self._analyze_composition(image),
                self._analyze_texture(image),
                self._analyze_color_harmony(image)
            )
            
            return {
                **basic_analysis,
                "composition": style_metrics[0],
                "texture": style_metrics[1],
                "color_harmony": style_metrics[2]
            }

        except Exception as e:
            logger.error(f"Error analyzing image style: {str(e)}")
            raise

    async def _analyze_composition(self, image: np.ndarray) -> Dict:
        """Analyze image composition characteristics."""
        try:
            # Calculate edge distribution
            edges = cv2.Canny(image, 100, 200)
            
            # Analyze rule of thirds
            height, width = image.shape[:2]
            third_h, third_w = height // 3, width // 3
            
            # Calculate interest points using Harris corner detector
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            harris = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
            
            return {
                "edge_density": float(np.mean(edges)),
                "thirds_strength": self._calculate_thirds_strength(harris, third_h, third_w),
                "symmetry_score": self._calculate_symmetry(gray)
            }
        except Exception as e:
            logger.error(f"Error analyzing composition: {str(e)}")
            raise

    def _calculate_thirds_strength(self, harris: np.ndarray, third_h: int, third_w: int) -> float:
        """Calculate strength of composition along rule of thirds lines."""
        try:
            thirds_mask = np.zeros_like(harris)
            
            # Create mask for thirds lines
            thirds_mask[third_h::third_h, :] = 1
            thirds_mask[:, third_w::third_w] = 1
            
            # Calculate interest along thirds lines
            thirds_strength = np.sum(harris * thirds_mask) / (np.sum(harris) + 1e-6)
            
            return float(thirds_strength)
        except Exception as e:
            logger.error(f"Error calculating thirds strength: {str(e)}")
            raise

    def _calculate_symmetry(self, gray: np.ndarray) -> float:
        """Calculate image symmetry score."""
        try:
            # Flip image horizontally
            flipped = cv2.flip(gray, 1)
            
            # Calculate difference between original and flipped
            diff = np.abs(gray - flipped)
            
            # Normalize symmetry score
            symmetry = 1 - np.mean(diff) / 255
            
            return float(symmetry)
        except Exception as e:
            logger.error(f"Error calculating symmetry: {str(e)}")
            raise

    def __del__(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=False)