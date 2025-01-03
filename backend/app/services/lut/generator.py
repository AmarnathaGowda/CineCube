import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import asyncio
from datetime import datetime
import logging

from app.core.config import settings
from app.core.errors import LUTGenerationError
from app.core.logger import get_logger
from app.services.lut.writer import CubeWriter

logger = get_logger(__name__)

class LUTGenerator:
    """Service for generating color lookup tables (LUTs)."""

    def __init__(self, lut_size: int = 32):
        """
        Initialize LUT generator.
        
        Args:
            lut_size: Size of the 3D LUT cube (default: 32)
        """
        self.lut_size = lut_size
        self.writer = CubeWriter()
        logger.info(f"LUT Generator initialized with size {lut_size}")

    async def generate(
        self,
        llm_params: Dict,
        image_params: Optional[Dict] = None,
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate a LUT based on parameters from LLM and image analysis.
        
        Args:
            llm_params: Parameters from LLM processing
            image_params: Optional parameters from image analysis
            output_path: Optional path to save the LUT file
            
        Returns:
            str: Generated LUT data in .cube format
        """
        try:
            # Merge and validate parameters
            params = self._merge_parameters(llm_params, image_params)
            self._validate_parameters(params)

            # Create base LUT structure
            lut_data = self._create_base_lut()

            # Apply color transformations
            lut_data = await asyncio.gather(
                self._apply_temperature_tint(lut_data, params),
                self._apply_contrast(lut_data, params),
                self._apply_saturation(lut_data, params),
                self._apply_color_balance(lut_data, params)
            )

            # Combine transformations
            final_lut = self._combine_transformations(lut_data)

            # Generate .cube format
            cube_data = self.writer.generate_cube_format(
                final_lut,
                title=f"Generated LUT {datetime.now().isoformat()}"
            )

            # Save to file if path provided
            if output_path:
                await self.writer.save_cube_file(cube_data, output_path)

            return cube_data

        except Exception as e:
            logger.error(f"Error generating LUT: {str(e)}")
            raise LUTGenerationError(f"Failed to generate LUT: {str(e)}")

    def _merge_parameters(self, llm_params: Dict, image_params: Optional[Dict]) -> Dict:
        """Merge parameters from LLM and image analysis."""
        merged = llm_params.copy()

        if image_params:
            # Weight parameters based on confidence or priority
            for key in merged:
                if key in image_params:
                    merged[key] = (
                        merged[key] * 0.7 +  # LLM parameter weight
                        image_params[key] * 0.3  # Image analysis weight
                    )

        return merged

    def _validate_parameters(self, params: Dict) -> None:
        """Validate LUT generation parameters."""
        required_params = {
            "temperature": (-100, 100),
            "tint": (-100, 100),
            "saturation": (-100, 100),
            "contrast": (-100, 100),
            "highlights": (-100, 100),
            "shadows": (-100, 100),
            "whites": (-100, 100),
            "blacks": (-100, 100)
        }

        for param, (min_val, max_val) in required_params.items():
            if param not in params:
                raise LUTGenerationError(f"Missing required parameter: {param}")
            if not min_val <= params[param] <= max_val:
                raise LUTGenerationError(
                    f"Parameter {param} out of range [{min_val}, {max_val}]"
                )

    def _create_base_lut(self) -> np.ndarray:
        """Create base 3D LUT structure."""
        # Create coordinate grid
        x = np.linspace(0, 1, self.lut_size)
        y = np.linspace(0, 1, self.lut_size)
        z = np.linspace(0, 1, self.lut_size)
        
        # Create meshgrid
        X, Y, Z = np.meshgrid(x, y, z)
        
        # Stack to create LUT
        return np.stack([X, Y, Z], axis=-1)

    async def _apply_temperature_tint(
        self,
        lut: np.ndarray,
        params: Dict
    ) -> np.ndarray:
        """Apply temperature and tint adjustments."""
        temp = params["temperature"] / 100  # Normalize to [-1, 1]
        tint = params["tint"] / 100

        # Temperature adjustment (red-blue balance)
        lut[..., 0] += temp * 0.2  # Red channel
        lut[..., 2] -= temp * 0.2  # Blue channel

        # Tint adjustment (green-magenta balance)
        lut[..., 1] += tint * 0.2  # Green channel

        return np.clip(lut, 0, 1)

    async def _apply_contrast(
        self,
        lut: np.ndarray,
        params: Dict
    ) -> np.ndarray:
        """Apply contrast adjustment."""
        contrast = params["contrast"] / 100
        midpoint = 0.5

        # Apply contrast curve
        lut = midpoint + (lut - midpoint) * (1 + contrast)
        
        return np.clip(lut, 0, 1)

    async def _apply_saturation(
        self,
        lut: np.ndarray,
        params: Dict
    ) -> np.ndarray:
        """Apply saturation adjustment."""
        saturation = params["saturation"] / 100

        # Convert to HSV-like space
        rgb_mean = np.mean(lut, axis=-1, keepdims=True)
        lut = rgb_mean + (lut - rgb_mean) * (1 + saturation)

        return np.clip(lut, 0, 1)

    async def _apply_color_balance(
        self,
        lut: np.ndarray,
        params: Dict
    ) -> np.ndarray:
        """Apply color balance adjustments."""
        if "color_balance" in params:
            balance = params["color_balance"]
            
            # Apply to different luminance ranges
            for zone in ["shadows", "midtones", "highlights"]:
                if zone in balance:
                    lut = self._apply_zone_balance(lut, balance[zone], zone)

        return np.clip(lut, 0, 1)

    def _apply_zone_balance(
        self,
        lut: np.ndarray,
        balance: Dict[str, float],
        zone: str
    ) -> np.ndarray:
        """Apply color balance to specific luminance zone."""
        # Calculate luminance
        luminance = np.mean(lut, axis=-1, keepdims=True)
        
        # Define zone weights
        if zone == "shadows":
            weight = 1 - luminance
        elif zone == "highlights":
            weight = luminance
        else:  # midtones
            weight = 1 - 4 * (luminance - 0.5) ** 2

        # Apply color adjustments
        adjustments = np.array([
            balance.get("red", 0),
            balance.get("green", 0),
            balance.get("blue", 0)
        ]) / 100

        lut += weight * adjustments

        return lut

    def _combine_transformations(self, transformations: List[np.ndarray]) -> np.ndarray:
        """Combine multiple LUT transformations."""
        # Average all transformations
        return np.mean(transformations, axis=0)

    def create_test_lut(self) -> str:
        """Create a test LUT for validation."""
        try:
            # Create identity LUT
            lut = self._create_base_lut()
            
            # Generate .cube format
            return self.writer.generate_cube_format(lut, title="Test LUT")
        except Exception as e:
            logger.error(f"Error creating test LUT: {str(e)}")
            raise LUTGenerationError(f"Failed to create test LUT: {str(e)}")

    def __del__(self):
        """Cleanup resources."""
        logger.info("LUT Generator cleanup completed")