import numpy as np
from typing import (
    Dict,        # For dictionaries
    List,        # For lists
    Tuple,       # For tuples
    Set,         # For sets
    Optional,    # For optional values
    Any,         # For any type
    Union,       # For union types
    Callable,    # For functions
    TypeVar,     # For generic types
    Generic,     # For generic classes
    Sequence,    # For sequence types
    Mapping,     # For mapping types
    Iterator,    # For iterators
    Iterable     # For iterables
)
from pathlib import Path
import asyncio
from datetime import datetime
import logging

from app.core.config import settings
from app.core.errors import LUTGenerationError
from app.core.logger import get_logger
from app.services.lut.writer import CubeWriter

import aiofiles
import json

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
        parameters: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate a LUT based on parameters.
        
        Args:
            parameters: LUT generation parameters
            output_path: Optional path to save the LUT file
            
        Returns:
            str: Generated LUT data in .cube format
        """
        try:
            # Create base LUT structure
            lut_data = self._create_base_lut()

            # Apply transformations
            lut_data = await self._apply_transformations(lut_data, parameters)

            # Convert to CUBE format
            cube_data = self._format_cube_data(lut_data)

            # Save to file if path provided
            if output_path:
                await self._save_cube_file(cube_data, output_path)
                logger.info(f"LUT saved to {output_path}")

            return cube_data

        except Exception as e:
            logger.error(f"LUT generation failed: {str(e)}")
            raise

    def _create_base_lut(self) -> np.ndarray:
        """Create base 3D LUT structure."""
        x = np.linspace(0, 1, self.lut_size)
        y = np.linspace(0, 1, self.lut_size)
        z = np.linspace(0, 1, self.lut_size)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        return np.stack([X, Y, Z], axis=-1)
    
    async def _apply_transformations(
        self,
        lut_data: np.ndarray,
        params: Dict[str, Any]
    ) -> np.ndarray:
        """Apply color transformations."""
        # Temperature
        if 'temperature' in params:
            temp = params['temperature'] / 100
            lut_data[..., 0] += temp * 0.2  # Red
            lut_data[..., 2] -= temp * 0.2  # Blue

        # Tint
        if 'tint' in params:
            tint = params['tint'] / 100
            lut_data[..., 1] += tint * 0.2  # Green

        # Contrast
        if 'contrast' in params:
            contrast = params['contrast'] / 100
            lut_data = 0.5 + (lut_data - 0.5) * (1 + contrast)

        # Saturation
        if 'saturation' in params:
            sat = params['saturation'] / 100
            luminance = np.mean(lut_data, axis=-1, keepdims=True)
            lut_data = luminance + (lut_data - luminance) * (1 + sat)

        # Apply color balance
        if 'color_balance' in params:
            self._apply_color_balance(lut_data, params['color_balance'])

        return np.clip(lut_data, 0, 1)
    
    def _apply_color_balance(
        self,
        lut_data: np.ndarray,
        color_balance: Dict[str, Dict[str, float]]
    ) -> None:
        """Apply color balance adjustments."""
        luminance = np.mean(lut_data, axis=-1, keepdims=True)
        
        # Shadows adjustment (dark areas)
        shadow_mask = 1 - luminance
        for i, color in enumerate(['red', 'green', 'blue']):
            if color in color_balance.get('shadows', {}):
                adj = color_balance['shadows'][color] / 100
                lut_data[..., i] += adj * shadow_mask[..., 0]
        
        # Highlights adjustment (bright areas)
        highlight_mask = luminance
        for i, color in enumerate(['red', 'green', 'blue']):
            if color in color_balance.get('highlights', {}):
                adj = color_balance['highlights'][color] / 100
                lut_data[..., i] += adj * highlight_mask[..., 0]
        
        # Midtones adjustment
        midtone_mask = 1 - (shadow_mask * shadow_mask + highlight_mask * highlight_mask)
        for i, color in enumerate(['red', 'green', 'blue']):
            if color in color_balance.get('midtones', {}):
                adj = color_balance['midtones'][color] / 100
                lut_data[..., i] += adj * midtone_mask[..., 0]

    def _format_cube_data(self, lut_data: np.ndarray) -> str:
        """Format LUT data as .cube file."""
        lines = []
        
        # Add header
        lines.append("# Generated by LUT Generator")
        lines.append(f"LUT_3D_SIZE {self.lut_size}")
        lines.append("")
        lines.append("DOMAIN_MIN 0.0 0.0 0.0")
        lines.append("DOMAIN_MAX 1.0 1.0 1.0")
        lines.append("")
        
        # Add LUT data
        for i in range(self.lut_size):
            for j in range(self.lut_size):
                for k in range(self.lut_size):
                    r, g, b = lut_data[i, j, k]
                    lines.append(f"{r:.6f} {g:.6f} {b:.6f}")
        
        return "\n".join(lines)
    
    async def _save_cube_file(self, cube_data: str, output_path: Path) -> None:
        """Save .cube file."""
        async with aiofiles.open(output_path, 'w') as f:
            await f.write(cube_data)

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