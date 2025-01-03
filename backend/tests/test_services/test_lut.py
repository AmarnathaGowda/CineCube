import pytest
import numpy as np
from pathlib import Path
import json
import io
from PIL import Image
import cv2

from app.services.lut.generator import LUTGenerator
from app.services.lut.writer import CubeWriter
from app.core.errors import LUTGenerationError

pytestmark = pytest.mark.asyncio

class TestLUTGenerator:
    """Test cases for LUT Generator service."""

    async def test_generate_basic_lut(
        self,
        lut_generator: LUTGenerator,
        sample_lut_params: dict,
        tmp_path: Path
    ):
        """Test basic LUT generation."""
        output_path = tmp_path / "test.cube"
        cube_data = await lut_generator.generate(
            llm_params=sample_lut_params,
            output_path=output_path
        )
        
        # Verify output
        assert cube_data
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    async def test_generate_with_image_params(
        self,
        lut_generator: LUTGenerator,
        sample_lut_params: dict,
        sample_image_params: dict
    ):
        """Test LUT generation with image parameters."""
        result = await lut_generator.generate(
            llm_params=sample_lut_params,
            image_params=sample_image_params
        )
        
        assert result
        assert self._validate_cube_format(result)

    async def test_parameter_validation(
        self,
        lut_generator: LUTGenerator
    ):
        """Test parameter validation."""
        invalid_params = {
            "temperature": 150,  # Outside valid range
            "tint": -200,
            "saturation": 0,
            "contrast": 0
        }
        
        with pytest.raises(LUTGenerationError):
            await lut_generator.generate(llm_params=invalid_params)

    async def test_create_base_lut(self, lut_generator: LUTGenerator):
        """Test base LUT creation."""
        lut_data = lut_generator._create_base_lut()
        
        assert isinstance(lut_data, np.ndarray)
        assert lut_data.shape == (
            lut_generator.lut_size,
            lut_generator.lut_size,
            lut_generator.lut_size,
            3
        )

    @pytest.mark.parametrize("temp,tint", [
        (50, 0),    # Warm
        (-50, 0),   # Cool
        (0, 50),    # Magenta
        (0, -50),   # Green
    ])
    async def test_temperature_tint(
        self,
        lut_generator: LUTGenerator,
        temp: float,
        tint: float
    ):
        """Test temperature and tint adjustments."""
        params = {
            "temperature": temp,
            "tint": tint,
            "saturation": 0,
            "contrast": 0,
            "highlights": 0,
            "shadows": 0,
            "whites": 0,
            "blacks": 0
        }
        
        lut_data = await lut_generator._apply_temperature_tint(
            lut_generator._create_base_lut(),
            params
        )
        
        # Verify color shifts
        if temp > 0:
            assert np.mean(lut_data[..., 0]) > np.mean(lut_data[..., 2])  # More red
        elif temp < 0:
            assert np.mean(lut_data[..., 2]) > np.mean(lut_data[..., 0])  # More blue

    @pytest.mark.parametrize("contrast_value", [-50, 0, 50])
    async def test_contrast_adjustment(
        self,
        lut_generator: LUTGenerator,
        contrast_value: float
    ):
        """Test contrast adjustments."""
        params = {"contrast": contrast_value}
        base_lut = lut_generator._create_base_lut()
        
        adjusted_lut = await lut_generator._apply_contrast(base_lut, params)
        
        if contrast_value > 0:
            # Higher contrast should increase standard deviation
            assert np.std(adjusted_lut) > np.std(base_lut)
        elif contrast_value < 0:
            # Lower contrast should decrease standard deviation
            assert np.std(adjusted_lut) < np.std(base_lut)

    async def test_color_balance(
        self,
        lut_generator: LUTGenerator,
        sample_lut_params: dict
    ):
        """Test color balance adjustments."""
        # Add specific color balance settings
        params = sample_lut_params.copy()
        params["color_balance"] = {
            "shadows": {"red": 10, "green": 0, "blue": -10},
            "midtones": {"red": 0, "green": 0, "blue": 0},
            "highlights": {"red": -10, "green": 0, "blue": 10}
        }
        
        result = await lut_generator.generate(llm_params=params)
        assert result
        assert self._validate_cube_format(result)

    async def test_concurrent_generation(
        self,
        lut_generator: LUTGenerator,
        sample_lut_params: dict
    ):
        """Test concurrent LUT generation."""
        params_list = [
            {**sample_lut_params, "temperature": t}
            for t in [-50, 0, 50]
        ]
        
        results = await asyncio.gather(*[
            lut_generator.generate(llm_params=params)
            for params in params_list
        ])
        
        assert len(results) == len(params_list)
        assert all(self._validate_cube_format(r) for r in results)

    async def test_save_and_load(
        self,
        lut_generator: LUTGenerator,
        sample_lut_params: dict,
        tmp_path: Path
    ):
        """Test saving and loading LUTs."""
        # Generate and save LUT
        output_path = tmp_path / "test.cube"
        await lut_generator.generate(
            llm_params=sample_lut_params,
            output_path=output_path
        )
        
        # Verify file content
        assert output_path.exists()
        content = output_path.read_text()
        assert "LUT_3D_SIZE" in content
        assert self._validate_cube_format(content)

    async def test_memory_usage(
        self,
        lut_generator: LUTGenerator,
        sample_lut_params: dict
    ):
        """Test memory handling during generation."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Generate multiple LUTs
        for _ in range(5):
            await lut_generator.generate(llm_params=sample_lut_params)
        
        # Check memory hasn't grown significantly
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        assert memory_growth < 50 * 1024 * 1024  # 50MB limit

    async def test_create_test_lut(self, lut_generator: LUTGenerator):
        """Test creation of test LUT."""
        test_lut = lut_generator.create_test_lut()
        assert test_lut
        assert self._validate_cube_format(test_lut)

    def _validate_cube_format(self, cube_data: str) -> bool:
        """Helper to validate CUBE format."""
        lines = cube_data.strip().split("\n")
        
        # Check header
        if not any(line.startswith("LUT_3D_SIZE") for line in lines):
            return False
        
        # Check data format
        data_lines = [line for line in lines 
                     if not line.startswith("#") and line.strip()]
        
        for line in data_lines:
            if line.startswith(("TITLE", "LUT_3D_SIZE", "DOMAIN_MIN", "DOMAIN_MAX")):
                continue
            
            try:
                values = [float(v) for v in line.split()]
                if len(values) != 3 or any(v < 0 or v > 1 for v in values):
                    return False
            except ValueError:
                return False
        
        return True

# Test Fixtures
@pytest.fixture
def sample_image_params() -> dict:
    """Sample image analysis parameters."""
    return {
        "color_distribution": {
            "red_mean": 0.5,
            "green_mean": 0.5,
            "blue_mean": 0.5
        },
        "contrast": 1.0,
        "brightness": 0.5,
        "temperature": 0,
        "tint": 0
    }

@pytest.fixture
def create_test_lut():
    """Fixture to create test LUT data."""
    def _create_lut(size: int = 32) -> np.ndarray:
        return np.linspace(0, 1, size * size * size * 3).reshape(size, size, size, 3)
    return _create_lut