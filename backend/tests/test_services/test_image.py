import pytest
import numpy as np
import cv2
from PIL import Image
import io
from pathlib import Path
from typing import Dict, Tuple
import asyncio

from app.services.image.analyzer import ImageAnalyzer
from app.core.errors import FileProcessingError

pytestmark = pytest.mark.asyncio

class TestImageAnalyzer:
    """Test cases for Image Analyzer service."""

    async def test_analyze_basic_image(
        self,
        image_analyzer: ImageAnalyzer,
        tmp_path: Path,
        sample_image: bytes
    ):
        """Test basic image analysis."""
        # Save sample image
        image_path = tmp_path / "test_image.jpg"
        with open(image_path, "wb") as f:
            f.write(sample_image)

        # Analyze image
        result = await image_analyzer.analyze_image(image_path)

        # Verify result structure
        assert "color_distribution" in result
        assert "contrast" in result
        assert "brightness" in result
        assert "temperature_tint" in result
        assert "dominant_colors" in result

    async def test_analyze_color_distribution(
        self,
        image_analyzer: ImageAnalyzer,
        create_test_image: callable
    ):
        """Test color distribution analysis."""
        # Create test image with known colors
        image_path = create_test_image(color=(255, 0, 0))  # Pure red
        
        result = await image_analyzer.analyze_image(image_path)
        
        # Check color distribution
        hist = result["color_distribution"]
        assert "red_histogram" in hist
        assert "green_histogram" in hist
        assert "blue_histogram" in hist
        
        # Verify red channel dominance
        red_values = sum(hist["red_histogram"])
        blue_values = sum(hist["blue_histogram"])
        green_values = sum(hist["green_histogram"])
        
        assert red_values > blue_values
        assert red_values > green_values

    async def test_analyze_contrast(
        self,
        image_analyzer: ImageAnalyzer,
        create_test_image: callable
    ):
        """Test contrast analysis."""
        # Create high contrast test image
        image_path = create_test_image(
            colors=[(0, 0, 0), (255, 255, 255)]
        )
        
        result = await image_analyzer.analyze_image(image_path)
        
        contrast = result["contrast"]
        assert contrast["global_contrast"] > 0.5  # High contrast
        assert "local_contrast" in contrast

    async def test_analyze_brightness(
        self,
        image_analyzer: ImageAnalyzer,
        create_test_image: callable
    ):
        """Test brightness analysis."""
        # Create bright test image
        image_path = create_test_image(color=(200, 200, 200))
        
        result = await image_analyzer.analyze_image(image_path)
        
        brightness = result["brightness"]
        assert brightness["average_brightness"] > 128  # Higher than middle gray
        assert "brightness_histogram" in brightness

    async def test_analyze_temperature_tint(
        self,
        image_analyzer: ImageAnalyzer,
        create_test_image: callable
    ):
        """Test temperature and tint analysis."""
        # Create warm-tinted test image
        image_path = create_test_image(color=(255, 200, 150))
        
        result = await image_analyzer.analyze_image(image_path)
        
        temp_tint = result["temperature_tint"]
        assert temp_tint["temperature"] > 0  # Warm temperature
        assert "tint" in temp_tint
        assert "rgb_averages" in temp_tint

    @pytest.mark.parametrize("invalid_path", [
        "nonexistent.jpg",
        "invalid/path/image.jpg",
        "",
    ])
    async def test_invalid_image_path(
        self,
        image_analyzer: ImageAnalyzer,
        invalid_path: str
    ):
        """Test handling of invalid image paths."""
        with pytest.raises(FileProcessingError):
            await image_analyzer.analyze_image(Path(invalid_path))

    async def test_concurrent_analysis(
        self,
        image_analyzer: ImageAnalyzer,
        tmp_path: Path,
        create_test_image: callable
    ):
        """Test concurrent image analysis."""
        # Create multiple test images
        image_paths = []
        for i in range(3):
            path = create_test_image(
                color=(50 * i, 100, 150),
                filename=f"test_{i}.jpg"
            )
            image_paths.append(path)
        
        # Analyze concurrently
        results = await asyncio.gather(*[
            image_analyzer.analyze_image(path)
            for path in image_paths
        ])
        
        assert len(results) == len(image_paths)
        assert all(isinstance(r, dict) for r in results)

    async def test_analyze_image_style(
        self,
        image_analyzer: ImageAnalyzer,
        create_test_image: callable
    ):
        """Test advanced style analysis."""
        image_path = create_test_image(
            colors=[(200, 150, 100), (50, 75, 100)]
        )
        
        result = await image_analyzer.analyze_image_style(image_path)
        
        assert "composition" in result
        assert "texture" in result
        assert "color_harmony" in result

    @pytest.mark.parametrize("image_format", ["jpeg", "png", "bmp"])
    async def test_different_formats(
        self,
        image_analyzer: ImageAnalyzer,
        tmp_path: Path,
        image_format: str
    ):
        """Test handling of different image formats."""
        # Create test image in specified format
        image = Image.new('RGB', (100, 100), color='red')
        image_path = tmp_path / f"test.{image_format}"
        image.save(image_path, format=image_format)
        
        result = await image_analyzer.analyze_image(image_path)
        assert result is not None

    async def test_memory_usage(
        self,
        image_analyzer: ImageAnalyzer,
        create_test_image: callable
    ):
        """Test memory handling with large images."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process large image multiple times
        large_image_path = create_test_image(
            size=(2000, 2000),
            color=(100, 150, 200)
        )
        
        for _ in range(3):
            await image_analyzer.analyze_image(large_image_path)
            
        # Check memory hasn't grown significantly
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        assert memory_growth < 100 * 1024 * 1024  # 100MB limit

    async def test_dominant_colors_accuracy(
        self,
        image_analyzer: ImageAnalyzer,
        create_test_image: callable
    ):
        """Test accuracy of dominant color extraction."""
        # Create image with known color distribution
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        image_path = create_test_image(colors=colors)
        
        result = await image_analyzer.analyze_image(image_path)
        dominant = result["dominant_colors"]
        
        # Verify each target color is represented
        found_colors = [color["color"] for color in dominant]
        for target in colors:
            assert any(self._colors_similar(target, found)
                      for found in found_colors)

    async def test_image_composition(
        self,
        image_analyzer: ImageAnalyzer,
        create_test_image: callable
    ):
        """Test image composition analysis."""
        image_path = create_test_image(
            size=(300, 300),
            colors=[(0, 0, 0), (255, 255, 255)]
        )
        
        result = await image_analyzer.analyze_image_style(image_path)
        comp = result["composition"]
        
        assert "edge_density" in comp
        assert "thirds_strength" in comp
        assert "symmetry_score" in comp

    def _colors_similar(
        self,
        color1: Tuple[int, int, int],
        color2: Tuple[int, int, int],
        threshold: int = 30
    ) -> bool:
        """Helper to check if colors are similar within threshold."""
        return all(abs(c1 - c2) <= threshold
                  for c1, c2 in zip(color1, color2))

# Test Utilities
@pytest.fixture
def create_test_image(tmp_path: Path):
    """Fixture to create test images with specified parameters."""
    def _create_image(
        size: Tuple[int, int] = (100, 100),
        color: Tuple[int, int, int] = (128, 128, 128),
        colors: List[Tuple[int, int, int]] = None,
        filename: str = "test.jpg"
    ) -> Path:
        if colors:
            # Create image with multiple colors
            img = np.zeros((*size, 3), dtype=np.uint8)
            height, width = size
            n_colors = len(colors)
            for i, color in enumerate(colors):
                start = i * width // n_colors
                end = (i + 1) * width // n_colors
                img[:, start:end] = color
        else:
            # Create solid color image
            img = np.full((*size, 3), color, dtype=np.uint8)

        path = tmp_path / filename
        cv2.imwrite(str(path), img)
        return path

    return _create_image