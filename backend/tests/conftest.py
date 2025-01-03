import os
import pytest
import asyncio
from typing import AsyncGenerator, Generator
import shutil
from pathlib import Path
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import io

# Import application components
from app.main import app
from app.core.config import Settings, settings
from app.services.llm.llama import LLaMAService
from app.services.image.analyzer import ImageAnalyzer
from app.services.lut.generator import LUTGenerator

# Test settings override
def get_test_settings() -> Settings:
    """Create test settings with overrides."""
    return Settings(
        ENVIRONMENT="test",
        DEBUG=True,
        TESTING=True,
        UPLOAD_DIR="tests/uploads",
        OUTPUT_DIR="tests/output",
        LOG_FILE="tests/logs/test.log",
        LLAMA_MODEL_PATH="tests/models/test_model.gguf",
        SECRET_KEY="test_secret_key",
        REQUIRE_API_KEY=False
    )

@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Fixture for test settings."""
    return get_test_settings()

@pytest.fixture(scope="session")
def test_app(test_settings):
    """Fixture for test application."""
    app.dependency_overrides = {}  # Reset any existing overrides
    return app

@pytest.fixture(scope="session")
def client(test_app) -> Generator:
    """Fixture for test client."""
    with TestClient(test_app) as test_client:
        yield test_client

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Directory fixtures
@pytest.fixture(scope="session", autouse=True)
def setup_test_directories(test_settings):
    """Create and clean test directories."""
    # Create test directories
    os.makedirs(test_settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(test_settings.OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(test_settings.LOG_FILE), exist_ok=True)
    
    yield
    
    # Cleanup after tests
    shutil.rmtree(test_settings.UPLOAD_DIR, ignore_errors=True)
    shutil.rmtree(test_settings.OUTPUT_DIR, ignore_errors=True)
    shutil.rmtree(os.path.dirname(test_settings.LOG_FILE), ignore_errors=True)

# Service fixtures
@pytest.fixture
async def llm_service() -> AsyncGenerator[LLaMAService, None]:
    """Fixture for LLaMA service."""
    service = LLaMAService()
    yield service
    # Cleanup if needed
    del service

@pytest.fixture
async def image_analyzer() -> AsyncGenerator[ImageAnalyzer, None]:
    """Fixture for image analyzer service."""
    analyzer = ImageAnalyzer()
    yield analyzer
    # Cleanup if needed
    del analyzer

@pytest.fixture
async def lut_generator() -> AsyncGenerator[LUTGenerator, None]:
    """Fixture for LUT generator service."""
    generator = LUTGenerator()
    yield generator
    # Cleanup if needed
    del generator

# Test data fixtures
@pytest.fixture
def sample_image() -> bytes:
    """Create sample test image."""
    # Create a simple RGB test image
    img = Image.new('RGB', (100, 100), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

@pytest.fixture
def sample_lut_params() -> dict:
    """Create sample LUT parameters."""
    return {
        "temperature": 20,
        "tint": -5,
        "saturation": 10,
        "contrast": 15,
        "highlights": 5,
        "shadows": -10,
        "whites": 0,
        "blacks": -15,
        "color_balance": {
            "shadows": {
                "red": -5,
                "green": 0,
                "blue": 5
            },
            "midtones": {
                "red": 0,
                "green": 0,
                "blue": 0
            },
            "highlights": {
                "red": 5,
                "green": 0,
                "blue": -5
            }
        }
    }

@pytest.fixture
def sample_image_metadata() -> dict:
    """Create sample image metadata."""
    return {
        "filename": "test_image.jpg",
        "size": 1024,
        "content_type": "image/jpeg",
        "width": 100,
        "height": 100,
        "color_space": "RGB"
    }

# Mock fixtures
@pytest.fixture
def mock_llm_response() -> dict:
    """Mock LLM response data."""
    return {
        "temperature": 15,
        "tint": -3,
        "saturation": 8,
        "contrast": 12,
        "highlights": 4,
        "shadows": -8,
        "whites": 0,
        "blacks": -12
    }

@pytest.fixture
def mock_image_analysis() -> dict:
    """Mock image analysis data."""
    return {
        "color_distribution": {
            "red_histogram": [0] * 256,
            "green_histogram": [0] * 256,
            "blue_histogram": [0] * 256
        },
        "contrast": {
            "global_contrast": 0.5,
            "local_contrast": 0.3
        },
        "brightness": {
            "average_brightness": 128,
            "brightness_std": 20
        }
    }

# Utility fixtures
@pytest.fixture
def create_temp_file(tmp_path):
    """Utility fixture to create temporary files."""
    def _create_file(content: bytes, filename: str) -> Path:
        file_path = tmp_path / filename
        file_path.write_bytes(content)
        return file_path
    return _create_file

@pytest.fixture
def assert_json_response():
    """Utility fixture for JSON response assertions."""
    def _assert_json(response, status_code: int, expected_keys: list):
        assert response.status_code == status_code
        assert response.headers["content-type"] == "application/json"
        data = response.json()
        for key in expected_keys:
            assert key in data
    return _assert_json

# Error fixtures
@pytest.fixture
def assert_error_response():
    """Utility fixture for error response assertions."""
    def _assert_error(response, status_code: int, error_type: str):
        assert response.status_code == status_code
        data = response.json()
        assert "error" in data
        assert data["error"]["error_code"] == error_type
    return _assert_error

# Cleanup
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up after each test."""
    yield
    # Clean any temporary files or resources