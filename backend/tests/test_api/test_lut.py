import pytest
from fastapi.testclient import TestClient
import json
from pathlib import Path
import io
from PIL import Image
import uuid

pytestmark = pytest.mark.asyncio

class TestLUTEndpoints:
    """Test cases for LUT generation endpoints."""

    async def test_generate_lut_text_only(
        self,
        client: TestClient,
        sample_lut_params: dict,
        assert_json_response
    ):
        """Test LUT generation with text description only."""
        response = client.post(
            "/api/v1/lut/generate",
            data={
                "description": "Create a warm, vintage film look with slightly lifted blacks"
            }
        )
        
        assert_json_response(
            response,
            status_code=200,
            expected_keys=["task_id", "status", "lut_file"]
        )

    async def test_generate_lut_with_image(
        self,
        client: TestClient,
        sample_image: bytes,
        assert_json_response
    ):
        """Test LUT generation with both text and reference image."""
        response = client.post(
            "/api/v1/lut/generate",
            data={
                "description": "Match the color grading of the reference image"
            },
            files={
                "reference_image": ("test_image.jpg", sample_image, "image/jpeg")
            }
        )
        
        assert_json_response(
            response,
            status_code=200,
            expected_keys=["task_id", "status", "lut_file", "preview_url"]
        )

    async def test_generate_lut_with_preset(
        self,
        client: TestClient,
        sample_lut_params: dict,
        assert_json_response
    ):
        """Test LUT generation using a preset."""
        # First create a preset
        preset_response = client.post(
            "/api/v1/lut/presets",
            data={
                "name": "test_preset",
                "description": "Test preset",
                "parameters": json.dumps(sample_lut_params)
            }
        )
        assert preset_response.status_code == 201

        # Generate LUT using preset
        response = client.post(
            "/api/v1/lut/generate",
            data={
                "description": "Apply test preset with slight modifications",
                "preset_name": "test_preset"
            }
        )
        
        assert_json_response(
            response,
            status_code=200,
            expected_keys=["task_id", "status", "lut_file"]
        )

    async def test_invalid_image_format(
        self,
        client: TestClient,
        assert_error_response
    ):
        """Test error handling for invalid image format."""
        # Create invalid image data
        invalid_image = b"invalid image data"
        
        response = client.post(
            "/api/v1/lut/generate",
            data={"description": "Test description"},
            files={
                "reference_image": ("test.txt", invalid_image, "text/plain")
            }
        )
        
        assert_error_response(
            response,
            status_code=415,
            error_type="FILE_FORMAT_ERROR"
        )

    async def test_large_image_file(
        self,
        client: TestClient,
        assert_error_response
    ):
        """Test error handling for oversized image file."""
        # Create large image
        large_image = Image.new('RGB', (5000, 5000))
        img_byte_arr = io.BytesIO()
        large_image.save(img_byte_arr, format='JPEG')
        
        response = client.post(
            "/api/v1/lut/generate",
            data={"description": "Test description"},
            files={
                "reference_image": ("large.jpg", img_byte_arr.getvalue(), "image/jpeg")
            }
        )
        
        assert_error_response(
            response,
            status_code=413,
            error_type="FILE_TOO_LARGE"
        )

    async def test_missing_description(
        self,
        client: TestClient,
        assert_error_response
    ):
        """Test error handling for missing description."""
        response = client.post(
            "/api/v1/lut/generate",
            data={}
        )
        
        assert_error_response(
            response,
            status_code=422,
            error_type="VALIDATION_ERROR"
        )

    async def test_get_generation_status(
        self,
        client: TestClient,
        assert_json_response
    ):
        """Test retrieving LUT generation status."""
        # First generate a LUT
        gen_response = client.post(
            "/api/v1/lut/generate",
            data={"description": "Test description"}
        )
        task_id = gen_response.json()["task_id"]
        
        # Check status
        response = client.get(f"/api/v1/lut/status/{task_id}")
        
        assert_json_response(
            response,
            status_code=200,
            expected_keys=["task_id", "status", "progress"]
        )

    async def test_invalid_task_id(
        self,
        client: TestClient,
        assert_error_response
    ):
        """Test error handling for invalid task ID."""
        invalid_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/lut/status/{invalid_id}")
        
        assert_error_response(
            response,
            status_code=404,
            error_type="TASK_NOT_FOUND"
        )

    async def test_download_lut(
        self,
        client: TestClient
    ):
        """Test downloading generated LUT file."""
        # First generate a LUT
        gen_response = client.post(
            "/api/v1/lut/generate",
            data={"description": "Test description"}
        )
        task_id = gen_response.json()["task_id"]
        
        # Download LUT
        response = client.get(f"/api/v1/lut/download/{task_id}")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/octet-stream"
        assert "content-disposition" in response.headers

    async def test_get_preview(
        self,
        client: TestClient
    ):
        """Test retrieving LUT preview image."""
        # First generate a LUT with reference image
        gen_response = client.post(
            "/api/v1/lut/generate",
            data={"description": "Test description"},
            files={
                "reference_image": ("test.jpg", sample_image, "image/jpeg")
            }
        )
        task_id = gen_response.json()["task_id"]
        
        # Get preview
        response = client.get(f"/api/v1/lut/preview/{task_id}")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    async def test_list_presets(
        self,
        client: TestClient,
        assert_json_response
    ):
        """Test listing available LUT presets."""
        response = client.get("/api/v1/lut/presets")
        
        assert_json_response(
            response,
            status_code=200,
            expected_keys=["presets"]
        )

    async def test_create_preset(
        self,
        client: TestClient,
        sample_lut_params: dict
    ):
        """Test creating a new LUT preset."""
        response = client.post(
            "/api/v1/lut/presets",
            data={
                "name": "new_preset",
                "description": "New test preset",
                "parameters": json.dumps(sample_lut_params)
            }
        )
        
        assert response.status_code == 201

    async def test_duplicate_preset_name(
        self,
        client: TestClient,
        sample_lut_params: dict,
        assert_error_response
    ):
        """Test error handling for duplicate preset names."""
        # Create first preset
        client.post(
            "/api/v1/lut/presets",
            data={
                "name": "duplicate_preset",
                "description": "Test preset",
                "parameters": json.dumps(sample_lut_params)
            }
        )
        
        # Try to create duplicate
        response = client.post(
            "/api/v1/lut/presets",
            data={
                "name": "duplicate_preset",
                "description": "Duplicate preset",
                "parameters": json.dumps(sample_lut_params)
            }
        )
        
        assert_error_response(
            response,
            status_code=409,
            error_type="PRESET_EXISTS"
        )

    async def test_get_metadata(
        self,
        client: TestClient,
        assert_json_response
    ):
        """Test retrieving LUT generation metadata."""
        # First generate a LUT
        gen_response = client.post(
            "/api/v1/lut/generate",
            data={"description": "Test description"}
        )
        task_id = gen_response.json()["task_id"]
        
        # Get metadata
        response = client.get(f"/api/v1/lut/metadata/{task_id}")
        
        assert_json_response(
            response,
            status_code=200,
            expected_keys=["task_id", "description", "creation_time"]
        )