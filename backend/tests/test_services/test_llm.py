import pytest
from unittest.mock import Mock, patch
import numpy as np
from pathlib import Path
import json

from app.services.llm.llama import LLaMAService
from app.core.errors import ModelError
from app.services.llm.prompt import PromptManager

pytestmark = pytest.mark.asyncio

class TestLLaMAService:
    """Test cases for LLaMA service."""

    async def test_process_description_basic(self, llm_service: LLaMAService):
        """Test basic description processing."""
        description = "Create a warm, vintage film look with slightly lifted blacks"
        
        result = await llm_service.process_description(description)
        
        # Verify result structure
        assert isinstance(result, dict)
        assert all(key in result for key in [
            "temperature",
            "tint",
            "saturation",
            "contrast",
            "highlights",
            "shadows",
            "whites",
            "blacks"
        ])
        
        # Verify value ranges
        for value in result.values():
            assert -100 <= float(value) <= 100

    async def test_process_description_complex(self, llm_service: LLaMAService):
        """Test processing of complex description."""
        description = """
        Create a moody cinematic grade with:
        - Teal and orange color palette
        - Deep crushed blacks
        - Slightly desaturated midtones
        - Warm highlights
        """
        
        result = await llm_service.process_description(description)
        
        # Check for color balance
        assert "color_balance" in result
        assert all(zone in result["color_balance"] 
                  for zone in ["shadows", "midtones", "highlights"])

    async def test_analyze_style(self, llm_service: LLaMAService):
        """Test style analysis functionality."""
        description = "Create a high contrast noir look with deep shadows"
        
        result = await llm_service.analyze_style(description)
        
        assert isinstance(result, dict)
        assert "style_type" in result
        assert "color_palette" in result
        assert "mood" in result

    @pytest.mark.parametrize("invalid_description", [
        "",  # Empty string
        " ",  # Whitespace
        "a" * 10000,  # Too long
    ])
    async def test_invalid_descriptions(
        self,
        llm_service: LLaMAService,
        invalid_description: str
    ):
        """Test handling of invalid descriptions."""
        with pytest.raises(ModelError):
            await llm_service.process_description(invalid_description)

    async def test_response_validation(self, llm_service: LLaMAService):
        """Test validation of model responses."""
        description = "Simple test description"
        
        result = await llm_service.process_description(description)
        
        # Validate parameter ranges
        for param in result:
            if isinstance(result[param], (int, float)):
                assert -100 <= result[param] <= 100

    async def test_model_error_handling(self):
        """Test handling of model initialization errors."""
        with patch('llama_cpp.Llama', side_effect=Exception("Model error")):
            with pytest.raises(ModelError) as exc_info:
                LLaMAService()
            assert "Model initialization failed" in str(exc_info.value)

    async def test_process_description_with_preset(
        self,
        llm_service: LLaMAService,
        sample_lut_params: dict
    ):
        """Test processing description with preset parameters."""
        description = "Apply vintage look with custom adjustments"
        
        # Mock preset parameters
        preset_params = {
            "temperature": 20,
            "tint": -5,
            "saturation": 10
        }
        
        with patch.object(llm_service, '_merge_parameters') as mock_merge:
            await llm_service.process_description(
                description,
                preset_params=preset_params
            )
            mock_merge.assert_called_once()

    async def test_concurrent_processing(self, llm_service: LLaMAService):
        """Test concurrent description processing."""
        descriptions = [
            "Create a warm vintage look",
            "Create a cool modern look",
            "Create a high contrast look"
        ]
        
        # Process descriptions concurrently
        results = await asyncio.gather(*[
            llm_service.process_description(desc)
            for desc in descriptions
        ])
        
        assert len(results) == len(descriptions)
        assert all(isinstance(r, dict) for r in results)

    @pytest.mark.parametrize("description,expected_style", [
        ("Warm vintage film look", "vintage"),
        ("Modern teal and orange grade", "cinematic"),
        ("Black and white noir style", "noir"),
        ("Natural documentary look", "documentary")
    ])
    async def test_style_detection(
        self,
        llm_service: LLaMAService,
        description: str,
        expected_style: str
    ):
        """Test detection of different style types."""
        result = await llm_service.analyze_style(description)
        assert result["style_type"].lower() == expected_style

    async def test_temperature_tint_correlation(self, llm_service: LLaMAService):
        """Test correlation between temperature and tint parameters."""
        description = "Create a very warm sunset look"
        
        result = await llm_service.process_description(description)
        
        # Warm look should have positive temperature
        assert result["temperature"] > 0
        # Verify tint compensation
        assert -30 <= result["tint"] <= 30

    async def test_model_cleanup(self):
        """Test proper cleanup of model resources."""
        service = LLaMAService()
        
        # Mock cleanup method
        with patch.object(service.model, '__del__') as mock_cleanup:
            del service
            mock_cleanup.assert_called_once()

    @pytest.mark.parametrize("prompt_type", [
        "lut_generation",
        "style_analysis",
        "color_matching"
    ])
    async def test_prompt_templates(
        self,
        llm_service: LLaMAService,
        prompt_type: str
    ):
        """Test different prompt templates."""
        description = "Test description"
        prompt = PromptManager.get_prompt(prompt_type, description=description)
        
        assert isinstance(prompt, str)
        assert description in prompt

    async def test_parameter_consistency(self, llm_service: LLaMAService):
        """Test consistency of generated parameters."""
        description = "Test description"
        
        # Generate parameters multiple times
        results = []
        for _ in range(3):
            result = await llm_service.process_description(description)
            results.append(result)
        
        # Check consistency within reasonable bounds
        for param in results[0]:
            if isinstance(results[0][param], (int, float)):
                values = [r[param] for r in results]
                std_dev = np.std(values)
                assert std_dev < 20  # Allow some variation but not too much

    async def test_save_load_parameters(
        self,
        llm_service: LLaMAService,
        tmp_path: Path
    ):
        """Test saving and loading of generated parameters."""
        description = "Test description"
        result = await llm_service.process_description(description)
        
        # Save parameters
        save_path = tmp_path / "params.json"
        with open(save_path, 'w') as f:
            json.dump(result, f)
        
        # Load parameters
        with open(save_path, 'r') as f:
            loaded = json.load(f)
        
        assert result == loaded

    async def test_memory_cleanup(self, llm_service: LLaMAService):
        """Test memory cleanup after processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process multiple descriptions
        for _ in range(5):
            await llm_service.process_description("Test description")
        
        # Check memory usage hasn't grown significantly
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Allow for some memory growth but not excessive
        assert memory_growth < 100 * 1024 * 1024  # 100MB limit