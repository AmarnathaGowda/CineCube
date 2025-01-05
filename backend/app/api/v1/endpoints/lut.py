from typing import List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
import aiofiles
from pathlib import Path
import uuid
import json
import time

from app.core.config import settings
from app.services.llm.llama import LLaMAService
from app.services.image.analyzer import ImageAnalyzer
from app.services.lut.generator import LUTGenerator

from app.models.schemas.lut import (
    LUTRequest,
    LUTResponse,
    LUTPreset,
    LUTGenerationStatus,
    LUTMetadata
)

from app.api.dependencies import (
    get_llm_service,
    get_image_analyzer,
    get_lut_generator,
    verify_upload_path,
    verify_output_path,
    validate_file_size,
    validate_file_extension
)
from app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Store ongoing LUT generation tasks
active_tasks = {}

@router.post("/generate")
async def generate_lut(
    description: str = Form(...),
    reference_image: Optional[UploadFile] = File(None),
    preset_name: Optional[str] = Form(None),
    llm_service: LLaMAService = Depends(get_llm_service),
    image_analyzer: ImageAnalyzer = Depends(get_image_analyzer),
    lut_generator: LUTGenerator = Depends(get_lut_generator),
    upload_path: Path = Depends(verify_upload_path),
    output_path: Path = Depends(verify_output_path)
):
    """Generate custom LUT based on description and optional reference image."""
    try:
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Process reference image if provided
        image_analysis = None
        if reference_image:
            # Save and analyze image
            image_data = await reference_image.read()
            image_path = upload_path / f"{task_id}_{reference_image.filename}"
            async with aiofiles.open(image_path, 'wb') as f:
                await f.write(image_data)
            image_analysis = await image_analyzer.analyze_image(image_path)
            logger.info(f"Image analysis completed for {image_path}")

        # Create description with all inputs
        full_description = description
        if preset_name:
            full_description += f"\nStyle Reference: {preset_name}"
        if image_analysis:
            full_description += f"\nImage Analysis: {json.dumps(image_analysis)}"

        # Generate LUT parameters using LLM
        lut_params = await llm_service.process_description(
            description=full_description,
            image_analysis=image_analysis
        )
        logger.info(f"Generated LUT parameters for task {task_id}")

        # Generate .cube file
        output_file = output_path / f"{task_id}.cube"
        await lut_generator.generate(
            parameters=lut_params,
            output_path=output_file
        )
        logger.info(f"Generated LUT file: {output_file}")

        # Save the preset for future reference
        if preset_name:
            preset_dir = output_path / "presets"
            preset_dir.mkdir(parents=True, exist_ok=True)
            preset_data = {
                "name": preset_name,
                "description": description,
                "parameters": lut_params,
                "task_id": task_id
            }
            preset_file = preset_dir / f"{task_id}_{preset_name}.json"
            async with aiofiles.open(preset_file, 'w') as f:
                await f.write(json.dumps(preset_data, indent=2))
            logger.info(f"Saved preset {preset_name} for task {task_id}")

        # Return response
        return {
            "task_id": task_id,
            "status": "completed",
            "lut_file": str(output_file),
            "parameters": lut_params,
            "preset_name": preset_name if preset_name else None
        }

    except Exception as e:
        logger.error(f"LUT generation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate LUT: {str(e)}"
        )
    
@router.get("/status/{task_id}", response_model=LUTGenerationStatus)
async def get_generation_status(task_id: str):
    """Get the status of a LUT generation task."""
    if task_id not in active_tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    return active_tasks[task_id]

@router.get("/download/{task_id}")
async def download_lut(
    task_id: str,
    output_path: Path = Depends(verify_output_path)
):
    """Download generated LUT file."""
    lut_file = output_path / f"{task_id}.cube"
    if not lut_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="LUT file not found"
        )
    return FileResponse(
        path=lut_file,
        filename=f"{task_id}.cube",
        media_type="application/octet-stream"
    )


@router.get("/preview/{task_id}")
async def get_preview(
    task_id: str,
    output_path: Path = Depends(verify_output_path)
):
    """Get the preview image for a generated LUT."""
    preview_file = output_path / f"{task_id}_preview.png"
    
    if not preview_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Preview not found"
        )
    
    return FileResponse(
        path=preview_file,
        media_type="image/png"
    )

@router.get("/metadata/{task_id}", response_model=LUTMetadata)
async def get_metadata(
    task_id: str,
    output_path: Path = Depends(verify_output_path)
):
    """Get metadata for a generated LUT."""
    metadata_file = output_path / f"{task_id}_metadata.json"
    
    if not metadata_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Metadata not found"
        )
    
    async with aiofiles.open(metadata_file, 'r') as f:
        metadata = json.loads(await f.read())
    
    return metadata

@router.get("/presets", response_model=List[LUTPreset])
async def list_presets():
    """List available LUT presets."""
    presets_path = settings.get_output_path() / "presets"
    presets = []
    
    if presets_path.exists():
        for preset_file in presets_path.glob("*.json"):
            async with aiofiles.open(preset_file, 'r') as f:
                preset_data = json.loads(await f.read())
                presets.append(LUTPreset(**preset_data))
    
    return presets

@router.post("/presets", status_code=status.HTTP_201_CREATED)
async def create_preset(
    name: str = Form(...),
    description: str = Form(...),
    parameters: str = Form(...),
):
    """Create a new LUT preset."""
    presets_path = settings.get_output_path() / "presets"
    presets_path.mkdir(exist_ok=True)
    
    preset_file = presets_path / f"{name}.json"
    if preset_file.exists():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Preset already exists"
        )
    
    try:
        parameters_dict = json.loads(parameters)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid parameters JSON"
        )
    
    preset_data = {
        "name": name,
        "description": description,
        "parameters": parameters_dict
    }
    
    async with aiofiles.open(preset_file, 'w') as f:
        await f.write(json.dumps(preset_data, indent=2))
    
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={"message": "Preset created successfully"}
    )

# Utility functions
async def load_preset(preset_name: str) -> dict:
    """Load preset parameters from file."""
    preset_file = settings.get_output_path() / "presets" / f"{preset_name}.json"
    if not preset_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Preset '{preset_name}' not found"
        )
    
    async with aiofiles.open(preset_file, 'r') as f:
        preset_data = json.loads(await f.read())
    
    return preset_data["parameters"]

def merge_params(base_params: dict, preset_params: dict) -> dict:
    """Merge LLM parameters with preset parameters."""
    merged = base_params.copy()
    merged.update(preset_params)
    return merged

async def generate_preview(
    lut_data: str,
    task_id: str,
    output_path: Path
) -> Optional[str]:
    """Generate preview image for the LUT."""
    try:
        preview_file = output_path / f"{task_id}_preview.png"
        # Implement preview generation logic here
        return str(preview_file)
    except Exception as e:
        logger.error(f"Preview generation failed: {str(e)}")
        return None

async def save_metadata(metadata: LUTMetadata, file_path: Path):
    """Save LUT generation metadata to file."""
    async with aiofiles.open(file_path, 'w') as f:
        await f.write(json.dumps(metadata.dict(), indent=2))