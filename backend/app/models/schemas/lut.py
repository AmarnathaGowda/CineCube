from pydantic import BaseModel, Field
from typing import Dict, Optional, List
from datetime import datetime

class LUTRequest(BaseModel):
    """Request model for LUT generation."""
    description: str = Field(..., description="Text description of desired color grading")
    preset_name: Optional[str] = Field(None, description="Name of preset to apply")

class LUTGenerationStatus(BaseModel):
    """Status model for LUT generation task."""
    task_id: str
    status: str = Field(..., description="Current status: pending, processing, completed, failed")
    progress: Optional[float] = Field(None, description="Progress percentage (0-100)")
    message: Optional[str] = Field(None, description="Status message or error details")
    start_time: datetime
    end_time: Optional[datetime] = None

class LUTMetadata(BaseModel):
    """Metadata for generated LUT."""
    task_id: str
    description: str
    creation_time: float
    preset_name: Optional[str] = None
    has_reference_image: bool
    parameters: Optional[Dict] = None
    tags: List[str] = []

class LUTPreset(BaseModel):
    """Model for LUT presets."""
    name: str
    description: str
    parameters: Dict
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class LUTResponse(BaseModel):
    """Response model for LUT generation."""
    task_id: str
    status: str
    lut_file: Optional[str] = None
    preview_url: Optional[str] = None
    metadata: Optional[LUTMetadata] = None

    class Config:
        schema_extra = {
            "example": {
                "task_id": "123e4567-e89b-12d3-a456-426614174000",
                "status": "completed",
                "lut_file": "/output/123e4567-e89b-12d3-a456-426614174000.cube",
                "preview_url": "/preview/123e4567-e89b-12d3-a456-426614174000.png",
                "metadata": {
                    "task_id": "123e4567-e89b-12d3-a456-426614174000",
                    "description": "Warm vintage film look",
                    "creation_time": 1635724800.0,
                    "preset_name": "vintage_warm",
                    "has_reference_image": True,
                    "parameters": {
                        "temperature": 5500,
                        "tint": 10,
                        "contrast": 1.2
                    },
                    "tags": ["vintage", "warm", "film"]
                }
            }
        }