from typing import Dict
from fastapi import APIRouter, Depends, status
from datetime import datetime
import psutil
import os
from pathlib import Path

from app.core.config import settings
from app.services.llm.llama import LLaMAService
from app.services.image.analyzer import ImageAnalyzer
from app.services.lut.generator import LUTGenerator
from app.api.dependencies import (
    get_llm_service,
    get_image_analyzer,
    get_lut_generator,
    verify_upload_path,
    verify_output_path
)
from app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

@router.get("/", status_code=status.HTTP_200_OK)
async def health_check() -> Dict:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.VERSION
    }

@router.get("/detailed", status_code=status.HTTP_200_OK)
async def detailed_health_check(
    llm_service: LLaMAService = Depends(get_llm_service),
    image_analyzer: ImageAnalyzer = Depends(get_image_analyzer),
    lut_generator: LUTGenerator = Depends(get_lut_generator)
) -> Dict:
    """Detailed health check for all system components."""
    try:
        # System metrics
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Check service health
        llm_status = await check_llm_health(llm_service)
        image_analyzer_status = await check_image_analyzer_health(image_analyzer)
        lut_generator_status = await check_lut_generator_health(lut_generator)
        
        # Check directories
        directory_status = await check_directory_health()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": settings.VERSION,
            "system": {
                "cpu_usage_percent": cpu_usage,
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent
                },
                "disk": {
                    "total": disk.total,
                    "free": disk.free,
                    "percent": disk.percent
                }
            },
            "services": {
                "llm": llm_status,
                "image_analyzer": image_analyzer_status,
                "lut_generator": lut_generator_status
            },
            "directories": directory_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

@router.get("/ready", status_code=status.HTTP_200_OK)
async def readiness_check() -> Dict:
    """Readiness probe for Kubernetes."""
    try:
        # Check if all required services are ready
        services_ready = await check_services_ready()
        
        if not services_ready:
            return {
                "status": "not_ready",
                "timestamp": datetime.utcnow().isoformat(),
                "message": "Some services are not ready"
            }
        
        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}", exc_info=True)
        return {
            "status": "not_ready",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

@router.get("/live", status_code=status.HTTP_200_OK)
async def liveness_check() -> Dict:
    """Liveness probe for Kubernetes."""
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }

# Utility functions for health checks
async def check_llm_health(llm_service: LLaMAService) -> Dict:
    """Check LLaMA service health."""
    try:
        # Perform a simple inference test
        test_result = await llm_service.process_description("test")
        return {
            "status": "healthy",
            "model_loaded": True
        }
    except Exception as e:
        logger.error(f"LLM health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

async def check_image_analyzer_health(image_analyzer: ImageAnalyzer) -> Dict:
    """Check image analyzer service health."""
    try:
        # Check if OpenCV is properly initialized
        test_image = image_analyzer.create_test_image()
        return {
            "status": "healthy",
            "opencv_initialized": True
        }
    except Exception as e:
        logger.error(f"Image analyzer health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

async def check_lut_generator_health(lut_generator: LUTGenerator) -> Dict:
    """Check LUT generator service health."""
    try:
        # Check if generator can create a simple LUT
        test_result = await lut_generator.generate_test_lut()
        return {
            "status": "healthy",
            "generator_initialized": True
        }
    except Exception as e:
        logger.error(f"LUT generator health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

async def check_directory_health() -> Dict:
    """Check if required directories are accessible and writable."""
    directories = {
        "uploads": settings.UPLOAD_DIR,
        "output": settings.OUTPUT_DIR,
        "logs": Path("logs")
    }
    
    status = {}
    for name, path in directories.items():
        try:
            # Check if directory exists and is writable
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            
            # Try to write a test file
            test_file = path / ".test"
            test_file.touch()
            test_file.unlink()
            
            status[name] = {
                "status": "healthy",
                "path": str(path),
                "writable": True
            }
        except Exception as e:
            status[name] = {
                "status": "unhealthy",
                "path": str(path),
                "error": str(e)
            }
    
    return status

async def check_services_ready() -> bool:
    """Check if all required services are ready."""
    try:
        # Check each service
        services = [
            get_llm_service(),
            get_image_analyzer(),
            get_lut_generator()
        ]
        
        for service in services:
            if not await service:
                return False
        
        return True
    except Exception:
        return False