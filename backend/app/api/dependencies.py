from typing import AsyncGenerator, Dict, Optional, Any
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
import time
from datetime import datetime
import aiofiles
from pathlib import Path

from app.core.config import settings
from app.services.llm.llama import LLaMAService
from app.services.image.analyzer import ImageAnalyzer
from app.services.lut.generator import LUTGenerator
from app.core.logger import get_logger

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


async def verify_upload_path() -> Path:
    """Ensure upload directory exists and return path."""
    try:
        path = Path(settings.UPLOAD_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception as e:
        logger.error(f"Failed to verify upload path: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not access upload directory"
        )

async def verify_output_path() -> Path:
    """Ensure output directory exists and return path."""
    try:
        path = Path(settings.OUTPUT_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception as e:
        logger.error(f"Failed to verify output path: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not access output directory"
        )

# Service dependency instances
llama_service = LLaMAService()
image_analyzer = ImageAnalyzer()
lut_generator = LUTGenerator()

# Rate limiting setup
rate_limit_store: Dict[str, Dict] = {}

async def get_llm_service() -> LLaMAService:
    """Dependency for LLaMA service."""
    try:
        return llama_service
    except Exception as e:
        logger.error(f"Failed to initialize LLaMA service: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM service is currently unavailable"
        )

async def get_image_analyzer() -> ImageAnalyzer:
    """Dependency for image analysis service."""
    try:
        return image_analyzer
    except Exception as e:
        logger.error(f"Failed to initialize image analyzer: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Image analysis service is currently unavailable"
        )

async def get_lut_generator() -> LUTGenerator:
    """Dependency for LUT generator service."""
    try:
        return lut_generator
    except Exception as e:
        logger.error(f"Failed to initialize LUT generator: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LUT generation service is currently unavailable"
        )

async def verify_upload_path() -> Path:
    """Ensure upload directory exists."""
    upload_path = settings.get_upload_path()
    upload_path.mkdir(parents=True, exist_ok=True)
    return upload_path

async def verify_output_path() -> Path:
    """Ensure output directory exists."""
    output_path = settings.get_output_path()
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path

async def check_rate_limit(request: Request) -> None:
    """Rate limiting middleware."""
    client_ip = request.client.host
    current_time = time.time()
    
    # Initialize or cleanup old rate limit data
    if client_ip not in rate_limit_store:
        rate_limit_store[client_ip] = {
            "requests": 0,
            "window_start": current_time
        }
    
    # Check if we need to reset the window
    window_start = rate_limit_store[client_ip]["window_start"]
    if current_time - window_start >= settings.RATE_LIMIT_PERIOD:
        rate_limit_store[client_ip] = {
            "requests": 0,
            "window_start": current_time
        }
    
    # Check rate limit
    if rate_limit_store[client_ip]["requests"] >= settings.RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    # Increment request count
    rate_limit_store[client_ip]["requests"] += 1

async def validate_file_size(file_size: int) -> None:
    """Validate uploaded file size."""
    if file_size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum limit of {settings.MAX_UPLOAD_SIZE} bytes"
        )

async def validate_file_extension(filename: str) -> None:
    """Validate file extension."""
    ext = Path(filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File extension {ext} not allowed. Allowed extensions: {settings.ALLOWED_EXTENSIONS}"
        )

async def save_upload_file(upload_path: Path, filename: str, file_data: bytes) -> Path:
    """Save uploaded file to disk."""
    try:
        file_path = upload_path / filename
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file_data)
        return file_path
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file"
        )

class RateLimitMiddleware:
    """Middleware for rate limiting."""
    
    async def __call__(self, request: Request, call_next):
        await check_rate_limit(request)
        return await call_next(request)

class RequestValidationMiddleware:
    """Middleware for request validation."""
    
    async def __call__(self, request: Request, call_next):
        # Add request validation logic here
        return await call_next(request)

class ResponseHeaderMiddleware:
    """Middleware for adding response headers."""
    
    async def __call__(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Server-Time"] = datetime.utcnow().isoformat()
        return response

class CacheMiddleware:
    """Middleware for response caching."""
    
    def __init__(self):
        self.cache = {}
    
    async def __call__(self, request: Request, call_next):
        if request.method == "GET":
            cache_key = request.url.path
            cached_response = self.cache.get(cache_key)
            
            if cached_response:
                return cached_response
        
        response = await call_next(request)
        
        if request.method == "GET":
            self.cache[cache_key] = response
        
        return response

# Dependency functions for common operations
async def get_current_time() -> datetime:
    """Get current UTC timestamp."""
    return datetime.utcnow()

async def get_request_metadata(request: Request) -> Dict:
    """Get metadata about the current request."""
    return {
        "client_ip": request.client.host,
        "method": request.method,
        "path": request.url.path,
        "timestamp": datetime.utcnow().isoformat()
    }

# Optional OAuth2 setup (if needed)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Get current user from token (placeholder)."""
    # Implement user authentication logic here
    pass