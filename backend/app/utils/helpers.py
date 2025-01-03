import os
import json
import uuid
import hashlib
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import shutil
import aiofiles
import logging
from functools import wraps
import time

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# File System Utilities
async def ensure_directory(directory: Union[str, Path]) -> Path:
    """
    Ensure directory exists and create if necessary.
    
    Args:
        directory: Directory path
        
    Returns:
        Path: Directory path object
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path

async def clean_directory(directory: Union[str, Path], max_age_days: int = 7) -> None:
    """
    Clean old files from directory.
    
    Args:
        directory: Directory path
        max_age_days: Maximum file age in days
    """
    try:
        path = Path(directory)
        if not path.exists():
            return

        cutoff = time.time() - (max_age_days * 24 * 3600)
        
        for item in path.iterdir():
            if item.is_file() and item.stat().st_mtime < cutoff:
                item.unlink()
                logger.info(f"Deleted old file: {item}")
    except Exception as e:
        logger.error(f"Error cleaning directory: {str(e)}")

async def get_file_hash(file_path: Union[str, Path], algorithm: str = 'sha256') -> str:
    """
    Calculate file hash.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm to use
        
    Returns:
        str: File hash
    """
    hash_func = getattr(hashlib, algorithm)()
    
    async with aiofiles.open(file_path, 'rb') as f:
        while chunk := await f.read(8192):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()

# Time and Date Utilities
def get_utc_now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)

def format_timestamp(dt: datetime) -> str:
    """Format datetime as ISO string."""
    return dt.isoformat()

def parse_timestamp(timestamp: str) -> datetime:
    """Parse ISO format timestamp."""
    return datetime.fromisoformat(timestamp)

# JSON Utilities
async def save_json(data: Any, file_path: Union[str, Path]) -> None:
    """
    Save data as JSON file.
    
    Args:
        data: Data to save
        file_path: Output file path
    """
    async with aiofiles.open(file_path, 'w') as f:
        await f.write(json.dumps(data, indent=2))

async def load_json(file_path: Union[str, Path]) -> Any:
    """
    Load data from JSON file.
    
    Args:
        file_path: Input file path
        
    Returns:
        Any: Loaded data
    """
    async with aiofiles.open(file_path, 'r') as f:
        content = await f.read()
        return json.loads(content)

# Cache Utilities
class SimpleCache:
    """Simple in-memory cache implementation."""
    
    def __init__(self, ttl: int = 3600):
        self.cache = {}
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            del self.cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        self.cache[key] = (value, time.time())
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()

# Performance Monitoring
def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            duration = end_time - start_time
            logger.debug(
                f"Function {func.__name__} took {duration:.2f} seconds"
            )
    return wrapper

# Rate Limiting
class RateLimiter:
    """Simple rate limiter implementation."""
    
    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = {}
    
    async def is_allowed(self, key: str) -> bool:
        """Check if request is allowed under rate limit."""
        now = time.time()
        
        # Clean old requests
        if key in self.requests:
            self.requests[key] = [
                req_time for req_time in self.requests[key]
                if now - req_time < self.time_window
            ]
        else:
            self.requests[key] = []
        
        # Check rate limit
        if len(self.requests[key]) >= self.max_requests:
            return False
        
        # Add new request
        self.requests[key].append(now)
        return True

# String Utilities
def generate_unique_id() -> str:
    """Generate unique identifier."""
    return str(uuid.uuid4())

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    return "".join(c for c in filename if c.isalnum() or c in "._-")

# Error Handling
def safe_execution(default: Any = None):
    """Decorator for safe function execution with default return."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Error in {func.__name__}: {str(e)}",
                    exc_info=True
                )
                return default
        return wrapper
    return decorator

# Environment Utilities
def is_development() -> bool:
    """Check if running in development environment."""
    return settings.ENVIRONMENT == "development"

def is_production() -> bool:
    """Check if running in production environment."""
    return settings.ENVIRONMENT == "production"

# Data Validation
def validate_required_fields(data: Dict, required_fields: List[str]) -> bool:
    """
    Validate required fields in data.
    
    Args:
        data: Data dictionary to validate
        required_fields: List of required field names
        
    Returns:
        bool: Whether all required fields are present
    """
    return all(field in data for field in required_fields)

# Resource Management
class ResourceManager:
    """Context manager for resource cleanup."""
    
    def __init__(self, resource_path: Union[str, Path]):
        self.path = Path(resource_path)
        self.created = False
    
    async def __aenter__(self):
        if not self.path.exists():
            await ensure_directory(self.path)
            self.created = True
        return self.path
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.created and self.path.exists():
            shutil.rmtree(self.path)

# Parameter Processing
def normalize_parameters(params: Dict[str, float], min_val: float, max_val: float) -> Dict[str, float]:
    """
    Normalize parameter values to specified range.
    
    Args:
        params: Parameters to normalize
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        dict: Normalized parameters
    """
    return {
        key: min_val + (val - min_val) * (max_val - min_val)
        for key, val in params.items()
    }