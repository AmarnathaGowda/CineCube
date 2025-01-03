from typing import Any, Dict, Optional
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import sys
import traceback
from datetime import datetime

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

class AppError(Exception):
    """Base application error."""
    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: Optional[str] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(self.message)

class LUTGenerationError(AppError):
    """Error during LUT generation."""
    def __init__(self, message: str):
        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="LUT_GENERATION_ERROR"
        )

class ModelError(AppError):
    """Error with LLM model."""
    def __init__(self, message: str):
        super().__init__(
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="MODEL_ERROR"
        )

class FileProcessingError(AppError):
    """Error processing uploaded files."""
    def __init__(self, message: str):
        super().__init__(
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="FILE_PROCESSING_ERROR"
        )

class RateLimitError(AppError):
    """Rate limit exceeded."""
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code="RATE_LIMIT_ERROR"
        )

def create_error_response(
    status_code: int,
    message: str,
    error_code: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create standardized error response."""
    return {
        "error": {
            "status_code": status_code,
            "message": message,
            "error_code": error_code,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details
        }
    }

def setup_exception_handlers(app: FastAPI) -> None:
    """Setup all exception handlers for the application."""
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions."""
        logger.warning(
            f"HTTP error occurred",
            extra={
                "status_code": exc.status_code,
                "detail": str(exc.detail),
                "path": request.url.path
            }
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=create_error_response(
                status_code=exc.status_code,
                message=str(exc.detail),
                error_code="HTTP_ERROR"
            )
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors."""
        logger.warning(
            f"Validation error occurred",
            extra={
                "errors": exc.errors(),
                "body": exc.body,
                "path": request.url.path
            }
        )
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=create_error_response(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                message="Validation error",
                error_code="VALIDATION_ERROR",
                details={"errors": exc.errors()}
            )
        )
    
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        """Handle application-specific errors."""
        logger.error(
            f"Application error occurred: {exc.message}",
            extra={
                "error_code": exc.error_code,
                "path": request.url.path
            }
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=create_error_response(
                status_code=exc.status_code,
                message=exc.message,
                error_code=exc.error_code
            )
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle all unhandled exceptions."""
        logger.error(
            f"Unhandled exception occurred: {str(exc)}",
            extra={
                "traceback": "".join(traceback.format_exception(*sys.exc_info())),
                "path": request.url.path
            }
        )
        
        message = "Internal server error"
        if settings.DEBUG:
            message = str(exc)
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=create_error_response(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                message=message,
                error_code="INTERNAL_ERROR",
                details={"type": exc.__class__.__name__} if settings.DEBUG else None
            )
        )

# Error handling middlewares
async def error_logging_middleware(request: Request, call_next):
    """Middleware for logging all errors."""
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(
            f"Error in request: {str(e)}",
            extra={
                "path": request.url.path,
                "method": request.method,
                "client_ip": request.client.host
            },
            exc_info=True
        )
        raise

async def error_handling_middleware(request: Request, call_next):
    """Middleware for handling errors and providing consistent responses."""
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        if isinstance(e, AppError):
            return JSONResponse(
                status_code=e.status_code,
                content=create_error_response(
                    status_code=e.status_code,
                    message=e.message,
                    error_code=e.error_code
                )
            )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=create_error_response(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                message="Internal server error",
                error_code="INTERNAL_ERROR"
            )
        )

# Utility functions for error handling
def handle_file_error(error: Exception) -> None:
    """Handle file processing errors."""
    if isinstance(error, IOError):
        raise FileProcessingError("Error processing file")
    raise error

def handle_model_error(error: Exception) -> None:
    """Handle LLM model errors."""
    if isinstance(error, RuntimeError):
        raise ModelError("Model processing error")
    raise error

def handle_generation_error(error: Exception) -> None:
    """Handle LUT generation errors."""
    if isinstance(error, ValueError):
        raise LUTGenerationError("Invalid parameters for LUT generation")
    raise error