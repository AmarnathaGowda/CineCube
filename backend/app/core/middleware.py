# app/core/middleware.py

import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from app.core.logger import get_logger

logger = get_logger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging request details."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request details and process the request."""
        
        # Generate request ID
        request_id = str(time.time())
        
        # Log request details
        logger.info(
            "Incoming request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else None,
            }
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Log response details
            logger.info(
                "Request completed",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                }
            )
            
            return response
            
        except Exception as e:
            logger.exception(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                }
            )
            raise

class ResponseTimeMiddleware(BaseHTTPMiddleware):
    """Middleware for tracking response times."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Track and log response time."""
        
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log if response time exceeds threshold
        if process_time > 1.0:  # 1 second threshold
            logger.warning(
                "Slow response detected",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "process_time": process_time,
                }
            )
        
        return response

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for consistent error handling."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle errors and provide consistent error responses."""
        
        try:
            response = await call_next(request)
            return response
            
        except Exception as e:
            logger.exception("Unhandled error", exc_info=e)
            
            # Create error response
            error_response = {
                "detail": "Internal server error",
                "path": request.url.path,
                "timestamp": time.time(),
            }
            
            if isinstance(e, ValueError):
                return Response(
                    content=str(error_response),
                    status_code=400,
                    media_type="application/json"
                )
            
            return Response(
                content=str(error_response),
                status_code=500,
                media_type="application/json"
            )