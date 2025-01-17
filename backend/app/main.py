import logging
import os
from datetime import datetime
from typing import Any, Dict
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logger import setup_logging
from app.core.middleware import (
    RequestLoggingMiddleware,
    ResponseTimeMiddleware,
    ErrorHandlingMiddleware
)


# Setup logging
logger = setup_logging()

def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        "static",
        "uploads",
        "output",
        "logs",
        settings.UPLOAD_DIR,
        settings.OUTPUT_DIR
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    ensure_directories()
    
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description="LUT Generator API with LLM and Image Analysis",
        version=settings.VERSION,
        openapi_tags=settings.TAGS_METADATA
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add custom middleware
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(ResponseTimeMiddleware)
    app.add_middleware(ErrorHandlingMiddleware)

    # Mount static directories
    app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
    app.mount("/output", StaticFiles(directory="output"), name="output")

    return app
# Create FastAPI instance
app = create_application()

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Custom OpenAPI documentation
# OpenAPI Documentation
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="LUT Generator API Documentation",
        routes=app.routes,
    )
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        title=f"{settings.PROJECT_NAME} - Swagger UI"
    )

@app.get(f"{settings.API_V1_STR}/openapi.json", include_in_schema=False)
async def get_openapi_endpoint():
    return app.openapi_schema

@app.get("/openapi.json", include_in_schema=False)
async def get_openapi_endpoint():
    return JSONResponse(custom_openapi())

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting up LUT Generator API")
    
    # Create required directories
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Initialize services if needed
    # Example: database connection, cache initialization, etc.

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down LUT Generator API")
    
    # Cleanup resources
    # Example: close database connections, cleanup temporary files

# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle all unhandled exceptions."""
    logger.exception("Unhandled exception occurred", exc_info=exc)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "timestamp": datetime.utcnow().isoformat(),
            "path": request.url.path
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting up LUT Generator API")
    ensure_directories()

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down LUT Generator API")

# Main entry point for running the application
def start():
    """Start the application using uvicorn."""
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=settings.RELOAD,
        workers=settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
    )

# Development server with hot reload
def dev():
    """Start the development server with hot reload."""
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=True,
        log_level="debug",
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD
    )