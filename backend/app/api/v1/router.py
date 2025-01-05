from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from typing import List

from app.api.v1.endpoints import health, lut
from app.core.config import settings
from app.core.security import get_api_key
from app.core.logger import get_logger

logger = get_logger(__name__)

# Create API router
api_router = APIRouter()

# Optional API key security scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key_header(api_key_header: str = Security(api_key_header)):
    """Validate API key if enabled."""
    if settings.REQUIRE_API_KEY:
        if not api_key_header:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key is required"
            )
        
        valid_key = await get_api_key(api_key_header)
        if not valid_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
    return api_key_header




# Health check routes
api_router.include_router(
    health.router,
    prefix="/health",
    tags=["health"]
)

# LUT generation routes
api_router.include_router(
    lut.router,
    prefix="/lut",
    tags=["lut"]
)

# Current version route
@api_router.get(
    "/version",
    tags=["system"],
    dependencies=[Depends(get_api_key_header)] if settings.REQUIRE_API_KEY else []
)
async def get_version():
    """Get current API version."""
    return {
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT
    }

# System status route
@api_router.get(
    "/status",
    tags=["system"],
    dependencies=[Depends(get_api_key_header)] if settings.REQUIRE_API_KEY else []
)
async def get_system_status():
    """Get system status information."""
    try:
        return {
            "status": "operational",
            "api_version": settings.VERSION,
            "environment": settings.ENVIRONMENT,
            "features": {
                "lut_generation": True,
                "image_analysis": True,
                "llm_processing": True
            },
            "endpoints": {
                "health": "/api/v1/health",
                "lut": "/api/v1/lut",
                "version": "/api/v1/version"
            }
        }
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error getting system status"
        )

# Documentation routes
@api_router.get(
    "/docs/endpoints",
    tags=["documentation"],
    dependencies=[Depends(get_api_key_header)] if settings.REQUIRE_API_KEY else []
)
async def get_endpoints_documentation() -> List[dict]:
    """Get documentation for all available endpoints."""
    return [
        {
            "path": "/api/v1/health",
            "methods": ["GET"],
            "description": "Health check endpoints",
            "endpoints": [
                {
                    "path": "/",
                    "method": "GET",
                    "description": "Basic health check"
                },
                {
                    "path": "/detailed",
                    "method": "GET",
                    "description": "Detailed system health check"
                }
            ]
        },
        {
            "path": "/api/v1/lut",
            "methods": ["POST", "GET"],
            "description": "LUT generation endpoints",
            "endpoints": [
                {
                    "path": "/generate",
                    "method": "POST",
                    "description": "Generate a new LUT"
                },
                {
                    "path": "/status/{task_id}",
                    "method": "GET",
                    "description": "Get LUT generation status"
                },
                {
                    "path": "/download/{task_id}",
                    "method": "GET",
                    "description": "Download generated LUT file"
                }
            ]
        }
    ]

# # Error handlers
# @api_router.exception_handler(HTTPException)
# async def http_exception_handler(request, exc):
#     """Handle HTTP exceptions."""
#     return JSONResponse(
#         status_code=exc.status_code,
#         content={
#             "status_code": exc.status_code,
#             "detail": exc.detail,
#             "headers": getattr(exc, "headers", None)
#         }
#     )

# @api_router.exception_handler(Exception)
# async def general_exception_handler(request, exc):
#     """Handle general exceptions."""
#     logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
#     return JSONResponse(
#         status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#         content={
#             "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
#             "detail": "Internal server error"
#         }
#     )

# Development routes (only available in non-production environments)


if not settings.is_production() != "production":
    @api_router.get("/debug/config", tags=["debug"])
    async def get_debug_config():
        """Get current configuration (non-sensitive settings)."""
        return {
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT,
            "debug_mode": settings.DEBUG,
            "cors_origins": settings.BACKEND_CORS_ORIGINS,
            "upload_dir": str(settings.UPLOAD_DIR),
            "output_dir": str(settings.OUTPUT_DIR),
            "workers": settings.WORKERS,
            "reload": settings.RELOAD
        }

    @api_router.post("/debug/error", tags=["debug"])
    async def trigger_test_error():
        """Endpoint to test error handling."""
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Test error triggered"
        )