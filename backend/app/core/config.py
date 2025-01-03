from typing import Any, Dict, List, Optional, Union
from pydantic import AnyHttpUrl, BaseSettings, EmailStr, validator, HttpUrl, PostgresDsn, RedisDsn
import secrets
from pathlib import Path
import os
import json

class Settings(BaseSettings):
    """Application settings and configuration."""
    
    # API Settings
    PROJECT_NAME: str = "LUT Generator"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    RELOAD: bool = False
    DEBUG: bool = False
    
    # Security
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    ALGORITHM: str = "HS256"
    REQUIRE_API_KEY: bool = False
    
    # CORS Configuration
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost:3000",  # React frontend
        "http://localhost:8000",  # Local development
    ]

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    # File Storage
    UPLOAD_DIR: Path = Path("uploads")
    OUTPUT_DIR: Path = Path("output")
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".raw"]
    
    @validator("UPLOAD_DIR", "OUTPUT_DIR", pre=True)
    def create_directories(cls, v: Path) -> Path:
        os.makedirs(v, exist_ok=True)
        return v
    
    # LLaMA Model Settings
    LLAMA_MODEL_PATH: Path = Path("models/llama-2-7b-chat.gguf")
    LLAMA_THREADS: int = 4
    LLAMA_CONTEXT_SIZE: int = 2048
    LLAMA_BATCH_SIZE: int = 512
    LLAMA_GPU_LAYERS: int = 0

    @validator("LLAMA_MODEL_PATH")
    def validate_model_path(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Model file not found at {v}")
        return v
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    LOG_FILE: Optional[Path] = Path("logs/app.log")
    ENABLE_ACCESS_LOG: bool = True

    @validator("LOG_FILE")
    def create_log_directory(cls, v: Optional[Path]) -> Optional[Path]:
        if v:
            os.makedirs(v.parent, exist_ok=True)
        return v
    
    # Cache Settings
    CACHE_TYPE: str = "memory"  # Options: memory, redis
    CACHE_REDIS_URL: Optional[RedisDsn] = None
    CACHE_DEFAULT_TIMEOUT: int = 600  # 10 minutes

    @validator("CACHE_TYPE")
    def validate_cache_type(cls, v: str, values: Dict[str, Any]) -> str:
        if v not in ["memory", "redis"]:
            raise ValueError("Cache type must be 'memory' or 'redis'")
        if v == "redis" and not values.get("CACHE_REDIS_URL"):
            raise ValueError("Redis URL must be provided when using Redis cache")
        return v
    
    # Database Settings (Optional)
    DATABASE_URL: Optional[PostgresDsn] = None
    DATABASE_MAX_CONNECTIONS: int = 10
    DATABASE_POOL_SIZE: int = 5
    
    # Email Configuration
    SMTP_TLS: bool = True
    SMTP_PORT: Optional[int] = None
    SMTP_HOST: Optional[str] = None
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    EMAILS_FROM_EMAIL: Optional[EmailStr] = None
    EMAILS_FROM_NAME: Optional[str] = None
    
    # Error Reporting
    ENABLE_SENTRY: bool = False
    SENTRY_DSN: Optional[HttpUrl] = None

    @validator("SENTRY_DSN")
    def validate_sentry(cls, v: Optional[HttpUrl], values: Dict[str, Any]) -> Optional[HttpUrl]:
        if values.get("ENABLE_SENTRY") and not v:
            raise ValueError("Sentry DSN must be provided when Sentry is enabled")
        return v
    
    # Performance and Resource Limits
    MAX_WORKERS: int = 4
    MAX_CONCURRENT_TASKS: int = 10
    TASK_TIMEOUT: int = 300  # 5 minutes
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: int = 3600  # 1 hour
    
    # LUT Generation Settings
    LUT_SIZE: int = 32  # Size of the LUT cube (32x32x32)
    LUT_FORMAT: str = "cube"  # Options: cube, 3dl
    ENABLE_PREVIEW: bool = True
    PREVIEW_SIZE: tuple = (800, 600)
    
    # Environment Settings
    ENVIRONMENT: str = "development"  # Options: development, staging, production

    @validator("ENVIRONMENT")
    def validate_environment(cls, v: str) -> str:
        if v not in ["development", "staging", "production"]:
            raise ValueError("Invalid environment")
        return v
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9000
    
    class Config:
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def get_environment_settings(self) -> Dict[str, Any]:
        """Get settings based on environment."""
        return {
            "development": {
                "DEBUG": True,
                "RELOAD": True,
                "LOG_LEVEL": "DEBUG"
            },
            "staging": {
                "DEBUG": False,
                "RELOAD": False,
                "LOG_LEVEL": "INFO"
            },
            "production": {
                "DEBUG": False,
                "RELOAD": False,
                "LOG_LEVEL": "WARNING",
                "REQUIRE_API_KEY": True
            }
        }.get(self.ENVIRONMENT, {})
    
    def update_environment_settings(self) -> None:
        """Update settings based on environment."""
        env_settings = self.get_environment_settings()
        for key, value in env_settings.items():
            setattr(self, key, value)
    
    def export_settings(self, exclude_secrets: bool = True) -> Dict[str, Any]:
        """Export settings as dictionary."""
        settings_dict = {}
        for key, value in self.dict().items():
            if exclude_secrets and any(secret in key.lower() for secret in ["secret", "password", "token"]):
                continue
            if isinstance(value, Path):
                settings_dict[key] = str(value)
            else:
                settings_dict[key] = value
        return settings_dict
    
    def save_settings(self, file_path: Union[str, Path]) -> None:
        """Save settings to file."""
        settings_dict = self.export_settings()
        with open(file_path, 'w') as f:
            json.dump(settings_dict, f, indent=2)

# Create settings instance
settings = Settings()

# Update environment-specific settings
settings.update_environment_settings()

# Validate settings on import
def validate_settings() -> None:
    """Validate critical settings."""
    required_dirs = [
        settings.UPLOAD_DIR,
        settings.OUTPUT_DIR,
        settings.LOG_FILE.parent if settings.LOG_FILE else None
    ]
    
    for directory in required_dirs:
        if directory and not directory.exists():
            raise ValueError(f"Required directory {directory} does not exist")
    
    if settings.ENVIRONMENT == "production":
        if settings.DEBUG:
            raise ValueError("Debug mode cannot be enabled in production")
        if not settings.SECRET_KEY or len(settings.SECRET_KEY) < 32:
            raise ValueError("Secure secret key required in production")

# Run validation
validate_settings()