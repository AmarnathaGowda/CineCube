from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator, EmailStr, HttpUrl
from datetime import datetime
from uuid import UUID
import re

class ResponseStatus(BaseModel):
    """Standard response status model."""
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(..., description="Response message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ErrorResponse(BaseModel):
    """Standard error response model."""
    error_code: str = Field(..., description="Unique error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class PaginationParams(BaseModel):
    """Standard pagination parameters."""
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(10, ge=1, le=100, description="Items per page")
    sort_by: Optional[str] = Field(None, description="Field to sort by")
    sort_order: Optional[str] = Field("asc", description="Sort order (asc/desc)")

    @validator("sort_order")
    def validate_sort_order(cls, v):
        if v and v.lower() not in ["asc", "desc"]:
            raise ValueError("Sort order must be 'asc' or 'desc'")
        return v.lower()

class PaginatedResponse(BaseModel):
    """Standard paginated response."""
    items: List[Any]
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")

class FileMetadata(BaseModel):
    """File metadata model."""
    filename: str
    size: int = Field(..., description="File size in bytes")
    content_type: str
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow)
    md5_hash: Optional[str] = None

    @validator("filename")
    def validate_filename(cls, v):
        if not re.match(r"^[\w\-. ]+$", v):
            raise ValueError("Invalid filename format")
        return v

class UserAgent(BaseModel):
    """User agent information."""
    browser: Optional[str] = None
    browser_version: Optional[str] = None
    os: Optional[str] = None
    device: Optional[str] = None
    is_mobile: bool = False

class RequestMetadata(BaseModel):
    """Request metadata model."""
    request_id: UUID
    client_ip: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_agent: Optional[UserAgent] = None
    referer: Optional[str] = None

class LogEntry(BaseModel):
    """Log entry model."""
    level: str = Field(..., description="Log level")
    message: str = Field(..., description="Log message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None
    trace_id: Optional[str] = None

class HealthStatus(BaseModel):
    """Health check status model."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details: Optional[Dict[str, Any]] = None

class APIKey(BaseModel):
    """API key model."""
    key: str = Field(..., description="API key")
    name: str = Field(..., description="Key name")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    is_active: bool = True
    permissions: List[str] = Field(default_factory=list)

class EmailConfig(BaseModel):
    """Email configuration model."""
    smtp_host: str
    smtp_port: int = Field(..., ge=1, le=65535)
    smtp_user: str
    smtp_password: str
    use_tls: bool = True
    from_email: EmailStr
    from_name: Optional[str] = None

class WebhookConfig(BaseModel):
    """Webhook configuration model."""
    url: HttpUrl
    method: str = Field("POST", description="HTTP method")
    headers: Optional[Dict[str, str]] = None
    retry_count: int = Field(3, ge=0, le=10)
    timeout: int = Field(30, ge=1, le=300)
    events: List[str] = Field(..., description="Events to trigger webhook")

class CacheConfig(BaseModel):
    """Cache configuration model."""
    cache_type: str = Field("memory", description="Cache type (memory/redis)")
    ttl: int = Field(3600, ge=0, description="Time to live in seconds")
    max_size: Optional[int] = Field(None, description="Maximum cache size")
    redis_url: Optional[str] = None

    @validator("cache_type")
    def validate_cache_type(cls, v):
        if v not in ["memory", "redis"]:
            raise ValueError("Cache type must be 'memory' or 'redis'")
        return v

class DateTimeRange(BaseModel):
    """Date time range model."""
    start: datetime
    end: datetime

    @validator("end")
    def validate_end_date(cls, v, values):
        if "start" in values and v < values["start"]:
            raise ValueError("End date must be after start date")
        return v

class ResourceUsage(BaseModel):
    """Resource usage statistics."""
    cpu_percent: float = Field(..., ge=0, le=100)
    memory_used: int = Field(..., ge=0)
    memory_total: int = Field(..., ge=0)
    disk_used: int = Field(..., ge=0)
    disk_total: int = Field(..., ge=0)
    network_in: int = Field(..., ge=0)
    network_out: int = Field(..., ge=0)

class TaskStatus(BaseModel):
    """Task status model."""
    task_id: UUID
    status: str = Field(..., description="Task status")
    progress: Optional[float] = Field(None, ge=0, le=100)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None