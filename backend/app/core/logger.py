import logging
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import logging.handlers
from pythonjsonlogger import jsonlogger

from app.core.config import settings

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for logging."""
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        """Add custom fields to the log record."""
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp
        log_record['timestamp'] = datetime.utcnow().isoformat()
        
        # Add log level
        log_record['level'] = record.levelname
        
        # Add environment
        log_record['environment'] = settings.ENVIRONMENT
        
        # Add version
        log_record['version'] = settings.VERSION
        
        # Add trace ID if available
        if hasattr(record, 'trace_id'):
            log_record['trace_id'] = record.trace_id

class RequestIdFilter(logging.Filter):
    """Filter to add request ID to log records."""
    
    def __init__(self, request_id: Optional[str] = None):
        super().__init__()
        self.request_id = request_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = self.request_id
        return True

def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create logger
    logger = logging.getLogger("app")
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))

    # Remove existing handlers
    logger.handlers = []

    # Create formatters
    if settings.LOG_FORMAT.lower() == "json":
        formatter = CustomJsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if settings.LOG_FILE:
        file_handler = logging.handlers.RotatingFileHandler(
            settings.LOG_FILE,
            maxBytes=10485760,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def get_logger(name: str) -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(f"app.{name}")

class RequestLogger:
    """Logger for API requests."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log_request(
        self,
        request_id: str,
        method: str,
        path: str,
        status_code: int,
        duration: float
    ) -> None:
        """Log API request details."""
        self.logger.info(
            "API Request",
            extra={
                "request_id": request_id,
                "method": method,
                "path": path,
                "status_code": status_code,
                "duration_ms": round(duration * 1000, 2)
            }
        )

    def log_error(
        self,
        request_id: str,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log API error details."""
        error_details = {
            "request_id": request_id,
            "error_type": type(error).__name__,
            "error_message": str(error)
        }
        if context:
            error_details.update(context)
        
        self.logger.error(
            "API Error",
            extra=error_details,
            exc_info=True
        )

class AsyncLogger:
    """Asynchronous logger for background tasks."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log_task_start(self, task_id: str, task_type: str) -> None:
        """Log task start."""
        self.logger.info(
            "Task Started",
            extra={
                "task_id": task_id,
                "task_type": task_type,
                "status": "started"
            }
        )

    def log_task_complete(
        self,
        task_id: str,
        task_type: str,
        duration: float
    ) -> None:
        """Log task completion."""
        self.logger.info(
            "Task Completed",
            extra={
                "task_id": task_id,
                "task_type": task_type,
                "status": "completed",
                "duration_ms": round(duration * 1000, 2)
            }
        )

    def log_task_error(
        self,
        task_id: str,
        task_type: str,
        error: Exception
    ) -> None:
        """Log task error."""
        self.logger.error(
            "Task Error",
            extra={
                "task_id": task_id,
                "task_type": task_type,
                "status": "error",
                "error_type": type(error).__name__,
                "error_message": str(error)
            },
            exc_info=True
        )

# Initialize loggers
logger = setup_logging()
request_logger = RequestLogger(logger)
async_logger = AsyncLogger(logger)

# Export loggers
__all__ = ['logger', 'get_logger', 'request_logger', 'async_logger']