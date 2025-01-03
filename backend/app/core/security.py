from datetime import datetime, timedelta
from typing import Optional, Union
import jwt
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer
from fastapi import HTTPException, Security, status
import secrets
import hashlib
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 configuration
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# API key store (replace with database in production)
api_keys = {}

def create_access_token(
    subject: Union[str, int],
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create JWT access token."""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    
    to_encode = {"exp": expire, "sub": str(subject)}
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm="HS256"
    )
    return encoded_jwt

def verify_token(token: str) -> Optional[str]:
    """Verify JWT token."""
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=["HS256"]
        )
        return payload.get("sub")
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)

async def get_api_key(api_key: str = Security(oauth2_scheme)) -> Optional[str]:
    """Validate API key."""
    if api_key in api_keys:
        return api_key
    return None

def create_api_key() -> str:
    """Generate new API key."""
    api_key = secrets.token_urlsafe(32)
    api_keys[api_key] = {
        "created_at": datetime.utcnow(),
        "last_used": None
    }
    return api_key

def revoke_api_key(api_key: str) -> bool:
    """Revoke an API key."""
    if api_key in api_keys:
        del api_keys[api_key]
        return True
    return False

class SecurityUtils:
    """Security utility functions."""
    
    @staticmethod
    def hash_content(content: Union[str, bytes]) -> str:
        """Create SHA-256 hash of content."""
        if isinstance(content, str):
            content = content.encode()
        return hashlib.sha256(content).hexdigest()
    
    @staticmethod
    def generate_random_token(length: int = 32) -> str:
        """Generate random secure token."""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent path traversal."""
        return "".join(c for c in filename if c.isalnum() or c in "._-")
    
    @staticmethod
    def validate_file_hash(file_content: bytes, provided_hash: str) -> bool:
        """Validate file content against provided hash."""
        computed_hash = SecurityUtils.hash_content(file_content)
        return secrets.compare_digest(computed_hash, provided_hash)

class RateLimiter:
    """Rate limiting implementation."""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
    
    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed under rate limit."""
        now = datetime.utcnow()
        
        # Clean old entries
        self._cleanup(now)
        
        # Get or create record
        if key not in self.requests:
            self.requests[key] = []
        
        # Check limit
        requests = self.requests[key]
        if len(requests) >= self.max_requests:
            return False
        
        # Add request
        requests.append(now)
        return True
    
    def _cleanup(self, now: datetime):
        """Remove old requests from tracking."""
        cutoff = now - timedelta(seconds=self.window_seconds)
        
        for key in list(self.requests.keys()):
            self.requests[key] = [
                req for req in self.requests[key]
                if req > cutoff
            ]
            if not self.requests[key]:
                del self.requests[key]

class FileValidator:
    """File validation utilities."""
    
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.raw'}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    @staticmethod
    def is_valid_extension(filename: str) -> bool:
        """Check if file extension is allowed."""
        return any(filename.lower().endswith(ext) for ext in FileValidator.ALLOWED_EXTENSIONS)
    
    @staticmethod
    def is_valid_size(file_size: int) -> bool:
        """Check if file size is within limits."""
        return file_size <= FileValidator.MAX_FILE_SIZE
    
    @staticmethod
    async def validate_file(filename: str, content: bytes) -> bool:
        """Validate both filename and content."""
        if not FileValidator.is_valid_extension(filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File type not allowed"
            )
        
        if not FileValidator.is_valid_size(len(content)):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File size too large"
            )
        
        return True

# Create rate limiter instance
rate_limiter = RateLimiter(
    max_requests=settings.RATE_LIMIT_REQUESTS,
    window_seconds=settings.RATE_LIMIT_PERIOD
)