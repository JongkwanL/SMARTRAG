"""
Middleware for FastAPI application.

This module provides middleware for authentication, rate limiting, CORS,
request logging, error handling, and security.
"""

import asyncio
import logging
import time
import json
import traceback
from typing import Callable, Dict, Any, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
import uuid

from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint
import jwt
from passlib.context import CryptContext

from ..core.config import get_settings
from ..core.exceptions import SmartRAGError, ValidationError, AuthenticationError
from .models import ErrorResponse, ErrorDetail

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging."""
    
    def __init__(
        self,
        app: FastAPI,
        log_body: bool = False,
        log_headers: bool = False,
        max_body_size: int = 1000
    ):
        """
        Initialize request logging middleware.
        
        Args:
            app: FastAPI application
            log_body: Whether to log request/response bodies
            log_headers: Whether to log headers
            max_body_size: Maximum body size to log
        """
        super().__init__(app)
        self.log_body = log_body
        self.log_headers = log_headers
        self.max_body_size = max_body_size
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request and log details."""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Log request
        await self._log_request(request, request_id)
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Add timing headers
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = request_id
        
        # Log response
        await self._log_response(request, response, process_time, request_id)
        
        return response
    
    async def _log_request(self, request: Request, request_id: str):
        """Log incoming request."""
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if self.log_headers:
            log_data["headers"] = dict(request.headers)
        
        if self.log_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if len(body) <= self.max_body_size:
                    log_data["body"] = body.decode("utf-8")
                else:
                    log_data["body"] = f"<truncated: {len(body)} bytes>"
            except Exception as e:
                log_data["body_error"] = str(e)
        
        logger.info(f"Incoming request: {json.dumps(log_data)}")
    
    async def _log_response(
        self,
        request: Request,
        response: Response,
        process_time: float,
        request_id: str
    ):
        """Log outgoing response."""
        log_data = {
            "request_id": request_id,
            "status_code": response.status_code,
            "process_time": process_time,
            "response_size": response.headers.get("content-length", 0),
        }
        
        if self.log_headers:
            log_data["headers"] = dict(response.headers)
        
        # Log level based on status code
        if response.status_code >= 500:
            logger.error(f"Response error: {json.dumps(log_data)}")
        elif response.status_code >= 400:
            logger.warning(f"Response client error: {json.dumps(log_data)}")
        else:
            logger.info(f"Response success: {json.dumps(log_data)}")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware with multiple strategies."""
    
    def __init__(
        self,
        app: FastAPI,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_size: int = 10,
        enable_burst: bool = True
    ):
        """
        Initialize rate limiting middleware.
        
        Args:
            app: FastAPI application
            requests_per_minute: Requests per minute limit
            requests_per_hour: Requests per hour limit
            burst_size: Burst allowance
            enable_burst: Whether to enable burst limiting
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_size = burst_size
        self.enable_burst = enable_burst
        
        # Storage for rate limit data
        self.minute_buckets: Dict[str, deque] = defaultdict(deque)
        self.hour_buckets: Dict[str, deque] = defaultdict(deque)
        self.burst_buckets: Dict[str, deque] = defaultdict(deque)
        
        # Cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_expired())
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """Check rate limits and process request."""
        client_id = self._get_client_id(request)
        current_time = time.time()
        
        # Check rate limits
        if not await self._check_rate_limits(client_id, current_time):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )
        
        # Record request
        await self._record_request(client_id, current_time)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        await self._add_rate_limit_headers(response, client_id, current_time)
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Try to get authenticated user ID
        if hasattr(request.state, 'user_id'):
            return f"user:{request.state.user_id}"
        
        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"
    
    async def _check_rate_limits(self, client_id: str, current_time: float) -> bool:
        """Check if request is within rate limits."""
        # Check burst limit
        if self.enable_burst:
            burst_requests = self.burst_buckets[client_id]
            # Remove old burst requests (last 10 seconds)
            while burst_requests and current_time - burst_requests[0] > 10:
                burst_requests.popleft()
            
            if len(burst_requests) >= self.burst_size:
                return False
        
        # Check minute limit
        minute_requests = self.minute_buckets[client_id]
        while minute_requests and current_time - minute_requests[0] > 60:
            minute_requests.popleft()
        
        if len(minute_requests) >= self.requests_per_minute:
            return False
        
        # Check hour limit
        hour_requests = self.hour_buckets[client_id]
        while hour_requests and current_time - hour_requests[0] > 3600:
            hour_requests.popleft()
        
        if len(hour_requests) >= self.requests_per_hour:
            return False
        
        return True
    
    async def _record_request(self, client_id: str, current_time: float):
        """Record request for rate limiting."""
        if self.enable_burst:
            self.burst_buckets[client_id].append(current_time)
        
        self.minute_buckets[client_id].append(current_time)
        self.hour_buckets[client_id].append(current_time)
    
    async def _add_rate_limit_headers(
        self,
        response: Response,
        client_id: str,
        current_time: float
    ):
        """Add rate limit headers to response."""
        minute_remaining = max(0, self.requests_per_minute - len(self.minute_buckets[client_id]))
        hour_remaining = max(0, self.requests_per_hour - len(self.hour_buckets[client_id]))
        
        response.headers["X-RateLimit-Limit-Minute"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining-Minute"] = str(minute_remaining)
        response.headers["X-RateLimit-Limit-Hour"] = str(self.requests_per_hour)
        response.headers["X-RateLimit-Remaining-Hour"] = str(hour_remaining)
    
    async def _cleanup_expired(self):
        """Background task to cleanup expired rate limit data."""
        while True:
            try:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                current_time = time.time()
                
                # Cleanup minute buckets
                for client_id in list(self.minute_buckets.keys()):
                    bucket = self.minute_buckets[client_id]
                    while bucket and current_time - bucket[0] > 60:
                        bucket.popleft()
                    if not bucket:
                        del self.minute_buckets[client_id]
                
                # Cleanup hour buckets
                for client_id in list(self.hour_buckets.keys()):
                    bucket = self.hour_buckets[client_id]
                    while bucket and current_time - bucket[0] > 3600:
                        bucket.popleft()
                    if not bucket:
                        del self.hour_buckets[client_id]
                
                # Cleanup burst buckets
                for client_id in list(self.burst_buckets.keys()):
                    bucket = self.burst_buckets[client_id]
                    while bucket and current_time - bucket[0] > 10:
                        bucket.popleft()
                    if not bucket:
                        del self.burst_buckets[client_id]
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in rate limit cleanup: {e}")


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """JWT-based authentication middleware."""
    
    def __init__(
        self,
        app: FastAPI,
        secret_key: str,
        algorithm: str = "HS256",
        exempt_paths: Optional[Set[str]] = None
    ):
        """
        Initialize authentication middleware.
        
        Args:
            app: FastAPI application
            secret_key: JWT secret key
            algorithm: JWT algorithm
            exempt_paths: Paths exempt from authentication
        """
        super().__init__(app)
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.exempt_paths = exempt_paths or {
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/auth/login",
            "/auth/register"
        }
        self.security = HTTPBearer(auto_error=False)
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """Check authentication and process request."""
        path = request.url.path
        
        # Skip authentication for exempt paths
        if path in self.exempt_paths or path.startswith("/static/"):
            return await call_next(request)
        
        # Extract and verify token
        credentials = await self.security(request)
        if not credentials:
            return self._unauthorized_response("Missing authentication token")
        
        try:
            payload = jwt.decode(
                credentials.credentials,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            # Add user info to request state
            request.state.user_id = payload.get("sub")
            request.state.user_email = payload.get("email")
            request.state.user_roles = payload.get("roles", [])
            request.state.token_exp = payload.get("exp")
            
            # Check token expiration
            if payload.get("exp", 0) < time.time():
                return self._unauthorized_response("Token expired")
            
        except jwt.InvalidTokenError as e:
            return self._unauthorized_response(f"Invalid token: {e}")
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return self._unauthorized_response("Authentication failed")
        
        return await call_next(request)
    
    def _unauthorized_response(self, message: str) -> JSONResponse:
        """Create unauthorized response."""
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "status": "error",
                "error_code": "UNAUTHORIZED",
                "error_message": message
            },
            headers={"WWW-Authenticate": "Bearer"}
        )


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Security headers middleware."""
    
    def __init__(self, app: FastAPI):
        """Initialize security headers middleware."""
        super().__init__(app)
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """Add security headers to response."""
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'"
        )
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware."""
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """Handle errors and create consistent error responses."""
        try:
            return await call_next(request)
            
        except HTTPException as e:
            # FastAPI HTTP exceptions
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "status": "error",
                    "error_code": "HTTP_ERROR",
                    "error_message": e.detail,
                    "request_id": getattr(request.state, 'request_id', None),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except SmartRAGError as e:
            # Custom application errors
            status_code = self._get_status_code_for_error(e)
            
            error_response = ErrorResponse(
                error_code=e.__class__.__name__,
                error_message=str(e),
                request_id=getattr(request.state, 'request_id', None),
                details=[
                    ErrorDetail(
                        code=e.__class__.__name__,
                        message=str(e),
                        context=getattr(e, 'context', {})
                    )
                ]
            )
            
            return JSONResponse(
                status_code=status_code,
                content=error_response.dict()
            )
            
        except Exception as e:
            # Unexpected errors
            logger.error(f"Unexpected error: {e}", exc_info=True)
            
            error_response = ErrorResponse(
                error_code="INTERNAL_ERROR",
                error_message="An unexpected error occurred",
                request_id=getattr(request.state, 'request_id', None),
                details=[
                    ErrorDetail(
                        code="INTERNAL_ERROR",
                        message=str(e) if logger.level <= logging.DEBUG else "Internal server error",
                        context={"traceback": traceback.format_exc()} if logger.level <= logging.DEBUG else {}
                    )
                ]
            )
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=error_response.dict()
            )
    
    def _get_status_code_for_error(self, error: SmartRAGError) -> int:
        """Get appropriate HTTP status code for error type."""
        if isinstance(error, ValidationError):
            return status.HTTP_400_BAD_REQUEST
        elif isinstance(error, AuthenticationError):
            return status.HTTP_401_UNAUTHORIZED
        elif hasattr(error, 'status_code'):
            return error.status_code
        else:
            return status.HTTP_500_INTERNAL_SERVER_ERROR


def setup_cors(app: FastAPI, settings=None):
    """Setup CORS middleware."""
    if settings is None:
        settings = get_settings()
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Process-Time", "X-Request-ID"]
    )


def setup_middleware(app: FastAPI):
    """Setup all middleware for the application."""
    settings = get_settings()
    
    # Error handling (should be first)
    app.add_middleware(ErrorHandlingMiddleware)
    
    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Authentication
    if settings.enable_auth:
        app.add_middleware(
            AuthenticationMiddleware,
            secret_key=settings.jwt_secret_key,
            algorithm=settings.jwt_algorithm
        )
    
    # Rate limiting
    if settings.enable_rate_limiting:
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=settings.rate_limit_per_minute,
            requests_per_hour=settings.rate_limit_per_hour,
            burst_size=settings.rate_limit_burst
        )
    
    # Request logging
    app.add_middleware(
        RequestLoggingMiddleware,
        log_body=settings.log_request_body,
        log_headers=settings.log_headers
    )
    
    # CORS (should be last)
    setup_cors(app, settings)
    
    logger.info("Middleware setup complete")