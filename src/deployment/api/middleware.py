"""
API Middleware
==============

Custom middleware for the FastAPI application including logging,
authentication, rate limiting, and request tracking.
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable, Dict, Any
import time
import json
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging all requests and responses.
    
    This middleware captures detailed information about each request
    and response for monitoring and debugging purposes.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and log details.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware in chain
            
        Returns:
            Response: HTTP response
        """
        # Start timer
        start_time = time.time()
        
        # Capture request details
        request_id = request.headers.get("X-Request-ID", "unknown")
        method = request.method
        url = str(request.url)
        client_host = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Log request
        logger.info(
            f"Request started - ID: {request_id}, Method: {method}, "
            f"URL: {url}, Client: {client_host}, UA: {user_agent}"
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"Request completed - ID: {request_id}, Status: {response.status_code}, "
                f"Time: {process_time:.3f}s"
            )
            
            # Add processing time header
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            # Log error
            process_time = time.time() - start_time
            logger.error(
                f"Request failed - ID: {request_id}, Error: {str(e)}, "
                f"Time: {process_time:.3f}s",
                exc_info=True
            )
            raise

class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for authenticating requests.
    
    Validates API tokens and adds user information to request state.
    """
    
    def __init__(self, app, auth_service=None):
        """
        Initialize authentication middleware.
        
        Args:
            app: FastAPI application
            auth_service: Authentication service for token validation
        """
        super().__init__(app)
        self.auth_service = auth_service
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Authenticate the request.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware in chain
            
        Returns:
            Response: HTTP response
        """
        # Skip authentication for public endpoints
        public_paths = ["/health", "/ready", "/docs", "/redoc", "/openapi.json"]
        
        if any(request.url.path.startswith(path) for path in public_paths):
            return await call_next(request)
        
        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization")
        
        if not auth_header or not auth_header.startswith("Bearer "):
            # Return 401 for missing token
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=401,
                content={"error": "Missing or invalid authentication token"}
            )
        
        token = auth_header.replace("Bearer ", "")
        
        # Validate token
        try:
            # In production, validate against auth service
            user_info = await self._validate_token(token)
            
            # Add user info to request state
            request.state.user = user_info
            
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=403,
                content={"error": "Invalid authentication token"}
            )
        
        # Continue processing
        response = await call_next(request)
        return response
    
    async def _validate_token(self, token: str) -> Dict[str, Any]:
        """
        Validate authentication token.
        
        Args:
            token: JWT or API token
            
        Returns:
            dict: User information
            
        Raises:
            Exception: If token is invalid
        """
        # This is a simplified example
        # In production, call auth service or validate JWT
        
        if len(token) < 10:
            raise Exception("Invalid token")
        
        # Extract user ID from token (simplified)
        user_id = token.split("_")[-1] if "_" in token else "unknown"
        
        return {
            "user_id": user_id,
            "customer_id": f"cust_{user_id}",
            "role": "customer",
            "permissions": ["predict", "explain"]
        }

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate limiting requests.
    
    Limits the number of requests per client IP within a time window.
    """
    
    def __init__(self, app, requests_per_minute: int = 60, window_seconds: int = 60):
        """
        Initialize rate limit middleware.
        
        Args:
            app: FastAPI application
            requests_per_minute: Maximum requests per minute per client
            window_seconds: Time window in seconds
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.window_seconds = window_seconds
        self.clients = {}  # In production, use Redis
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Apply rate limiting to the request.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware in chain
            
        Returns:
            Response: HTTP response
        """
        # Skip rate limiting for public endpoints
        if request.url.path in ["/health", "/ready"]:
            return await call_next(request)
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Clean up old entries
        self._cleanup_old_entries(current_time)
        
        # Get client request history
        if client_ip not in self.clients:
            self.clients[client_ip] = []
        
        client_history = self.clients[client_ip]
        
        # Check rate limit
        if len(client_history) >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too Many Requests",
                    "message": f"Rate limit of {self.requests_per_minute} requests per minute exceeded"
                },
                headers={
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(current_time + self.window_seconds))
                }
            )
        
        # Add current request timestamp
        client_history.append(current_time)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, self.requests_per_minute - len(client_history))
        )
        response.headers["X-RateLimit-Reset"] = str(int(current_time + self.window_seconds))
        
        return response
    
    def _cleanup_old_entries(self, current_time: float) -> None:
        """
        Remove entries older than the time window.
        
        Args:
            current_time: Current timestamp
        """
        cutoff_time = current_time - self.window_seconds
        
        for client_ip in list(self.clients.keys()):
            self.clients[client_ip] = [
                t for t in self.clients[client_ip]
                if t > cutoff_time
            ]
            
            # Remove empty entries
            if not self.clients[client_ip]:
                del self.clients[client_ip]

class ResponseTimeMiddleware(BaseHTTPMiddleware):
    """
    Middleware for tracking response times and performance metrics.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Track response time for the request.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware in chain
            
        Returns:
            Response: HTTP response
        """
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Store in request state for later use
        request.state.response_time = response_time
        
        # Add to response headers
        response.headers["X-Response-Time"] = f"{response_time:.3f}s"
        
        # Log slow requests
        if response_time > 1.0:  # 1 second threshold
            logger.warning(
                f"Slow request detected - Path: {request.url.path}, "
                f"Time: {response_time:.3f}s"
            )
        
        return response

def setup_middleware(app, config: Dict[str, Any] = None) -> None:
    """
    Set up all middleware for the FastAPI application.
    
    This function adds all custom middleware to the app in the correct order.
    
    Args:
        app: FastAPI application
        config: Middleware configuration
    """
    config = config or {}
    
    # Order matters - middlewares are executed in reverse order of addition
    # Last added middleware executes first
    
    # Add response time tracking (executes early in the chain)
    app.add_middleware(ResponseTimeMiddleware)
    
    # Add rate limiting (executes after response time)
    if config.get('rate_limit', {}).get('enabled', True):
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=config['rate_limit'].get('requests_per_minute', 60)
        )
    
    # Add authentication (executes after rate limiting)
    if config.get('auth', {}).get('enabled', True):
        app.add_middleware(
            AuthenticationMiddleware,
            auth_service=config.get('auth_service')
        )
    
    # Add request logging (executes last - first in chain)
    app.add_middleware(RequestLoggingMiddleware)
    
    logger.info("Middleware setup complete")