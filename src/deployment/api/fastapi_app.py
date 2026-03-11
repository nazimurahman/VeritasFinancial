"""
FastAPI Application Factory
===========================

Creates and configures the FastAPI application for serving fraud detection models.
Includes CORS, authentication, rate limiting, and comprehensive error handling.
"""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

import time
import logging
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

# Configure logging
logger = logging.getLogger(__name__)

def create_app(config: Dict[str, Any] = None) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    This factory function creates a new FastAPI instance with all necessary
    middleware, routes, and error handlers configured based on the provided
    configuration.
    
    Args:
        config: Configuration dictionary containing:
            - title: API title (default: "VeritasFinancial Fraud Detection API")
            - version: API version (default: "1.0.0")
            - description: API description
            - cors_origins: List of allowed CORS origins
            - rate_limit: Rate limiting configuration
            - auth: Authentication configuration
            
    Returns:
        FastAPI: Configured FastAPI application
        
    Example:
        >>> config = {
        ...     'title': 'Fraud Detection API',
        ...     'cors_origins': ['https://app.veritasfinancial.com'],
        ...     'rate_limit': {'enabled': True, 'requests_per_minute': 100}
        ... }
        >>> app = create_app(config)
    """
    
    # Set default configuration
    config = config or {}
    app_title = config.get('title', 'VeritasFinancial Fraud Detection API')
    app_version = config.get('version', '1.0.0')
    app_description = config.get('description', 
        'Enterprise-grade API for real-time banking fraud detection. '
        'Provides risk scores, fraud predictions, and explainability features.'
    )
    
    # Define lifespan context manager for startup/shutdown events
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """
        Handle startup and shutdown events.
        
        This context manager runs when the application starts and stops,
        allowing for proper initialization and cleanup of resources.
        """
        # Startup: Initialize resources
        logger.info("Starting VeritasFinancial Fraud Detection API")
        
        # Load models and initialize connections
        await initialize_resources(app)
        
        yield  # Application runs here
        
        # Shutdown: Clean up resources
        logger.info("Shutting down VeritasFinancial Fraud Detection API")
        await cleanup_resources(app)
    
    # Create FastAPI instance with lifespan
    app = FastAPI(
        title=app_title,
        version=app_version,
        description=app_description,
        lifespan=lifespan,
        docs_url="/api/docs",  # Swagger UI
        redoc_url="/api/redoc",  # ReDoc documentation
        openapi_url="/api/openapi.json",  # OpenAPI schema
        contact={
            "name": "VeritasFinancial ML Team",
            "email": "ml-team@veritasfinancial.com",
            "url": "https://veritasfinancial.com/ai",
        },
        license_info={
            "name": "Proprietary - VeritasFinancial",
            "url": "https://veritasfinancial.com/license",
        }
    )
    
    # Configure middleware
    configure_middleware(app, config)
    
    # Configure error handlers
    configure_error_handlers(app)
    
    # Import and include routers
    from .endpoints import router
    app.include_router(router, prefix="/api/v1")
    
    # Add health check endpoint
    @app.get("/health", tags=["System"])
    async def health_check():
        """
        Health check endpoint for monitoring and load balancers.
        
        Returns:
            dict: Health status and component states
        """
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "version": app_version,
            "components": {
                "api": "ok",
                "model": await check_model_health(),
                "database": await check_database_health(),
                "cache": await check_cache_health()
            }
        }
    
    # Add readiness probe
    @app.get("/ready", tags=["System"])
    async def readiness_check():
        """
        Readiness probe for Kubernetes.
        
        Indicates whether the application is ready to serve traffic.
        """
        return {"status": "ready"}
    
    logger.info(f"FastAPI application created with title: {app_title}")
    return app

def configure_middleware(app: FastAPI, config: Dict[str, Any]) -> None:
    """
    Configure all middleware for the FastAPI application.
    
    Args:
        app: FastAPI application instance
        config: Application configuration
    """
    
    # CORS middleware - Allow cross-origin requests from specified origins
    cors_origins = config.get('cors_origins', [])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins if cors_origins else ["*"],  # In production, specify exact origins
        allow_credentials=True,
        allow_methods=["*"],  # Allow all HTTP methods
        allow_headers=["*"],  # Allow all headers
        expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining"]
    )
    
    # Trusted host middleware - Prevent host header attacks
    allowed_hosts = config.get('allowed_hosts', [
        "localhost",
        "127.0.0.1",
        "api.veritasfinancial.com",
        "*.veritasfinancial.com"
    ])
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=allowed_hosts
    )
    
    # Request ID middleware - Add unique ID to each request for tracing
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        """
        Add a unique request ID to each request for distributed tracing.
        
        This middleware generates or extracts a request ID and adds it to
        the request state and response headers for tracking across services.
        """
        import uuid
        
        # Get or generate request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        
        # Process the request
        start_time = time.time()
        response = await call_next(request)
        
        # Add processing time and request ID to response
        process_time = time.time() - start_time
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log request details
        logger.info(
            f"Request {request_id} - "
            f"{request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.3f}s"
        )
        
        return response
    
    # Rate limiting middleware - Prevent API abuse
    if config.get('rate_limit', {}).get('enabled', True):
        from fastapi import HTTPException
        
        # Simple in-memory rate limiter (use Redis in production)
        rate_limit_cache = {}
        
        @app.middleware("http")
        async def rate_limiter(request: Request, call_next):
            """
            Rate limiting middleware to prevent API abuse.
            
            Limits the number of requests per client IP address within a
            specified time window.
            """
            # Skip rate limiting for health checks
            if request.url.path in ["/health", "/ready"]:
                return await call_next(request)
            
            # Get client IP
            client_ip = request.client.host
            current_time = time.time()
            
            # Rate limit configuration
            limit = config['rate_limit'].get('requests_per_minute', 60)
            window = 60  # 1 minute in seconds
            
            # Clean up old entries
            rate_limit_cache[client_ip] = [
                t for t in rate_limit_cache.get(client_ip, [])
                if current_time - t < window
            ]
            
            # Check rate limit
            if len(rate_limit_cache.get(client_ip, [])) >= limit:
                logger.warning(f"Rate limit exceeded for IP: {client_ip}")
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Too Many Requests",
                        "message": f"Rate limit of {limit} requests per minute exceeded"
                    }
                )
            
            # Add timestamp for this request
            if client_ip not in rate_limit_cache:
                rate_limit_cache[client_ip] = []
            rate_limit_cache[client_ip].append(current_time)
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers
            response.headers["X-RateLimit-Limit"] = str(limit)
            response.headers["X-RateLimit-Remaining"] = str(
                max(0, limit - len(rate_limit_cache[client_ip]))
            )
            response.headers["X-RateLimit-Reset"] = str(int(current_time + window))
            
            return response

def configure_error_handlers(app: FastAPI) -> None:
    """
    Configure custom error handlers for the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """
        Handle HTTP exceptions with custom response format.
        
        Args:
            request: The request that caused the exception
            exc: The HTTP exception
            
        Returns:
            JSONResponse: Formatted error response
        """
        logger.error(f"HTTP exception: {exc.detail} - Path: {request.url.path}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.status_code,
                    "message": exc.detail,
                    "path": request.url.path,
                    "timestamp": time.time()
                }
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """
        Handle request validation errors with detailed feedback.
        
        Args:
            request: The request that caused the validation error
            exc: The validation exception
            
        Returns:
            JSONResponse: Detailed validation error response
        """
        logger.error(f"Validation error: {exc.errors()} - Path: {request.url.path}")
        
        # Format validation errors for better readability
        errors = []
        for error in exc.errors():
            errors.append({
                "field": " -> ".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"]
            })
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": 422,
                    "message": "Request validation failed",
                    "path": request.url.path,
                    "timestamp": time.time(),
                    "details": errors
                }
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """
        Handle all unhandled exceptions gracefully.
        
        Args:
            request: The request that caused the exception
            exc: The unhandled exception
            
        Returns:
            JSONResponse: Generic error response
        """
        logger.error(f"Unhandled exception: {str(exc)} - Path: {request.url.path}", exc_info=True)
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": 500,
                    "message": "Internal server error",
                    "path": request.url.path,
                    "timestamp": time.time()
                }
            }
        )

async def initialize_resources(app: FastAPI) -> None:
    """
    Initialize application resources on startup.
    
    This function loads models, establishes database connections,
    and initializes caches.
    
    Args:
        app: FastAPI application instance
    """
    logger.info("Initializing application resources...")
    
    # Store resources in app state for access throughout the application
    app.state.resources = {}
    
    try:
        # Load fraud detection models
        from ...models.classical_ml.xgboost_model import FraudXGBoostModel
        from ...models.deep_learning.neural_networks import FraudNeuralNetwork
        
        # Load XGBoost model
        app.state.resources['xgboost_model'] = FraudXGBoostModel.load(
            'artifacts/models/xgboost_fraud_model.pkl'
        )
        logger.info("XGBoost model loaded successfully")
        
        # Load Neural Network model (if GPU available)
        import torch
        if torch.cuda.is_available():
            app.state.resources['nn_model'] = FraudNeuralNetwork.load(
                'artifacts/models/neural_network.pt',
                device='cuda'
            )
            logger.info("Neural network model loaded on GPU")
        else:
            logger.info("GPU not available, skipping neural network model")
        
        # Initialize feature store connection
        from ...deployment.pipeline.feature_store import FeatureStore
        app.state.resources['feature_store'] = FeatureStore(
            config_path='configs/feature_config.yaml'
        )
        logger.info("Feature store initialized")
        
        # Initialize Redis cache
        import redis.asyncio as redis
        app.state.resources['cache'] = await redis.from_url(
            'redis://localhost:6379',
            decode_responses=True
        )
        logger.info("Redis cache initialized")
        
        # Initialize database connection pool
        import asyncpg
        app.state.resources['db_pool'] = await asyncpg.create_pool(
            'postgresql://user:pass@localhost/banking',
            min_size=5,
            max_size=20
        )
        logger.info("Database connection pool initialized")
        
        logger.info("All resources initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize resources: {str(e)}", exc_info=True)
        raise

async def cleanup_resources(app: FastAPI) -> None:
    """
    Clean up application resources on shutdown.
    
    Args:
        app: FastAPI application instance
    """
    logger.info("Cleaning up application resources...")
    
    try:
        # Close database connections
        if 'db_pool' in app.state.resources:
            await app.state.resources['db_pool'].close()
            logger.info("Database connections closed")
        
        # Close Redis connection
        if 'cache' in app.state.resources:
            await app.state.resources['cache'].close()
            logger.info("Redis connection closed")
        
        # Save any pending state
        if 'feature_store' in app.state.resources:
            await app.state.resources['feature_store'].flush()
            logger.info("Feature store flushed")
        
        logger.info("All resources cleaned up successfully")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}", exc_info=True)

async def check_model_health() -> str:
    """
    Check if models are loaded and healthy.
    
    Returns:
        str: Health status ('ok', 'degraded', or 'down')
    """
    # This would check if models are loaded and responsive
    return "ok"

async def check_database_health() -> str:
    """
    Check database connectivity.
    
    Returns:
        str: Health status
    """
    # This would perform a simple database query to check connectivity
    return "ok"

async def check_cache_health() -> str:
    """
    Check cache connectivity.
    
    Returns:
        str: Health status
    """
    # This would ping Redis to check connectivity
    return "ok"