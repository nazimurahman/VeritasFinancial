"""
VeritasFinancial Utilities Package
==================================
This package provides core utility functions and classes for the banking fraud detection system.
It includes logging, configuration management, data serialization, parallel processing,
and security features essential for production-grade fraud detection.

Author: VeritasFinancial Team
Version: 1.0.0
"""

# Import all utility modules to make them available at package level
from .logger import (
    setup_logger,                 # Initialize logging system
    FraudLogger,                   # Main logger class for fraud events
    log_execution_time,            # Decorator for timing function execution
    LogRotationHandler,            # Custom log rotation handler
    StructuredLogger               # JSON-structured logger for machine parsing
)

from .config_manager import (
    ConfigManager,                 # Central configuration management
    load_config,                   # Load configuration from files
    validate_config,               # Validate configuration structure
    ConfigError,                    # Custom configuration error
    EnvironmentManager              # Environment variable manager
)

from .data_serializers import (
    DataSerializer,                # Main serialization interface
    ParquetSerializer,             # Parquet format serializer
    JSONSerializer,                 # JSON format serializer
    PickleSerializer,               # Pickle format serializer
    SerializationError,             # Custom serialization error
    CompressionHandler,             # Handle data compression
    SchemaValidator                  # Validate data schemas
)

from .parallel_processing import (
    ParallelProcessor,              # Main parallel processing interface
    ThreadPoolManager,               # Thread-based parallel execution
    ProcessPoolManager,              # Process-based parallel execution
    DistributedProcessor,            # Distributed computing (Ray/Dask)
    BatchProcessor,                  # Batch data processing
    PipelineExecutor,                 # Execute processing pipelines
    ParallelError                     # Custom parallel processing error
)

from .security import (
    SecurityManager,                 # Main security interface
    DataEncryptor,                    # Data encryption/decryption
    TokenManager,                     # JWT token management
    AuditLogger,                      # Security audit logging
    AccessControl,                    # Role-based access control
    DataMasker,                        # PII data masking
    SecureHasher,                      # Secure hashing utilities
    SecurityError                       # Custom security error
)

# Package metadata
__version__ = '1.0.0'
__author__ = 'VeritasFinancial Team'
__all__ = [
    # Logger exports
    'setup_logger',
    'FraudLogger',
    'log_execution_time',
    'LogRotationHandler',
    'StructuredLogger',
    
    # Config manager exports
    'ConfigManager',
    'load_config',
    'validate_config',
    'ConfigError',
    'EnvironmentManager',
    
    # Serializer exports
    'DataSerializer',
    'ParquetSerializer',
    'JSONSerializer',
    'PickleSerializer',
    'SerializationError',
    'CompressionHandler',
    'SchemaValidator',
    
    # Parallel processing exports
    'ParallelProcessor',
    'ThreadPoolManager',
    'ProcessPoolManager',
    'DistributedProcessor',
    'BatchProcessor',
    'PipelineExecutor',
    'ParallelError',
    
    # Security exports
    'SecurityManager',
    'DataEncryptor',
    'TokenManager',
    'AuditLogger',
    'AccessControl',
    'DataMasker',
    'SecureHasher',
    'SecurityError'
]

# Package initialization
def initialize_utilities(config_path: str = None) -> dict:
    """
    Initialize all utility modules with common configuration.
    
    This function sets up all utility components with a shared configuration,
    ensuring consistent behavior across the entire system.
    
    Args:
        config_path (str, optional): Path to configuration file
        
    Returns:
        dict: Dictionary containing initialized utility instances
        
    Example:
        >>> utils = initialize_utilities('configs/utils_config.yaml')
        >>> logger = utils['logger']
        >>> config = utils['config_manager']
    """
    # Load configuration
    config_manager = ConfigManager(config_path)
    utils_config = config_manager.get_config('utils', {})
    
    # Initialize logger
    logger = setup_logger(
        name='veritas_financial',
        log_level=utils_config.get('log_level', 'INFO'),
        log_dir=utils_config.get('log_dir', 'logs')
    )
    
    # Initialize security manager
    security_manager = SecurityManager(
        encryption_key=utils_config.get('encryption_key'),
        token_secret=utils_config.get('token_secret')
    )
    
    # Initialize parallel processor
    parallel_processor = ParallelProcessor(
        max_workers=utils_config.get('max_workers', 4),
        use_gpu=utils_config.get('use_gpu', False)
    )
    
    logger.info("All utility modules initialized successfully")
    
    return {
        'config_manager': config_manager,
        'logger': logger,
        'security_manager': security_manager,
        'parallel_processor': parallel_processor,
        'version': __version__
    }