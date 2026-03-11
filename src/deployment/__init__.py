"""
VeritasFinancial Deployment Package
====================================

This package contains all production deployment components for the banking fraud detection system.
It provides APIs, monitoring, and data pipeline infrastructure for real-time and batch processing.

Author: VeritasFinancial ML Team
Version: 1.0.0
"""

# Package metadata
__version__ = "1.0.0"
__author__ = "VeritasFinancial ML Team"
__license__ = "Proprietary - VeritasFinancial Internal Use Only"

# Import key components for easy access at package level
from .api.fastapi_app import create_app  # Factory function to create FastAPI application
from .api.endpoints import router  # Main API router with all endpoints
from .api.middleware import setup_middleware  # Middleware configuration function

from .monitoring.drift_detection import (
    DataDriftDetector,  # Detect data drift in production
    ConceptDriftDetector,  # Detect concept drift in model performance
    DriftAlert  # Alert management for drift events
)

from .monitoring.performance_tracking import (
    PerformanceTracker,  # Track model performance metrics
    ModelMonitor,  # Monitor model health and performance
    MetricsCollector  # Collect and store performance metrics
)

from .monitoring.alerting import (
    AlertManager,  # Manage and route alerts
    AlertChannel,  # Alert delivery channels (Slack, Email, etc.)
    AlertRule  # Define alerting rules and thresholds
)

from .pipeline.batch_processing import (
    BatchProcessor,  # Process large volumes of data in batches
    BatchJob,  # Batch job definition and management
    BatchResult  # Batch processing results
)

from .pipeline.realtime_processing import (
    StreamProcessor,  # Real-time stream processing
    TransactionProcessor,  # Process individual transactions in real-time
    FeatureCache  # Cache for real-time feature computation
)

from .pipeline.feature_store import (
    FeatureStore,  # Central feature storage and retrieval
    FeatureGroup,  # Logical grouping of features
    FeatureView,  # Materialized view of features for serving
    OnlineStore,  # Online feature storage (Redis/Memcached)
    OfflineStore  # Offline feature storage (S3/Parquet)
)

# Define what gets imported with "from deployment import *"
__all__ = [
    # API components
    'create_app',
    'router',
    'setup_middleware',
    
    # Monitoring components
    'DataDriftDetector',
    'ConceptDriftDetector',
    'DriftAlert',
    'PerformanceTracker',
    'ModelMonitor',
    'MetricsCollector',
    'AlertManager',
    'AlertChannel',
    'AlertRule',
    
    # Pipeline components
    'BatchProcessor',
    'BatchJob',
    'BatchResult',
    'StreamProcessor',
    'TransactionProcessor',
    'FeatureCache',
    'FeatureStore',
    'FeatureGroup',
    'FeatureView',
    'OnlineStore',
    'OfflineStore'
]

# Package initialization code (runs when package is imported)
import logging
from pathlib import Path

# Configure package-level logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Prevent "No handlers found" warnings

def initialize_deployment(config_path: str = None):
    """
    Initialize the deployment package with configuration.
    
    This function should be called when starting the application to set up
    all deployment components with the provided configuration.
    
    Args:
        config_path: Path to configuration file (YAML or JSON)
        
    Returns:
        dict: Initialized components dictionary
        
    Example:
        >>> from deployment import initialize_deployment
        >>> components = initialize_deployment('config/deployment.yaml')
        >>> api_app = components['app']
        >>> monitor = components['monitor']
    """
    logger.info("Initializing VeritasFinancial deployment package")
    
    # Load configuration
    config = _load_config(config_path) if config_path else {}
    
    # Initialize components based on config
    components = {}
    
    # Initialize feature store first (core dependency)
    if 'feature_store' in config:
        from .pipeline.feature_store import FeatureStore
        components['feature_store'] = FeatureStore(config['feature_store'])
        logger.info("Feature store initialized")
    
    # Initialize monitoring components
    if 'monitoring' in config:
        from .monitoring.performance_tracking import PerformanceTracker
        from .monitoring.drift_detection import DataDriftDetector
        components['performance_tracker'] = PerformanceTracker(config['monitoring'])
        components['drift_detector'] = DataDriftDetector(config['monitoring'])
        logger.info("Monitoring components initialized")
    
    # Initialize API app
    if 'api' in config:
        from .api.fastapi_app import create_app
        components['app'] = create_app(config['api'])
        logger.info("API application initialized")
    
    # Initialize processing pipelines
    if 'pipeline' in config:
        from .pipeline.realtime_processing import StreamProcessor
        from .pipeline.batch_processing import BatchProcessor
        components['stream_processor'] = StreamProcessor(config['pipeline'])
        components['batch_processor'] = BatchProcessor(config['pipeline'])
        logger.info("Processing pipelines initialized")
    
    logger.info("Deployment package initialization complete")
    return components

def _load_config(config_path: str) -> dict:
    """
    Load configuration from file.
    
    Supports YAML and JSON formats.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        dict: Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If file format is not supported
    """
    import yaml
    import json
    
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load based on file extension
    if config_file.suffix in ['.yaml', '.yml']:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    elif config_file.suffix == '.json':
        with open(config_file, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_file.suffix}")

# Clean up function for graceful shutdown
def shutdown_deployment():
    """
    Gracefully shut down all deployment components.
    
    This function should be called when stopping the application to ensure
    all resources are properly released.
    """
    logger.info("Shutting down VeritasFinancial deployment components")
    # Perform cleanup operations
    # Close connections, flush buffers, etc.
    logger.info("Shutdown complete")