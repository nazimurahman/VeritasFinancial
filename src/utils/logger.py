"""
Logging Utilities for VeritasFinancial
======================================
Provides comprehensive logging capabilities for fraud detection including:
- Structured logging for machine parsing
- Fraud-specific event logging
- Performance tracking
- Log rotation and management
- Correlation ID tracking for transaction tracing

Author: VeritasFinancial Team
"""

import logging
import logging.handlers
import json
import time
import functools
import traceback
from datetime import datetime
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import uuid
import threading
import socket
import os

# Thread-local storage for correlation IDs
_thread_local = threading.local()

class StructuredLogger:
    """
    JSON-structured logger for machine-parsable logs.
    
    This logger formats log entries as JSON objects, making them easy to parse
    by log aggregation tools like ELK Stack, Splunk, or Datadog.
    
    Attributes:
        logger (logging.Logger): Underlying Python logger
        service_name (str): Name of the service generating logs
        environment (str): Deployment environment (dev/staging/prod)
    """
    
    def __init__(self, name: str, service_name: str = 'veritas_financial', 
                 environment: str = 'development'):
        """
        Initialize structured logger.
        
        Args:
            name (str): Logger name
            service_name (str): Service identifier
            environment (str): Deployment environment
        """
        self.logger = logging.getLogger(name)
        self.service_name = service_name
        self.environment = environment
        self.hostname = socket.gethostname()
        
    def _get_correlation_id(self) -> str:
        """
        Get the current correlation ID from thread-local storage.
        
        Correlation IDs allow tracing a single transaction across multiple services.
        
        Returns:
            str: Correlation ID or 'N/A' if not set
        """
        return getattr(_thread_local, 'correlation_id', 'N/A')
    
    def _get_base_log_entry(self, level: str) -> Dict[str, Any]:
        """
        Create base log entry with common fields.
        
        Args:
            level (str): Log level
            
        Returns:
            Dict: Base log entry structure
        """
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level,
            'service': self.service_name,
            'environment': self.environment,
            'hostname': self.hostname,
            'correlation_id': self._get_correlation_id(),
            'thread_id': threading.get_ident(),
            'process_id': os.getpid()
        }
    
    def info(self, message: str, **kwargs):
        """
        Log info level message with structured data.
        
        Args:
            message (str): Log message
            **kwargs: Additional structured data to log
        """
        entry = self._get_base_log_entry('INFO')
        entry['message'] = message
        entry.update(kwargs)
        self.logger.info(json.dumps(entry))
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """
        Log error level message with exception details.
        
        Args:
            message (str): Error message
            exception (Exception, optional): Exception object
            **kwargs: Additional structured data
        """
        entry = self._get_base_log_entry('ERROR')
        entry['message'] = message
        if exception:
            entry['exception_type'] = type(exception).__name__
            entry['exception_message'] = str(exception)
            entry['traceback'] = traceback.format_exc()
        entry.update(kwargs)
        self.logger.error(json.dumps(entry))
    
    def warning(self, message: str, **kwargs):
        """Log warning level message."""
        entry = self._get_base_log_entry('WARNING')
        entry['message'] = message
        entry.update(kwargs)
        self.logger.warning(json.dumps(entry))
    
    def debug(self, message: str, **kwargs):
        """Log debug level message."""
        entry = self._get_base_log_entry('DEBUG')
        entry['message'] = message
        entry.update(kwargs)
        self.logger.debug(json.dumps(entry))
    
    def fraud_alert(self, transaction_id: str, risk_score: float, 
                    fraud_probability: float, **kwargs):
        """
        Special logging method for fraud alerts.
        
        Args:
            transaction_id (str): ID of the transaction
            risk_score (float): Calculated risk score
            fraud_probability (float): Probability of fraud
            **kwargs: Additional fraud-related data
        """
        entry = self._get_base_log_entry('FRAUD_ALERT')
        entry.update({
            'transaction_id': transaction_id,
            'risk_score': risk_score,
            'fraud_probability': fraud_probability,
            'alert_type': 'potential_fraud',
            'requires_review': risk_score > 0.7 or fraud_probability > 0.5
        })
        entry.update(kwargs)
        self.logger.warning(json.dumps(entry))


class FraudLogger:
    """
    Specialized logger for fraud detection events.
    
    This logger captures fraud-specific metrics and events for monitoring
    and analysis.
    """
    
    def __init__(self, logger: StructuredLogger):
        """
        Initialize fraud logger.
        
        Args:
            logger (StructuredLogger): Base structured logger
        """
        self.logger = logger
        self.fraud_metrics = {
            'total_transactions': 0,
            'fraud_detected': 0,
            'false_positives': 0,
            'avg_risk_score': 0.0
        }
        
    def log_transaction_analysis(self, transaction_id: str, features: Dict,
                                  prediction: float, actual: Optional[int] = None):
        """
        Log detailed transaction analysis.
        
        Args:
            transaction_id (str): Transaction identifier
            features (Dict): Features used for prediction
            prediction (float): Model prediction
            actual (int, optional): Actual label if known
        """
        self.fraud_metrics['total_transactions'] += 1
        
        log_entry = {
            'transaction_id': transaction_id,
            'feature_count': len(features),
            'prediction_score': prediction,
            'top_features': self._get_top_features(features)
        }
        
        if actual is not None:
            log_entry['actual_label'] = actual
            if prediction >= 0.5 and actual == 0:
                self.fraud_metrics['false_positives'] += 1
            elif prediction >= 0.5 and actual == 1:
                self.fraud_metrics['fraud_detected'] += 1
        
        self.logger.info("Transaction analyzed", **log_entry)
    
    def _get_top_features(self, features: Dict, top_n: int = 5) -> Dict:
        """
        Get top N features by absolute value.
        
        Args:
            features (Dict): Feature dictionary
            top_n (int): Number of top features to return
            
        Returns:
            Dict: Top features with their values
        """
        sorted_features = sorted(features.items(), 
                                key=lambda x: abs(x[1]), 
                                reverse=True)[:top_n]
        return dict(sorted_features)
    
    def log_model_performance(self, model_name: str, metrics: Dict):
        """
        Log model performance metrics.
        
        Args:
            model_name (str): Name of the model
            metrics (Dict): Performance metrics
        """
        self.logger.info("Model performance metrics", 
                        model_name=model_name, 
                        metrics=metrics)
    
    def get_metrics_summary(self) -> Dict:
        """
        Get summary of fraud metrics.
        
        Returns:
            Dict: Summary statistics
        """
        if self.fraud_metrics['total_transactions'] > 0:
            fraud_rate = (self.fraud_metrics['fraud_detected'] / 
                         self.fraud_metrics['total_transactions'])
        else:
            fraud_rate = 0.0
            
        return {
            **self.fraud_metrics,
            'fraud_rate': fraud_rate,
            'precision': self._calculate_precision()
        }
    
    def _calculate_precision(self) -> float:
        """
        Calculate precision of fraud detection.
        
        Returns:
            float: Precision score
        """
        total_fraud_predictions = (self.fraud_metrics['fraud_detected'] + 
                                   self.fraud_metrics['false_positives'])
        if total_fraud_predictions == 0:
            return 0.0
        return self.fraud_metrics['fraud_detected'] / total_fraud_predictions


def setup_logger(name: str = 'veritas_financial', 
                 log_level: str = 'INFO',
                 log_dir: str = 'logs',
                 max_bytes: int = 10485760,  # 10MB
                 backup_count: int = 5,
                 structured: bool = True) -> StructuredLogger:
    """
    Set up and configure logger.
    
    This function creates a comprehensive logging setup with:
    - File rotation
    - Structured JSON logging (optional)
    - Console output for development
    - Error-specific log file
    
    Args:
        name (str): Logger name
        log_level (str): Logging level
        log_dir (str): Directory for log files
        max_bytes (int): Maximum size per log file
        backup_count (int): Number of backup files to keep
        structured (bool): Whether to use structured JSON logging
        
    Returns:
        StructuredLogger: Configured structured logger
        
    Example:
        >>> logger = setup_logger('fraud_detection', log_level='INFO')
        >>> logger.info("System started", version="1.0.0")
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    if structured:
        # For structured logging, we'll use our StructuredLogger wrapper
        # The actual file handler will just store the JSON strings
        formatter = logging.Formatter('%(message)s')
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # File handler with rotation for all logs
    all_log_file = log_path / f'{name}.log'
    file_handler = logging.handlers.RotatingFileHandler(
        all_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Separate error file handler
    error_log_file = log_path / f'{name}_error.log'
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    
    # Console handler for development
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Wrap in structured logger if requested
    if structured:
        return StructuredLogger(name, service_name=name)
    else:
        # Return a wrapper that mimics StructuredLogger interface
        return _LegacyLoggerWrapper(logger)


def log_execution_time(logger: Optional[StructuredLogger] = None):
    """
    Decorator to log function execution time.
    
    This decorator measures and logs how long a function takes to execute,
    which is crucial for performance monitoring in production.
    
    Args:
        logger (StructuredLogger, optional): Logger instance
        
    Returns:
        Callable: Decorated function
        
    Example:
        >>> @log_execution_time()
        >>> def process_transactions(data):
        >>>     # Process data
        >>>     return result
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger - either provided or try to get from instance
            nonlocal logger
            if logger is None:
                # Try to get logger from self if it's a method
                if args and hasattr(args[0], 'logger'):
                    logger = args[0].logger
                else:
                    # Create a temporary logger
                    logger = setup_logger('execution_timer')
            
            # Start timing
            start_time = time.time()
            start_cpu = time.process_time()
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Calculate execution time
                end_time = time.time()
                end_cpu = time.process_time()
                
                wall_time = end_time - start_time
                cpu_time = end_cpu - start_cpu
                
                # Log performance metrics
                logger.info(f"Function {func.__name__} executed",
                          function=func.__name__,
                          wall_time_seconds=round(wall_time, 4),
                          cpu_time_seconds=round(cpu_time, 4),
                          arguments_count=len(args) + len(kwargs))
                
                return result
                
            except Exception as e:
                # Log error with timing
                end_time = time.time()
                execution_time = end_time - start_time
                
                logger.error(f"Function {func.__name__} failed",
                           exception=e,
                           execution_time_seconds=round(execution_time, 4))
                raise
                
        return wrapper
    return decorator


class LogRotationHandler(logging.handlers.RotatingFileHandler):
    """
    Enhanced log rotation handler with compression support.
    
    This handler extends the standard RotatingFileHandler to support
    compression of rotated log files to save disk space.
    """
    
    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0,
                 encoding=None, delay=False, compress=True):
        """
        Initialize enhanced rotation handler.
        
        Args:
            filename (str): Log file path
            mode (str): File open mode
            maxBytes (int): Maximum file size before rotation
            backupCount (int): Number of backup files to keep
            encoding (str): File encoding
            delay (bool): Delay file creation
            compress (bool): Whether to compress rotated files
        """
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)
        self.compress = compress
        
    def doRollover(self):
        """
        Perform log rotation with optional compression.
        """
        if self.stream:
            self.stream.close()
            self.stream = None
            
        if self.backupCount > 0:
            for i in range(self.backupCount - 1, 0, -1):
                sfn = self.rotation_filename("%s.%d" % (self.baseFilename, i))
                dfn = self.rotation_filename("%s.%d" % (self.baseFilename, i + 1))
                if os.path.exists(sfn):
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    os.rename(sfn, dfn)
                    
            dfn = self.rotation_filename(self.baseFilename + ".1")
            if os.path.exists(dfn):
                os.remove(dfn)
                
            self.rotate(self.baseFilename, dfn)
            
            # Compress if enabled
            if self.compress and os.path.exists(dfn):
                import gzip
                with open(dfn, 'rb') as f_in:
                    with gzip.open(f'{dfn}.gz', 'wb') as f_out:
                        f_out.writelines(f_in)
                os.remove(dfn)
                
        if not self.delay:
            self.stream = self._open()


class _LegacyLoggerWrapper:
    """
    Wrapper for legacy loggers to maintain consistent interface.
    
    This internal class wraps standard logging.Logger to provide the same
    interface as StructuredLogger for backward compatibility.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
    def info(self, message: str, **kwargs):
        if kwargs:
            self.logger.info(f"{message} - {kwargs}")
        else:
            self.logger.info(message)
            
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        if exception:
            self.logger.error(f"{message} - {exception} - {kwargs}")
        else:
            self.logger.error(f"{message} - {kwargs}")
            
    def warning(self, message: str, **kwargs):
        if kwargs:
            self.logger.warning(f"{message} - {kwargs}")
        else:
            self.logger.warning(message)
            
    def debug(self, message: str, **kwargs):
        if kwargs:
            self.logger.debug(f"{message} - {kwargs}")
        else:
            self.logger.debug(message)
            
    def fraud_alert(self, transaction_id: str, risk_score: float, 
                    fraud_probability: float, **kwargs):
        self.logger.warning(
            f"FRAUD ALERT - Transaction: {transaction_id}, "
            f"Risk: {risk_score}, Probability: {fraud_probability}, "
            f"Details: {kwargs}"
        )


# Context manager for correlation ID
class correlation_context:
    """
    Context manager for setting correlation ID.
    
    This allows tracing a series of operations with the same correlation ID.
    
    Example:
        >>> with correlation_context('txn_123'):
        >>>     process_transaction()
        >>>     log_analysis()  # All logs will have correlation_id='txn_123'
    """
    
    def __init__(self, correlation_id: str):
        """
        Initialize correlation context.
        
        Args:
            correlation_id (str): Correlation ID to use
        """
        self.correlation_id = correlation_id
        self.old_id = None
        
    def __enter__(self):
        """Set correlation ID for this thread."""
        self.old_id = getattr(_thread_local, 'correlation_id', None)
        _thread_local.correlation_id = self.correlation_id
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore previous correlation ID."""
        if self.old_id is not None:
            _thread_local.correlation_id = self.old_id
        else:
            delattr(_thread_local, 'correlation_id')