# =============================================================================
# VERITASFINANCIAL - FRAUD DETECTION SYSTEM
# Module: deployment/pipeline/batch_processing.py
# Description: Batch processing pipeline for large-scale fraud analysis
# Author: Data Science Team
# Version: 2.0.0
# Last Updated: 2024-01-15
# =============================================================================

"""
Batch Processing Pipeline for Fraud Detection
===============================================
This module handles large-scale batch processing of transactions:
- Scheduled batch jobs for historical analysis
- Parallel processing with configurable workers
- Checkpointing and fault tolerance
- Data partitioning and distribution
- Integration with feature store
- Model retraining workflows
- Backtesting and validation
"""

import os
import sys
import json
import time
import uuid
import hashlib
import pickle
import logging
import threading
import multiprocessing as mp
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Callable, Iterator, Tuple
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path

# Third-party imports with error handling
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logging.warning("Pandas not available")

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.dataset as ds
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    logging.warning("PyArrow not available")

try:
    import boto3
    from botocore.exceptions import ClientError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    logging.warning("Boto3 not available")

try:
    import sqlalchemy as sa
    from sqlalchemy import create_engine, text
    SQL_AVAILABLE = True
except ImportError:
    SQL_AVAILABLE = False
    logging.warning("SQLAlchemy not available")

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/batch_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES FOR BATCH PROCESSING
# =============================================================================

@dataclass
class BatchJob:
    """
    Represents a batch processing job.
    
    Attributes:
        job_id: Unique job identifier
        job_type: Type of batch job (training, inference, backtest, etc.)
        status: Current job status
        config: Job configuration
        created_at: Job creation timestamp
        started_at: Job start timestamp
        completed_at: Job completion timestamp
        input_paths: List of input data paths
        output_path: Path for job results
        checkpoint_path: Path for job checkpoints
        error: Error information if failed
        metrics: Job performance metrics
        metadata: Additional metadata
    """
    job_id: str
    job_type: str  # 'training', 'inference', 'backtest', 'feature_engineering', 'evaluation'
    status: str  # 'pending', 'running', 'completed', 'failed', 'cancelled'
    config: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    input_paths: List[str] = field(default_factory=list)
    output_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['created_at'] = self.created_at.isoformat()
        if self.started_at:
            result['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            result['completed_at'] = self.completed_at.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchJob':
        """Create from dictionary."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('started_at'):
            data['started_at'] = datetime.fromisoformat(data['started_at'])
        if data.get('completed_at'):
            data['completed_at'] = datetime.fromisoformat(data['completed_at'])
        return cls(**data)


@dataclass
class BatchResult:
    """
    Result of a batch processing job.
    
    Attributes:
        job_id: Associated job ID
        success: Whether job succeeded
        output_data: Result data (if any)
        metrics: Performance metrics
        error: Error message (if failed)
        processing_time: Total processing time in seconds
        rows_processed: Number of rows processed
        output_paths: Paths to output files
    """
    job_id: str
    success: bool
    output_data: Optional[Any] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    processing_time: float = 0.0
    rows_processed: int = 0
    output_paths: List[str] = field(default_factory=list)


@dataclass
class DataPartition:
    """
    Represents a data partition for parallel processing.
    
    Attributes:
        partition_id: Partition identifier
        source: Data source (path, query, etc.)
        offset: Start offset
        limit: Number of records
        filters: Filter conditions
        schema: Expected schema
    """
    partition_id: str
    source: str
    offset: int
    limit: int
    filters: Optional[Dict[str, Any]] = None
    schema: Optional[Dict[str, str]] = None


# =============================================================================
# DATA READERS AND WRITERS
# =============================================================================

class DataReader(ABC):
    """
    Abstract base class for data readers.
    Defines interface for reading data from various sources.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize reader with configuration.
        
        Args:
            config: Reader configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def read(self, source: str, **kwargs) -> Any:
        """
        Read data from source.
        
        Args:
            source: Data source identifier
            **kwargs: Additional read parameters
            
        Returns:
            Read data
        """
        pass
    
    @abstractmethod
    def read_partition(self, partition: DataPartition) -> Any:
        """
        Read a specific partition.
        
        Args:
            partition: Data partition to read
            
        Returns:
            Partition data
        """
        pass
    
    @abstractmethod
    def get_partitions(self, source: str, partition_size: int) -> List[DataPartition]:
        """
        Split source into partitions.
        
        Args:
            source: Data source
            partition_size: Records per partition
            
        Returns:
            List of partitions
        """
        pass


class ParquetDataReader(DataReader):
    """
    Reads data from Parquet files.
    Optimized for columnar storage with predicate pushdown.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Parquet reader.
        
        Args:
            config: Configuration with file paths, schema, etc.
        """
        super().__init__(config)
        
        if not PYARROW_AVAILABLE:
            raise ImportError("PyArrow is required for ParquetDataReader")
        
        self.file_pattern = config.get('file_pattern', '*.parquet')
        self.use_threads = config.get('use_threads', True)
        self.batch_size = config.get('batch_size', 100000)
        
    def read(self, source: str, **kwargs) -> pd.DataFrame:
        """
        Read Parquet file(s) into DataFrame.
        
        Args:
            source: Path to Parquet file or directory
            **kwargs: Additional parameters (columns, filters, etc.)
            
        Returns:
            DataFrame with read data
        """
        try:
            # Handle directory or file
            source_path = Path(source)
            if source_path.is_dir():
                # Read all parquet files in directory
                files = list(source_path.glob(self.file_pattern))
                if not files:
                    self.logger.warning(f"No parquet files found in {source}")
                    return pd.DataFrame()
                
                # Use dataset for efficient reading
                dataset = ds.dataset(files, format='parquet')
                
                # Apply filters if provided
                filters = kwargs.get('filters')
                columns = kwargs.get('columns')
                
                # Read to table
                table = dataset.to_table(
                    columns=columns,
                    filter=filters,
                    batch_size=self.batch_size
                )
            else:
                # Read single file
                table = pq.read_table(
                    source,
                    columns=kwargs.get('columns'),
                    filters=kwargs.get('filters'),
                    use_threads=self.use_threads
                )
            
            # Convert to pandas
            df = table.to_pandas()
            
            self.logger.info(f"Read {len(df)} rows from {source}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to read Parquet from {source}: {e}")
            raise
    
    def read_partition(self, partition: DataPartition) -> pd.DataFrame:
        """
        Read a specific partition.
        
        Args:
            partition: Partition to read
            
        Returns:
            Partition data
        """
        # For Parquet, we need to read all and then slice
        # This is inefficient for large files - in production,
        # you'd use partitioned datasets or row group filtering
        df = self.read(partition.source)
        
        # Apply offset and limit
        if partition.offset > 0 or partition.limit < len(df):
            df = df.iloc[partition.offset:partition.offset + partition.limit]
        
        # Apply filters
        if partition.filters:
            for col, value in partition.filters.items():
                if isinstance(value, (list, tuple)):
                    df = df[df[col].isin(value)]
                else:
                    df = df[df[col] == value]
        
        return df
    
    def get_partitions(self, source: str, partition_size: int) -> List[DataPartition]:
        """
        Create partitions based on file structure.
        
        Args:
            source: Source path
            partition_size: Target partition size
            
        Returns:
            List of partitions
        """
        partitions = []
        
        source_path = Path(source)
        if source_path.is_dir():
            # Each file can be a partition
            files = sorted(list(source_path.glob(self.file_pattern)))
            for i, file_path in enumerate(files):
                partitions.append(DataPartition(
                    partition_id=f"file_{i}",
                    source=str(file_path),
                    offset=0,
                    limit=-1  # Read entire file
                ))
        else:
            # Single file - need row group information
            # For simplicity, we'll create partitions based on row count
            try:
                metadata = pq.read_metadata(source)
                total_rows = metadata.num_rows
                
                for i in range(0, total_rows, partition_size):
                    partitions.append(DataPartition(
                        partition_id=f"row_group_{i}",
                        source=source,
                        offset=i,
                        limit=min(partition_size, total_rows - i)
                    ))
            except Exception as e:
                self.logger.error(f"Failed to get partitions for {source}: {e}")
                partitions.append(DataPartition(
                    partition_id="full",
                    source=source,
                    offset=0,
                    limit=-1
                ))
        
        return partitions


class SQLDataReader(DataReader):
    """
    Reads data from SQL databases.
    Supports streaming and partitioned reads.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SQL reader.
        
        Args:
            config: Configuration with connection string, etc.
        """
        super().__init__(config)
        
        if not SQL_AVAILABLE:
            raise ImportError("SQLAlchemy is required for SQLDataReader")
        
        self.connection_string = config.get('connection_string')
        self.engine = create_engine(self.connection_string) if self.connection_string else None
        self.chunksize = config.get('chunksize', 10000)
        
    def read(self, source: str, **kwargs) -> pd.DataFrame:
        """
        Read from SQL table or query.
        
        Args:
            source: Table name or SQL query
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with results
        """
        try:
            # Determine if source is table or query
            if source.strip().upper().startswith('SELECT'):
                # It's a query
                query = source
                params = kwargs.get('params', {})
            else:
                # It's a table
                query = f"SELECT * FROM {source}"
                params = {}
            
            # Add filters
            filters = kwargs.get('filters')
            if filters and not source.strip().upper().startswith('SELECT'):
                where_clauses = []
                for col, value in filters.items():
                    if isinstance(value, (list, tuple)):
                        placeholders = ','.join(['?' for _ in value])
                        where_clauses.append(f"{col} IN ({placeholders})")
                        params.update({f"{col}_{i}": v for i, v in enumerate(value)})
                    else:
                        where_clauses.append(f"{col} = :{col}")
                        params[col] = value
                
                if where_clauses:
                    query += " WHERE " + " AND ".join(where_clauses)
            
            # Read in chunks
            chunks = []
            for chunk in pd.read_sql_query(
                query,
                self.engine,
                params=params,
                chunksize=self.chunksize
            ):
                chunks.append(chunk)
            
            if chunks:
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.DataFrame()
            
            self.logger.info(f"Read {len(df)} rows from SQL source: {source}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to read from SQL: {e}")
            raise
    
    def read_partition(self, partition: DataPartition) -> pd.DataFrame:
        """
        Read a specific partition using SQL pagination.
        
        Args:
            partition: Partition to read
            
        Returns:
            Partition data
        """
        try:
            # Build paginated query
            source = partition.source
            
            if source.strip().upper().startswith('SELECT'):
                # For queries, wrap with pagination
                base_query = source.rstrip(';')
                query = f"""
                    SELECT * FROM (
                        {base_query}
                    ) AS partitioned_query
                    LIMIT {partition.limit} OFFSET {partition.offset}
                """
            else:
                # For tables, use ORDER BY for consistent pagination
                primary_key = self.config.get('primary_key', 'id')
                query = f"""
                    SELECT * FROM {source}
                    ORDER BY {primary_key}
                    LIMIT {partition.limit} OFFSET {partition.offset}
                """
            
            return self.read(query)
            
        except Exception as e:
            self.logger.error(f"Failed to read partition {partition.partition_id}: {e}")
            raise
    
    def get_partitions(self, source: str, partition_size: int) -> List[DataPartition]:
        """
        Create SQL partitions based on row count.
        
        Args:
            source: Table name or query
            partition_size: Records per partition
            
        Returns:
            List of partitions
        """
        partitions = []
        
        try:
            # Get total row count
            if source.strip().upper().startswith('SELECT'):
                count_query = f"SELECT COUNT(*) FROM ({source}) AS count_query"
            else:
                count_query = f"SELECT COUNT(*) FROM {source}"
            
            result = pd.read_sql_query(count_query, self.engine)
            total_rows = result.iloc[0, 0]
            
            # Create partitions
            for i in range(0, total_rows, partition_size):
                partitions.append(DataPartition(
                    partition_id=f"sql_part_{i}",
                    source=source,
                    offset=i,
                    limit=min(partition_size, total_rows - i)
                ))
                
        except Exception as e:
            self.logger.error(f"Failed to get partitions for {source}: {e}")
            partitions.append(DataPartition(
                partition_id="full",
                source=source,
                offset=0,
                limit=-1
            ))
        
        return partitions


class DataWriter(ABC):
    """
    Abstract base class for data writers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize writer with configuration.
        
        Args:
            config: Writer configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def write(self, data: Any, destination: str, **kwargs) -> str:
        """
        Write data to destination.
        
        Args:
            data: Data to write
            destination: Output destination
            **kwargs: Additional write parameters
            
        Returns:
            Path/identifier of written data
        """
        pass


class ParquetDataWriter(DataWriter):
    """
    Writes data to Parquet format.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Parquet writer.
        
        Args:
            config: Configuration with compression, partitioning, etc.
        """
        super().__init__(config)
        
        if not PYARROW_AVAILABLE:
            raise ImportError("PyArrow is required for ParquetDataWriter")
        
        self.compression = config.get('compression', 'snappy')
        self.partition_cols = config.get('partition_cols', [])
        self.row_group_size = config.get('row_group_size', 100000)
    
    def write(self, data: Any, destination: str, **kwargs) -> str:
        """
        Write DataFrame to Parquet.
        
        Args:
            data: DataFrame to write
            destination: Output path
            **kwargs: Additional parameters
            
        Returns:
            Path to written file
        """
        try:
            if isinstance(data, pd.DataFrame):
                # Convert to Arrow Table
                table = pa.Table.from_pandas(data)
                
                # Write to Parquet
                if self.partition_cols:
                    # Partitioned write
                    pq.write_to_dataset(
                        table,
                        root_path=destination,
                        partition_cols=self.partition_cols,
                        compression=self.compression
                    )
                else:
                    # Single file write
                    output_path = Path(destination)
                    if output_path.suffix != '.parquet':
                        output_path = output_path.with_suffix('.parquet')
                    
                    pq.write_table(
                        table,
                        output_path,
                        compression=self.compression,
                        row_group_size=self.row_group_size
                    )
                
                self.logger.info(f"Wrote {len(data)} rows to {destination}")
                return str(destination)
            
            else:
                self.logger.error(f"Unsupported data type: {type(data)}")
                raise ValueError(f"Unsupported data type: {type(data)}")
                
        except Exception as e:
            self.logger.error(f"Failed to write Parquet to {destination}: {e}")
            raise


# =============================================================================
# PROCESSOR IMPLEMENTATIONS
# =============================================================================

class BatchProcessor(ABC):
    """
    Abstract base class for batch processors.
    Defines interface for processing batches of data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize processor.
        
        Args:
            config: Processor configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.checkpoint_manager = CheckpointManager(config.get('checkpoint_dir', 'checkpoints'))
    
    @abstractmethod
    def process(self, data: Any, context: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """
        Process a batch of data.
        
        Args:
            data: Input data batch
            context: Processing context
            
        Returns:
            Tuple of (processed_data, metrics)
        """
        pass
    
    def save_checkpoint(self, job_id: str, partition_id: str, state: Dict[str, Any]) -> None:
        """
        Save processing checkpoint.
        
        Args:
            job_id: Job identifier
            partition_id: Partition identifier
            state: Checkpoint state
        """
        self.checkpoint_manager.save_checkpoint(job_id, partition_id, state)
    
    def load_checkpoint(self, job_id: str, partition_id: str) -> Optional[Dict[str, Any]]:
        """
        Load processing checkpoint.
        
        Args:
            job_id: Job identifier
            partition_id: Partition identifier
            
        Returns:
            Checkpoint state or None
        """
        return self.checkpoint_manager.load_checkpoint(job_id, partition_id)


class FraudDetectionBatchProcessor(BatchProcessor):
    """
    Batch processor for fraud detection.
    Handles feature engineering, model inference, and result aggregation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize fraud detection processor.
        
        Args:
            config: Configuration with model path, features, etc.
        """
        super().__init__(config)
        
        self.model_path = config.get('model_path')
        self.model = self._load_model() if self.model_path else None
        
        self.feature_columns = config.get('feature_columns', [])
        self.target_column = config.get('target_column', 'is_fraud')
        self.batch_size = config.get('batch_size', 10000)
        
        # Initialize statistics
        self.stats = {
            'total_processed': 0,
            'fraud_detected': 0,
            'avg_confidence': 0.0,
            'processing_times': []
        }
    
    def _load_model(self):
        """Load trained model from disk."""
        try:
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)
            self.logger.info(f"Loaded model from {self.model_path}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return None
    
    def process(self, data: Any, context: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """
        Process batch for fraud detection.
        
        Args:
            data: Input DataFrame
            context: Processing context
            
        Returns:
            Tuple of (results DataFrame, metrics dict)
        """
        start_time = time.time()
        
        if not isinstance(data, pd.DataFrame):
            self.logger.error(f"Expected DataFrame, got {type(data)}")
            raise ValueError(f"Expected DataFrame, got {type(data)}")
        
        if data.empty:
            self.logger.warning("Empty DataFrame received")
            return data, {'rows_processed': 0, 'processing_time': 0}
        
        # Apply feature engineering
        data = self._engineer_features(data)
        
        # Make predictions
        if self.model:
            predictions = self._predict(data)
            data['fraud_probability'] = predictions
            data['predicted_fraud'] = (predictions > 0.5).astype(int)
        
        # Calculate metrics
        metrics = self._calculate_metrics(data, context)
        
        # Update statistics
        processing_time = time.time() - start_time
        metrics['processing_time'] = processing_time
        metrics['rows_per_second'] = len(data) / processing_time if processing_time > 0 else 0
        
        self.stats['total_processed'] += len(data)
        self.stats['fraud_detected'] += data['predicted_fraud'].sum() if 'predicted_fraud' in data else 0
        self.stats['processing_times'].append(processing_time)
        
        return data, metrics
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering to DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Amount-based features
        if 'amount' in df.columns:
            df['amount_log'] = np.log1p(df['amount'])
            df['amount_sqrt'] = np.sqrt(df['amount'])
            df['amount_squared'] = df['amount'] ** 2
        
        # Time-based features
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Customer-based features
        if 'customer_id' in df.columns:
            # Calculate customer statistics (would normally use feature store)
            customer_stats = df.groupby('customer_id')['amount'].agg(['mean', 'std', 'count'])
            customer_stats.columns = ['customer_avg_amount', 'customer_std_amount', 'customer_tx_count']
            df = df.merge(customer_stats, on='customer_id', how='left')
            
            # Amount deviation
            df['amount_deviation'] = (df['amount'] - df['customer_avg_amount']) / df['customer_std_amount'].clip(lower=0.01)
        
        # Merchant-based features
        if 'merchant_id' in df.columns:
            merchant_stats = df.groupby('merchant_id')['amount'].agg(['mean', 'count'])
            merchant_stats.columns = ['merchant_avg_amount', 'merchant_tx_count']
            df = df.merge(merchant_stats, on='merchant_id', how='left')
        
        return df
    
    def _predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using loaded model.
        
        Args:
            df: Feature DataFrame
            
        Returns:
            Array of prediction probabilities
        """
        # Select feature columns
        feature_cols = [col for col in self.feature_columns if col in df.columns]
        
        if not feature_cols:
            self.logger.warning("No feature columns found, using all numeric columns")
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove target if present
            if self.target_column in feature_cols:
                feature_cols.remove(self.target_column)
        
        X = df[feature_cols].fillna(0)
        
        # Make predictions
        try:
            if hasattr(self.model, 'predict_proba'):
                predictions = self.model.predict_proba(X)[:, 1]
            else:
                predictions = self.model.predict(X)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return np.zeros(len(df))
    
    def _calculate_metrics(self, df: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate processing metrics.
        
        Args:
            df: Processed DataFrame
            context: Processing context
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'rows_processed': len(df),
            'columns': len(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
        }
        
        # Fraud-related metrics
        if 'predicted_fraud' in df.columns:
            metrics['fraud_count'] = int(df['predicted_fraud'].sum())
            metrics['fraud_rate'] = float(df['predicted_fraud'].mean())
        
        if 'fraud_probability' in df.columns:
            metrics['avg_fraud_probability'] = float(df['fraud_probability'].mean())
            metrics['max_fraud_probability'] = float(df['fraud_probability'].max())
        
        # Add context metrics
        metrics.update({
            'job_type': context.get('job_type', 'unknown'),
            'partition_id': context.get('partition_id', 'unknown')
        })
        
        return metrics


class CheckpointManager:
    """
    Manages checkpoints for fault-tolerant batch processing.
    """
    
    def __init__(self, checkpoint_dir: str):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.CheckpointManager")
        
        # Cache for loaded checkpoints
        self.cache = {}
        
        # Redis connection for distributed checkpointing
        if REDIS_AVAILABLE:
            self.redis_client = None
            redis_config = {}  # Would load from config
    
    def save_checkpoint(self, job_id: str, partition_id: str, state: Dict[str, Any]) -> None:
        """
        Save checkpoint to disk.
        
        Args:
            job_id: Job identifier
            partition_id: Partition identifier
            state: Checkpoint state to save
        """
        try:
            # Create checkpoint path
            checkpoint_path = self.checkpoint_dir / job_id / f"{partition_id}.json"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Add timestamp
            state['checkpoint_time'] = datetime.now().isoformat()
            
            # Save to file
            with open(checkpoint_path, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            # Update cache
            cache_key = f"{job_id}:{partition_id}"
            self.cache[cache_key] = state
            
            self.logger.debug(f"Saved checkpoint for {job_id}:{partition_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, job_id: str, partition_id: str) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint from disk.
        
        Args:
            job_id: Job identifier
            partition_id: Partition identifier
            
        Returns:
            Checkpoint state or None if not found
        """
        try:
            # Check cache first
            cache_key = f"{job_id}:{partition_id}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Load from file
            checkpoint_path = self.checkpoint_dir / job_id / f"{partition_id}.json"
            
            if checkpoint_path.exists():
                with open(checkpoint_path, 'r') as f:
                    state = json.load(f)
                
                # Update cache
                self.cache[cache_key] = state
                
                self.logger.debug(f"Loaded checkpoint for {job_id}:{partition_id}")
                return state
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def delete_checkpoint(self, job_id: str, partition_id: str) -> None:
        """
        Delete checkpoint.
        
        Args:
            job_id: Job identifier
            partition_id: Partition identifier
        """
        try:
            checkpoint_path = self.checkpoint_dir / job_id / f"{partition_id}.json"
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            
            # Clear cache
            cache_key = f"{job_id}:{partition_id}"
            if cache_key in self.cache:
                del self.cache[cache_key]
                
            self.logger.debug(f"Deleted checkpoint for {job_id}:{partition_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to delete checkpoint: {e}")
    
    def list_checkpoints(self, job_id: Optional[str] = None) -> List[str]:
        """
        List available checkpoints.
        
        Args:
            job_id: Optional job ID to filter
            
        Returns:
            List of checkpoint identifiers
        """
        checkpoints = []
        
        try:
            if job_id:
                job_dir = self.checkpoint_dir / job_id
                if job_dir.exists():
                    checkpoints = [f"{job_id}:{f.stem}" for f in job_dir.glob("*.json")]
            else:
                for job_dir in self.checkpoint_dir.glob("*"):
                    if job_dir.is_dir():
                        checkpoints.extend([f"{job_dir.name}:{f.stem}" 
                                          for f in job_dir.glob("*.json")])
            
            return checkpoints
            
        except Exception as e:
            self.logger.error(f"Failed to list checkpoints: {e}")
            return []


# =============================================================================
# JOB SCHEDULER AND EXECUTOR
# =============================================================================

class BatchScheduler:
    """
    Schedules and manages batch processing jobs.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize batch scheduler.
        
        Args:
            config: Scheduler configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.BatchScheduler")
        
        # Job management
        self.jobs: Dict[str, BatchJob] = {}
        self.job_queue: deque = deque()
        self.active_jobs: Dict[str, threading.Thread] = {}
        self.completed_jobs: List[BatchJob] = []
        
        # Configuration
        self.max_concurrent_jobs = config.get('max_concurrent_jobs', 4)
        self.job_timeout = config.get('job_timeout_seconds', 3600)  # 1 hour default
        self.retry_failed = config.get('retry_failed_jobs', True)
        self.max_retries = config.get('max_retries', 3)
        
        # Processing components
        self.readers: Dict[str, DataReader] = {}
        self.writers: Dict[str, DataWriter] = {}
        self.processors: Dict[str, BatchProcessor] = {}
        
        # Statistics
        self.stats = {
            'jobs_submitted': 0,
            'jobs_completed': 0,
            'jobs_failed': 0,
            'total_processing_time': 0,
            'avg_processing_time': 0
        }
        
        # Initialize from config
        self._init_components()
        
        # Start scheduler thread
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        self.logger.info("BatchScheduler initialized")
    
    def _init_components(self):
        """Initialize readers, writers, and processors from config."""
        # Initialize readers
        readers_config = self.config.get('readers', {})
        for name, reader_config in readers_config.items():
            reader_type = reader_config.get('type', 'parquet')
            if reader_type == 'parquet':
                self.readers[name] = ParquetDataReader(reader_config)
            elif reader_type == 'sql':
                self.readers[name] = SQLDataReader(reader_config)
            else:
                self.logger.warning(f"Unknown reader type: {reader_type}")
        
        # Initialize writers
        writers_config = self.config.get('writers', {})
        for name, writer_config in writers_config.items():
            writer_type = writer_config.get('type', 'parquet')
            if writer_type == 'parquet':
                self.writers[name] = ParquetDataWriter(writer_config)
            else:
                self.logger.warning(f"Unknown writer type: {writer_type}")
        
        # Initialize processors
        processors_config = self.config.get('processors', {})
        for name, processor_config in processors_config.items():
            processor_type = processor_config.get('type', 'fraud_detection')
            if processor_type == 'fraud_detection':
                self.processors[name] = FraudDetectionBatchProcessor(processor_config)
            else:
                self.logger.warning(f"Unknown processor type: {processor_type}")
    
    def submit_job(self, job: BatchJob) -> str:
        """
        Submit a new batch job.
        
        Args:
            job: Job to submit
            
        Returns:
            Job ID
        """
        if not job.job_id:
            job.job_id = f"job_{uuid.uuid4().hex[:8]}"
        
        job.status = 'pending'
        job.created_at = datetime.now()
        
        self.jobs[job.job_id] = job
        self.job_queue.append(job.job_id)
        
        self.stats['jobs_submitted'] += 1
        self.logger.info(f"Job {job.job_id} submitted (type: {job.job_type})")
        
        return job.job_id
    
    def _scheduler_loop(self):
        """Main scheduler loop that dispatches jobs."""
        while self.running:
            try:
                # Check if we can start more jobs
                if len(self.active_jobs) < self.max_concurrent_jobs and self.job_queue:
                    job_id = self.job_queue.popleft()
                    job = self.jobs.get(job_id)
                    
                    if job:
                        # Start job in new thread
                        thread = threading.Thread(
                            target=self._execute_job,
                            args=(job_id,),
                            daemon=True
                        )
                        self.active_jobs[job_id] = thread
                        thread.start()
                        
                        self.logger.info(f"Started job {job_id}")
                
                # Check for completed/failed jobs
                self._check_job_status()
                
                # Sleep to avoid busy waiting
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
                time.sleep(5)
    
    def _execute_job(self, job_id: str):
        """
        Execute a batch job.
        
        Args:
            job_id: Job to execute
        """
        job = self.jobs.get(job_id)
        if not job:
            self.logger.error(f"Job {job_id} not found")
            return
        
        start_time = time.time()
        job.started_at = datetime.now()
        job.status = 'running'
        
        try:
            # Get components
            reader = self.readers.get(job.config.get('reader', 'default'))
            if not reader:
                raise ValueError(f"Reader not found: {job.config.get('reader')}")
            
            writer = self.writers.get(job.config.get('writer', 'default'))
            if not writer:
                raise ValueError(f"Writer not found: {job.config.get('writer')}")
            
            processor = self.processors.get(job.config.get('processor', 'default'))
            if not processor:
                raise ValueError(f"Processor not found: {job.config.get('processor')}")
            
            # Get partitions
            source = job.config.get('source')
            partition_size = job.config.get('partition_size', 10000)
            
            partitions = reader.get_partitions(source, partition_size)
            self.logger.info(f"Job {job_id}: Created {len(partitions)} partitions")
            
            # Process partitions in parallel
            results = self._process_partitions(job, reader, processor, partitions)
            
            # Combine results
            combined_results = self._combine_results(results)
            
            # Write output
            output_path = job.config.get('output_path')
            if output_path and combined_results is not None:
                writer.write(combined_results, output_path)
                job.output_path = output_path
            
            # Update job
            job.status = 'completed'
            job.completed_at = datetime.now()
            
            processing_time = (job.completed_at - job.started_at).total_seconds()
            job.metrics = {
                'processing_time': processing_time,
                'partitions_processed': len(results),
                'total_rows': sum(r.get('rows_processed', 0) for r in results),
                'fraud_detected': sum(r.get('fraud_detected', 0) for r in results)
            }
            
            self.stats['jobs_completed'] += 1
            self.stats['total_processing_time'] += processing_time
            
            self.logger.info(f"Job {job_id} completed in {processing_time:.2f}s")
            
        except Exception as e:
            job.status = 'failed'
            job.error = str(e)
            job.completed_at = datetime.now()
            
            self.stats['jobs_failed'] += 1
            
            self.logger.error(f"Job {job_id} failed: {e}")
            
            # Handle retry
            if self.retry_failed:
                retry_count = job.metadata.get('retry_count', 0) + 1
                if retry_count <= self.max_retries:
                    self.logger.info(f"Retrying job {job_id} (attempt {retry_count})")
                    job.metadata['retry_count'] = retry_count
                    job.status = 'pending'
                    self.job_queue.append(job_id)
    
    def _process_partitions(self, job: BatchJob, reader: DataReader, 
                           processor: BatchProcessor, 
                           partitions: List[DataPartition]) -> List[Dict[str, Any]]:
        """
        Process partitions in parallel.
        
        Args:
            job: Current job
            reader: Data reader
            processor: Batch processor
            partitions: List of partitions
            
        Returns:
            List of processing results
        """
        results = []
        max_workers = job.config.get('max_workers', mp.cpu_count())
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_partition = {}
            for partition in partitions:
                # Check for checkpoint
                checkpoint = processor.load_checkpoint(job.job_id, partition.partition_id)
                if checkpoint:
                    self.logger.info(f"Resuming from checkpoint for {partition.partition_id}")
                    context = checkpoint.get('context', {})
                    context['resumed'] = True
                else:
                    context = {
                        'job_id': job.job_id,
                        'job_type': job.job_type,
                        'partition_id': partition.partition_id
                    }
                
                future = executor.submit(
                    self._process_single_partition,
                    job, reader, processor, partition, context
                )
                future_to_partition[future] = partition
            
            # Collect results
            for future in as_completed(future_to_partition):
                partition = future_to_partition[future]
                try:
                    result = future.result(timeout=job.config.get('partition_timeout', 300))
                    results.append(result)
                    
                    # Save checkpoint
                    processor.save_checkpoint(job.job_id, partition.partition_id, {
                        'processed': True,
                        'result': result,
                        'context': {
                            'job_id': job.job_id,
                            'partition_id': partition.partition_id,
                            'completed_at': datetime.now().isoformat()
                        }
                    })
                    
                except Exception as e:
                    self.logger.error(f"Partition {partition.partition_id} failed: {e}")
                    # Don't re-raise, continue with other partitions
        
        return results
    
    def _process_single_partition(self, job: BatchJob, reader: DataReader,
                                  processor: BatchProcessor,
                                  partition: DataPartition,
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single partition.
        
        Args:
            job: Current job
            reader: Data reader
            processor: Batch processor
            partition: Partition to process
            context: Processing context
            
        Returns:
            Processing result metrics
        """
        try:
            # Read partition data
            data = reader.read_partition(partition)
            
            # Process data
            processed_data, metrics = processor.process(data, context)
            
            # Add partition info to metrics
            metrics['partition_id'] = partition.partition_id
            metrics['rows_in_partition'] = len(data)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error processing partition {partition.partition_id}: {e}")
            raise
    
    def _combine_results(self, results: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
        """
        Combine results from all partitions.
        
        Args:
            results: List of result metrics
            
        Returns:
            Combined DataFrame or None
        """
        # This is a placeholder - actual implementation would
        # combine the actual data, not just metrics
        return None
    
    def _check_job_status(self):
        """Check status of running jobs."""
        completed = []
        
        for job_id, thread in list(self.active_jobs.items()):
            if not thread.is_alive():
                completed.append(job_id)
        
        for job_id in completed:
            del self.active_jobs[job_id]
            self.logger.debug(f"Job {job_id} removed from active jobs")
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job status dictionary or None
        """
        job = self.jobs.get(job_id)
        if job:
            return {
                'job_id': job.job_id,
                'job_type': job.job_type,
                'status': job.status,
                'created_at': job.created_at.isoformat(),
                'started_at': job.started_at.isoformat() if job.started_at else None,
                'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                'metrics': job.metrics,
                'error': job.error
            }
        return None
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running or pending job.
        
        Args:
            job_id: Job to cancel
            
        Returns:
            True if cancelled, False if not found
        """
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        if job.status == 'pending':
            # Remove from queue
            if job_id in self.job_queue:
                self.job_queue.remove(job_id)
            job.status = 'cancelled'
            return True
            
        elif job.status == 'running':
            # Can't easily cancel running thread, mark for cancellation
            job.status = 'cancelling'
            # Thread should check status periodically
            return True
        
        return False
    
    def shutdown(self):
        """Shutdown the scheduler."""
        self.running = False
        self.scheduler_thread.join(timeout=5)
        self.logger.info("BatchScheduler shutdown complete")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_batch_pipeline(config_path: Optional[str] = None) -> BatchScheduler:
    """
    Create and configure batch processing pipeline.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured BatchScheduler instance
    """
    config = {}
    
    if config_path:
        path = Path(config_path)
        try:
            if path.suffix == '.json':
                with open(path, 'r') as f:
                    config = json.load(f)
            elif path.suffix in ['.yaml', '.yml']:
                import yaml
                with open(path, 'r') as f:
                    config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
    
    # Default configuration
    if not config:
        config = {
            'max_concurrent_jobs': 4,
            'job_timeout_seconds': 3600,
            'retry_failed_jobs': True,
            'max_retries': 3,
            'checkpoint_dir': 'checkpoints',
            
            'readers': {
                'default': {
                    'type': 'parquet',
                    'file_pattern': '*.parquet',
                    'batch_size': 100000
                },
                'transactions': {
                    'type': 'parquet',
                    'file_pattern': 'transactions_*.parquet'
                }
            },
            
            'writers': {
                'default': {
                    'type': 'parquet',
                    'compression': 'snappy'
                }
            },
            
            'processors': {
                'default': {
                    'type': 'fraud_detection',
                    'model_path': 'artifacts/models/fraud_model.pkl',
                    'feature_columns': [
                        'amount', 'amount_log', 'hour', 'day_of_week',
                        'customer_avg_amount', 'amount_deviation'
                    ],
                    'batch_size': 10000
                }
            }
        }
    
    return BatchScheduler(config)


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    """
    Example usage of batch processing pipeline.
    """
    
    # Create pipeline
    scheduler = create_batch_pipeline()
    
    # Create a job
    job = BatchJob(
        job_id="fraud_batch_001",
        job_type="inference",
        status="pending",
        config={
            'reader': 'default',
            'writer': 'default',
            'processor': 'default',
            'source': 'data/processed/transactions_202401.parquet',
            'output_path': 'data/results/fraud_predictions_202401.parquet',
            'partition_size': 50000,
            'max_workers': 8
        },
        input_paths=['data/processed/transactions_202401.parquet']
    )
    
    # Submit job
    job_id = scheduler.submit_job(job)
    print(f"Submitted job: {job_id}")
    
    # Wait for completion (in production, use async monitoring)
    try:
        time.sleep(5)  # Give job time to start
        
        while True:
            status = scheduler.get_job_status(job_id)
            if status:
                print(f"Job status: {status['status']}")
                if status['status'] in ['completed', 'failed', 'cancelled']:
                    if status['status'] == 'completed':
                        print(f"Job completed in {status['metrics']['processing_time']:.2f}s")
                        print(f"Processed {status['metrics']['total_rows']} rows")
                        print(f"Detected {status['metrics']['fraud_detected']} frauds")
                    elif status['status'] == 'failed':
                        print(f"Job failed: {status['error']}")
                    break
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("Shutting down...")
        scheduler.shutdown()
    
    print("Batch processing example complete")