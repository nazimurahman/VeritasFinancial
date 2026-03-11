# =============================================================================
# VERITASFINANCIAL - FRAUD DETECTION SYSTEM
# Module: deployment/pipeline/feature_store.py
# Description: Feature store for online/offline feature management
# Author: Data Science Team
# Version: 2.0.0
# Last Updated: 2024-01-15
# =============================================================================

"""
Feature Store for Fraud Detection
===================================
This module provides a comprehensive feature store for:
- Online feature serving (low-latency)
- Offline feature storage for training
- Feature versioning and lineage
- Point-in-time correctness
- Feature validation and monitoring
- Feature registry and discovery
- Integration with ML pipelines
"""

import json
import time
import uuid
import hashlib
import pickle
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
from pathlib import Path

# Third-party imports with error handling
try:
    import pandas as pd
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logging.warning("Pandas not available")

try:
    import redis
    import redis.asyncio as redis_async
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available")

try:
    import sqlalchemy as sa
    from sqlalchemy import create_engine, text, Column, String, Float, Integer, DateTime, JSON
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
    SQL_AVAILABLE = True
except ImportError:
    SQL_AVAILABLE = False
    logging.warning("SQLAlchemy not available")

try:
    import boto3
    from botocore.exceptions import ClientError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    logging.warning("Boto3 not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/feature_store.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class FeatureType(Enum):
    """Types of features in the store."""
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    COUNT = "count"
    EMBEDDING = "embedding"
    TEXT = "text"
    TIME = "time"


class FeatureGroup(Enum):
    """Logical groups of features."""
    TRANSACTION = "transaction"
    CUSTOMER = "customer"
    DEVICE = "device"
    MERCHANT = "merchant"
    LOCATION = "location"
    BEHAVIORAL = "behavioral"
    TEMPORAL = "temporal"
    AGGREGATE = "aggregate"
    GRAPH = "graph"


@dataclass
class FeatureDefinition:
    """
    Definition of a feature in the feature store.
    
    Attributes:
        name: Feature name
        feature_type: Type of feature
        group: Feature group
        description: Feature description
        source: Data source (table, column, etc.)
        transformation: Transformation applied
        version: Feature version
        created_at: Creation timestamp
        owner: Feature owner/team
        tags: Categorization tags
        validation_rules: Rules for validating feature values
        metadata: Additional metadata
    """
    name: str
    feature_type: FeatureType
    group: FeatureGroup
    description: str
    source: str
    transformation: Optional[str] = None
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    owner: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['feature_type'] = self.feature_type.value
        result['group'] = self.group.value
        result['created_at'] = self.created_at.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureDefinition':
        """Create from dictionary."""
        data['feature_type'] = FeatureType(data['feature_type'])
        data['group'] = FeatureGroup(data['group'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


@dataclass
class FeatureVector:
    """
    Vector of features for a specific entity.
    
    Attributes:
        entity_id: Entity identifier (customer_id, transaction_id, etc.)
        entity_type: Type of entity
        features: Dictionary of feature_name -> value
        timestamp: When the features were computed
        source: Source of the features
        version: Feature version
    """
    entity_id: str
    entity_type: str
    features: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "computed"
    version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'entity_id': self.entity_id,
            'entity_type': self.entity_type,
            'features': self.features,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'version': self.version
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureVector':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


# =============================================================================
# STORAGE BACKENDS
# =============================================================================

class StorageBackend(ABC):
    """
    Abstract base class for feature storage backends.
    Defines interface for storing and retrieving features.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize storage backend.
        
        Args:
            config: Backend configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def get(self, namespace: str, key: str) -> Optional[Any]:
        """Get value for key."""
        pass
    
    @abstractmethod
    async def set(self, namespace: str, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value for key."""
        pass
    
    @abstractmethod
    async def mget(self, namespace: str, keys: List[str]) -> List[Optional[Any]]:
        """Get multiple values."""
        pass
    
    @abstractmethod
    async def mset(self, namespace: str, items: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values."""
        pass
    
    @abstractmethod
    async def delete(self, namespace: str, key: str) -> bool:
        """Delete key."""
        pass
    
    @abstractmethod
    async def exists(self, namespace: str, key: str) -> bool:
        """Check if key exists."""
        pass
    
    @abstractmethod
    async def keys(self, namespace: str, pattern: str = "*") -> List[str]:
        """List keys matching pattern."""
        pass


class RedisBackend(StorageBackend):
    """
    Redis storage backend for low-latency online feature serving.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Redis backend.
        
        Args:
            config: Redis configuration (host, port, db, etc.)
        """
        super().__init__(config)
        
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is required for RedisBackend")
        
        self.host = config.get('host', 'localhost')
        self.port = config.get('port', 6379)
        self.db = config.get('db', 0)
        self.password = config.get('password', None)
        self.ssl = config.get('ssl', False)
        self.prefix = config.get('key_prefix', 'feature_store:')
        
        # Connection pool
        self.pool = redis.ConnectionPool(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            ssl=self.ssl,
            decode_responses=True,
            max_connections=config.get('max_connections', 10)
        )
        
        # Sync client
        self.client = redis.Redis(connection_pool=self.pool)
        
        # Async client
        self.async_client = None
        if config.get('async_enabled', True):
            self.async_client = redis_async.Redis(
                connection_pool=self.pool,
                decode_responses=True
            )
    
    def _make_key(self, namespace: str, key: str) -> str:
        """Create full Redis key."""
        return f"{self.prefix}{namespace}:{key}"
    
    async def get(self, namespace: str, key: str) -> Optional[Any]:
        """Get value from Redis."""
        try:
            full_key = self._make_key(namespace, key)
            
            if self.async_client:
                value = await self.async_client.get(full_key)
            else:
                value = self.client.get(full_key)
            
            if value:
                return json.loads(value)
            return None
            
        except Exception as e:
            self.logger.error(f"Redis get failed: {e}")
            return None
    
    async def set(self, namespace: str, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis."""
        try:
            full_key = self._make_key(namespace, key)
            serialized = json.dumps(value, default=str)
            
            if self.async_client:
                if ttl:
                    await self.async_client.setex(full_key, ttl, serialized)
                else:
                    await self.async_client.set(full_key, serialized)
            else:
                if ttl:
                    self.client.setex(full_key, ttl, serialized)
                else:
                    self.client.set(full_key, serialized)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Redis set failed: {e}")
            return False
    
    async def mget(self, namespace: str, keys: List[str]) -> List[Optional[Any]]:
        """Get multiple values."""
        try:
            full_keys = [self._make_key(namespace, k) for k in keys]
            
            if self.async_client:
                values = await self.async_client.mget(full_keys)
            else:
                values = self.client.mget(full_keys)
            
            results = []
            for v in values:
                if v:
                    results.append(json.loads(v))
                else:
                    results.append(None)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Redis mget failed: {e}")
            return [None] * len(keys)
    
    async def mset(self, namespace: str, items: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values."""
        try:
            # For mset, we need to handle TTL differently
            # Redis doesn't support TTL in MSET, so we set individually
            tasks = []
            for key, value in items.items():
                tasks.append(self.set(namespace, key, value, ttl))
            
            if self.async_client:
                results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                results = [await t if asyncio.iscoroutine(t) else t for t in tasks]
            
            return all(results)
            
        except Exception as e:
            self.logger.error(f"Redis mset failed: {e}")
            return False
    
    async def delete(self, namespace: str, key: str) -> bool:
        """Delete key."""
        try:
            full_key = self._make_key(namespace, key)
            
            if self.async_client:
                result = await self.async_client.delete(full_key)
            else:
                result = self.client.delete(full_key)
            
            return result > 0
            
        except Exception as e:
            self.logger.error(f"Redis delete failed: {e}")
            return False
    
    async def exists(self, namespace: str, key: str) -> bool:
        """Check if key exists."""
        try:
            full_key = self._make_key(namespace, key)
            
            if self.async_client:
                result = await self.async_client.exists(full_key)
            else:
                result = self.client.exists(full_key)
            
            return result > 0
            
        except Exception as e:
            self.logger.error(f"Redis exists failed: {e}")
            return False
    
    async def keys(self, namespace: str, pattern: str = "*") -> List[str]:
        """List keys matching pattern."""
        try:
            full_pattern = self._make_key(namespace, pattern)
            
            if self.async_client:
                keys = await self.async_client.keys(full_pattern)
            else:
                keys = self.client.keys(full_pattern)
            
            # Remove prefix
            prefix_len = len(self._make_key(namespace, ""))
            return [k[prefix_len:] for k in keys]
            
        except Exception as e:
            self.logger.error(f"Redis keys failed: {e}")
            return []
    
    async def close(self):
        """Close connections."""
        if self.async_client:
            await self.async_client.close()
        self.client.close()


class SQLBackend(StorageBackend):
    """
    SQL storage backend for persistent feature storage.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SQL backend.
        
        Args:
            config: SQL configuration (connection string, table name, etc.)
        """
        super().__init__(config)
        
        if not SQL_AVAILABLE:
            raise ImportError("SQLAlchemy is required for SQLBackend")
        
        self.connection_string = config.get('connection_string', 'sqlite:///feature_store.db')
        self.table_name = config.get('table_name', 'feature_store')
        
        # Create engine
        self.engine = create_engine(
            self.connection_string,
            pool_size=config.get('pool_size', 5),
            max_overflow=config.get('max_overflow', 10)
        )
        
        # Create session factory
        self.Session = sessionmaker(bind=self.engine)
        
        # Create table if not exists
        self._create_table()
    
    def _create_table(self):
        """Create feature store table if not exists."""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            namespace VARCHAR(255) NOT NULL,
            key VARCHAR(255) NOT NULL,
            value JSON NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NULL,
            PRIMARY KEY (namespace, key)
        )
        """
        
        with self.engine.connect() as conn:
            conn.execute(text(create_table_sql))
            conn.commit()
    
    async def get(self, namespace: str, key: str) -> Optional[Any]:
        """Get value from SQL."""
        try:
            query = text(f"""
                SELECT value FROM {self.table_name}
                WHERE namespace = :namespace AND key = :key
                AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """)
            
            with self.Session() as session:
                result = session.execute(
                    query,
                    {'namespace': namespace, 'key': key}
                ).first()
                
                if result:
                    return json.loads(result[0])
                return None
                
        except Exception as e:
            self.logger.error(f"SQL get failed: {e}")
            return None
    
    async def set(self, namespace: str, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in SQL."""
        try:
            expires_at = None
            if ttl:
                expires_at = datetime.now() + timedelta(seconds=ttl)
            
            # Upsert
            query = text(f"""
                INSERT INTO {self.table_name} (namespace, key, value, expires_at)
                VALUES (:namespace, :key, :value, :expires_at)
                ON CONFLICT (namespace, key) DO UPDATE SET
                    value = EXCLUDED.value,
                    expires_at = EXCLUDED.expires_at,
                    updated_at = CURRENT_TIMESTAMP
            """)
            
            with self.Session() as session:
                session.execute(
                    query,
                    {
                        'namespace': namespace,
                        'key': key,
                        'value': json.dumps(value, default=str),
                        'expires_at': expires_at
                    }
                )
                session.commit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"SQL set failed: {e}")
            return False
    
    async def mget(self, namespace: str, keys: List[str]) -> List[Optional[Any]]:
        """Get multiple values."""
        try:
            query = text(f"""
                SELECT key, value FROM {self.table_name}
                WHERE namespace = :namespace AND key = ANY(:keys)
                AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """)
            
            with self.Session() as session:
                results = session.execute(
                    query,
                    {'namespace': namespace, 'keys': keys}
                ).fetchall()
            
            # Map results to keys
            value_map = {r[0]: json.loads(r[1]) for r in results}
            return [value_map.get(k) for k in keys]
            
        except Exception as e:
            self.logger.error(f"SQL mget failed: {e}")
            return [None] * len(keys)
    
    async def mset(self, namespace: str, items: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values."""
        try:
            expires_at = None
            if ttl:
                expires_at = datetime.now() + timedelta(seconds=ttl)
            
            with self.Session() as session:
                for key, value in items.items():
                    query = text(f"""
                        INSERT INTO {self.table_name} (namespace, key, value, expires_at)
                        VALUES (:namespace, :key, :value, :expires_at)
                        ON CONFLICT (namespace, key) DO UPDATE SET
                            value = EXCLUDED.value,
                            expires_at = EXCLUDED.expires_at,
                            updated_at = CURRENT_TIMESTAMP
                    """)
                    
                    session.execute(
                        query,
                        {
                            'namespace': namespace,
                            'key': key,
                            'value': json.dumps(value, default=str),
                            'expires_at': expires_at
                        }
                    )
                
                session.commit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"SQL mset failed: {e}")
            return False
    
    async def delete(self, namespace: str, key: str) -> bool:
        """Delete key."""
        try:
            query = text(f"""
                DELETE FROM {self.table_name}
                WHERE namespace = :namespace AND key = :key
            """)
            
            with self.Session() as session:
                result = session.execute(query, {'namespace': namespace, 'key': key})
                session.commit()
            
            return result.rowcount > 0
            
        except Exception as e:
            self.logger.error(f"SQL delete failed: {e}")
            return False
    
    async def exists(self, namespace: str, key: str) -> bool:
        """Check if key exists."""
        try:
            query = text(f"""
                SELECT 1 FROM {self.table_name}
                WHERE namespace = :namespace AND key = :key
                AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """)
            
            with self.Session() as session:
                result = session.execute(query, {'namespace': namespace, 'key': key}).first()
            
            return result is not None
            
        except Exception as e:
            self.logger.error(f"SQL exists failed: {e}")
            return False
    
    async def keys(self, namespace: str, pattern: str = "*") -> List[str]:
        """List keys matching pattern."""
        try:
            # Convert glob pattern to SQL LIKE
            like_pattern = pattern.replace('*', '%').replace('?', '_')
            
            query = text(f"""
                SELECT key FROM {self.table_name}
                WHERE namespace = :namespace AND key LIKE :pattern
            """)
            
            with self.Session() as session:
                results = session.execute(
                    query,
                    {'namespace': namespace, 'pattern': like_pattern}
                ).fetchall()
            
            return [r[0] for r in results]
            
        except Exception as e:
            self.logger.error(f"SQL keys failed: {e}")
            return []


class S3Backend(StorageBackend):
    """
    S3 storage backend for offline feature storage (training data).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize S3 backend.
        
        Args:
            config: S3 configuration (bucket, prefix, etc.)
        """
        super().__init__(config)
        
        if not S3_AVAILABLE:
            raise ImportError("Boto3 is required for S3Backend")
        
        self.bucket = config.get('bucket')
        self.prefix = config.get('prefix', 'feature_store/')
        self.region = config.get('region', 'us-east-1')
        
        if not self.bucket:
            raise ValueError("S3 bucket must be specified")
        
        # Create S3 client
        self.s3 = boto3.client(
            's3',
            region_name=self.region,
            aws_access_key_id=config.get('aws_access_key_id'),
            aws_secret_access_key=config.get('aws_secret_access_key')
        )
        
        # Ensure bucket exists
        self._ensure_bucket()
    
    def _ensure_bucket(self):
        """Ensure S3 bucket exists."""
        try:
            self.s3.head_bucket(Bucket=self.bucket)
        except ClientError:
            # Bucket doesn't exist, create it
            if self.region == 'us-east-1':
                self.s3.create_bucket(Bucket=self.bucket)
            else:
                self.s3.create_bucket(
                    Bucket=self.bucket,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
    
    def _make_key(self, namespace: str, key: str) -> str:
        """Create full S3 key."""
        return f"{self.prefix}{namespace}/{key}.json"
    
    async def get(self, namespace: str, key: str) -> Optional[Any]:
        """Get value from S3."""
        try:
            s3_key = self._make_key(namespace, key)
            
            response = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
            content = response['Body'].read().decode('utf-8')
            
            return json.loads(content)
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            self.logger.error(f"S3 get failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"S3 get failed: {e}")
            return None
    
    async def set(self, namespace: str, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in S3."""
        try:
            s3_key = self._make_key(namespace, key)
            
            # Add metadata
            metadata = {
                'created_at': datetime.now().isoformat(),
                'namespace': namespace,
                'key': key
            }
            
            if ttl:
                expires_at = datetime.now() + timedelta(seconds=ttl)
                metadata['expires_at'] = expires_at.isoformat()
            
            # Upload
            self.s3.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=json.dumps(value, default=str).encode('utf-8'),
                Metadata=metadata,
                ContentType='application/json'
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"S3 set failed: {e}")
            return False
    
    async def mget(self, namespace: str, keys: List[str]) -> List[Optional[Any]]:
        """Get multiple values."""
        # S3 doesn't support batch get, so we do individual gets
        results = []
        for key in keys:
            results.append(await self.get(namespace, key))
        return results
    
    async def mset(self, namespace: str, items: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values."""
        results = []
        for key, value in items.items():
            results.append(await self.set(namespace, key, value, ttl))
        return all(results)
    
    async def delete(self, namespace: str, key: str) -> bool:
        """Delete key."""
        try:
            s3_key = self._make_key(namespace, key)
            self.s3.delete_object(Bucket=self.bucket, Key=s3_key)
            return True
        except Exception as e:
            self.logger.error(f"S3 delete failed: {e}")
            return False
    
    async def exists(self, namespace: str, key: str) -> bool:
        """Check if key exists."""
        try:
            s3_key = self._make_key(namespace, key)
            self.s3.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            self.logger.error(f"S3 exists failed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"S3 exists failed: {e}")
            return False
    
    async def keys(self, namespace: str, pattern: str = "*") -> List[str]:
        """List keys matching pattern."""
        try:
            prefix = f"{self.prefix}{namespace}/"
            
            # List objects with prefix
            paginator = self.s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket, Prefix=prefix)
            
            keys = []
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        # Extract key without prefix and extension
                        key = obj['Key'][len(prefix):-5]  # Remove .json
                        
                        # Apply pattern matching (simple glob)
                        if self._match_pattern(key, pattern):
                            keys.append(key)
            
            return keys
            
        except Exception as e:
            self.logger.error(f"S3 keys failed: {e}")
            return []
    
    def _match_pattern(self, key: str, pattern: str) -> bool:
        """Simple pattern matching for keys."""
        if pattern == "*":
            return True
        
        # Convert glob to regex
        import re
        regex = pattern.replace('.', '\\.').replace('*', '.*').replace('?', '.')
        return re.match(f"^{regex}$", key) is not None


# =============================================================================
# FEATURE COMPUTATION ENGINE
# =============================================================================

class FeatureComputationEngine:
    """
    Engine for computing features in real-time and batch.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize computation engine.
        
        Args:
            config: Engine configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.FeatureComputationEngine")
        
        # Feature definitions
        self.feature_defs: Dict[str, FeatureDefinition] = {}
        
        # Computation functions
        self.computers: Dict[str, Callable] = {}
        
        # Statistics
        self.stats = {
            'computations': 0,
            'errors': 0,
            'avg_computation_time_ms': 0
        }
    
    def register_feature(self, feature_def: FeatureDefinition, 
                        computer: Callable) -> None:
        """
        Register a feature with its computation function.
        
        Args:
            feature_def: Feature definition
            computer: Function that computes the feature
        """
        self.feature_defs[feature_def.name] = feature_def
        self.computers[feature_def.name] = computer
        
        self.logger.info(f"Registered feature: {feature_def.name}")
    
    async def compute_feature(self, feature_name: str, 
                             entity_data: Dict[str, Any]) -> Any:
        """
        Compute a single feature.
        
        Args:
            feature_name: Name of feature to compute
            entity_data: Entity data for computation
            
        Returns:
            Computed feature value
        """
        start_time = time.time()
        
        if feature_name not in self.computers:
            raise ValueError(f"Unknown feature: {feature_name}")
        
        try:
            computer = self.computers[feature_name]
            
            # Handle both sync and async computers
            if asyncio.iscoroutinefunction(computer):
                value = await computer(entity_data)
            else:
                value = computer(entity_data)
            
            # Validate value
            feature_def = self.feature_defs[feature_name]
            self._validate_value(feature_name, value, feature_def)
            
            # Update statistics
            computation_time = (time.time() - start_time) * 1000
            self.stats['computations'] += 1
            self.stats['avg_computation_time_ms'] = (
                (self.stats['avg_computation_time_ms'] * (self.stats['computations'] - 1) + computation_time) /
                self.stats['computations']
            )
            
            return value
            
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Feature computation failed for {feature_name}: {e}")
            raise
    
    async def compute_feature_vector(self, entity_id: str, entity_type: str,
                                     entity_data: Dict[str, Any],
                                     feature_names: List[str]) -> FeatureVector:
        """
        Compute multiple features for an entity.
        
        Args:
            entity_id: Entity identifier
            entity_type: Type of entity
            entity_data: Entity data for computation
            feature_names: List of features to compute
            
        Returns:
            FeatureVector with computed features
        """
        features = {}
        
        # Compute all requested features
        for feature_name in feature_names:
            try:
                value = await self.compute_feature(feature_name, entity_data)
                features[feature_name] = value
            except Exception as e:
                self.logger.warning(f"Could not compute {feature_name}: {e}")
                features[feature_name] = None
        
        # Create feature vector
        vector = FeatureVector(
            entity_id=entity_id,
            entity_type=entity_type,
            features=features,
            source="computed",
            version=self.config.get('version', '1.0.0')
        )
        
        return vector
    
    def _validate_value(self, feature_name: str, value: Any, 
                       feature_def: FeatureDefinition) -> None:
        """
        Validate feature value against definition.
        
        Args:
            feature_name: Feature name
            value: Value to validate
            feature_def: Feature definition with validation rules
        """
        rules = feature_def.validation_rules
        
        # Check type
        if feature_def.feature_type == FeatureType.CONTINUOUS:
            if not isinstance(value, (int, float)):
                raise ValueError(f"Expected numeric, got {type(value)}")
            
            # Check range
            if 'min' in rules and value < rules['min']:
                raise ValueError(f"Value {value} below minimum {rules['min']}")
            if 'max' in rules and value > rules['max']:
                raise ValueError(f"Value {value} above maximum {rules['max']}")
        
        elif feature_def.feature_type == FeatureType.BINARY:
            if value not in [0, 1]:
                raise ValueError(f"Expected binary (0/1), got {value}")
        
        elif feature_def.feature_type == FeatureType.CATEGORICAL:
            if 'allowed_values' in rules and value not in rules['allowed_values']:
                raise ValueError(f"Value {value} not in allowed set")


# =============================================================================
# FEATURE REGISTRY
# =============================================================================

class FeatureRegistry:
    """
    Registry for feature definitions and metadata.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature registry.
        
        Args:
            config: Registry configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.FeatureRegistry")
        
        # Feature definitions
        self.features: Dict[str, FeatureDefinition] = {}
        
        # Feature groups
        self.groups: Dict[FeatureGroup, List[str]] = defaultdict(list)
        
        # Dependencies between features
        self.dependencies: Dict[str, List[str]] = defaultdict(list)
        
        # Storage backend for registry metadata
        backend_type = config.get('backend', 'memory')
        if backend_type == 'redis':
            self.backend = RedisBackend(config.get('redis', {}))
        elif backend_type == 'sql':
            self.backend = SQLBackend(config.get('sql', {}))
        else:
            self.backend = None  # In-memory only
    
    def register_feature(self, feature_def: FeatureDefinition) -> None:
        """
        Register a feature definition.
        
        Args:
            feature_def: Feature definition to register
        """
        if feature_def.name in self.features:
            self.logger.warning(f"Overwriting feature: {feature_def.name}")
        
        self.features[feature_def.name] = feature_def
        self.groups[feature_def.group].append(feature_def.name)
        
        self.logger.info(f"Registered feature: {feature_def.name}")
    
    def get_feature(self, name: str) -> Optional[FeatureDefinition]:
        """
        Get feature definition by name.
        
        Args:
            name: Feature name
            
        Returns:
            Feature definition or None
        """
        return self.features.get(name)
    
    def list_features(self, group: Optional[FeatureGroup] = None) -> List[str]:
        """
        List all features, optionally filtered by group.
        
        Args:
            group: Optional group filter
            
        Returns:
            List of feature names
        """
        if group:
            return self.groups.get(group, [])
        return list(self.features.keys())
    
    def get_features_by_type(self, feature_type: FeatureType) -> List[str]:
        """
        Get features of a specific type.
        
        Args:
            feature_type: Type to filter by
            
        Returns:
            List of feature names
        """
        return [
            name for name, defn in self.features.items()
            if defn.feature_type == feature_type
        ]
    
    def add_dependency(self, feature: str, depends_on: str) -> None:
        """
        Add a dependency between features.
        
        Args:
            feature: Feature that depends on another
            depends_on: Feature that is depended upon
        """
        if feature not in self.features:
            raise ValueError(f"Unknown feature: {feature}")
        if depends_on not in self.features:
            raise ValueError(f"Unknown feature: {depends_on}")
        
        self.dependencies[feature].append(depends_on)
    
    def get_dependencies(self, feature: str) -> List[str]:
        """
        Get dependencies for a feature.
        
        Args:
            feature: Feature name
            
        Returns:
            List of dependent feature names
        """
        return self.dependencies.get(feature, [])
    
    def get_dependents(self, feature: str) -> List[str]:
        """
        Get features that depend on this feature.
        
        Args:
            feature: Feature name
            
        Returns:
            List of features that depend on it
        """
        return [
            f for f, deps in self.dependencies.items()
            if feature in deps
        ]
    
    async def save(self) -> bool:
        """Save registry to storage backend."""
        if not self.backend:
            return False
        
        try:
            # Save feature definitions
            for name, defn in self.features.items():
                await self.backend.set(
                    'feature_registry',
                    f"def:{name}",
                    defn.to_dict()
                )
            
            # Save groups
            for group, features in self.groups.items():
                await self.backend.set(
                    'feature_registry',
                    f"group:{group.value}",
                    features
                )
            
            # Save dependencies
            await self.backend.set(
                'feature_registry',
                'dependencies',
                self.dependencies
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save registry: {e}")
            return False
    
    async def load(self) -> bool:
        """Load registry from storage backend."""
        if not self.backend:
            return False
        
        try:
            # Clear current registry
            self.features.clear()
            self.groups.clear()
            self.dependencies.clear()
            
            # Load feature definitions
            keys = await self.backend.keys('feature_registry', 'def:*')
            for key in keys:
                data = await self.backend.get('feature_registry', key)
                if data:
                    defn = FeatureDefinition.from_dict(data)
                    self.features[defn.name] = defn
                    self.groups[defn.group].append(defn.name)
            
            # Load groups (redundant but ensure consistency)
            group_keys = await self.backend.keys('feature_registry', 'group:*')
            for key in group_keys:
                features = await self.backend.get('feature_registry', key)
                if features:
                    group_name = key[6:]  # Remove 'group:'
                    group = FeatureGroup(group_name)
                    self.groups[group] = features
            
            # Load dependencies
            deps = await self.backend.get('feature_registry', 'dependencies')
            if deps:
                self.dependencies = deps
            
            self.logger.info(f"Loaded {len(self.features)} features from registry")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load registry: {e}")
            return False


# =============================================================================
# MAIN FEATURE STORE
# =============================================================================

class FeatureStore:
    """
    Main feature store for online and offline feature management.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature store.
        
        Args:
            config: Feature store configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.FeatureStore")
        
        # Storage backends
        self.backends: Dict[str, StorageBackend] = {}
        
        # Online backend (low latency)
        online_config = config.get('online', {})
        online_type = online_config.get('type', 'redis')
        if online_type == 'redis':
            self.backends['online'] = RedisBackend(online_config)
        else:
            self.logger.warning(f"Unknown online backend: {online_type}")
        
        # Offline backend (batch/analytics)
        offline_config = config.get('offline', {})
        offline_type = offline_config.get('type', 'sql')
        if offline_type == 'sql':
            self.backends['offline'] = SQLBackend(offline_config)
        elif offline_type == 's3':
            self.backends['offline'] = S3Backend(offline_config)
        else:
            self.logger.warning(f"Unknown offline backend: {offline_type}")
        
        # Registry
        self.registry = FeatureRegistry(config.get('registry', {}))
        
        # Computation engine
        self.engine = FeatureComputationEngine(config.get('computation', {}))
        
        # Cache for frequently accessed features
        self.cache = FeatureCache(config.get('cache', {}))
        
        # Statistics
        self.stats = {
            'online_gets': 0,
            'online_sets': 0,
            'offline_gets': 0,
            'offline_sets': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def register_feature(self, feature_def: FeatureDefinition,
                        computer: Optional[Callable] = None) -> None:
        """
        Register a feature.
        
        Args:
            feature_def: Feature definition
            computer: Optional computation function
        """
        # Register in registry
        self.registry.register_feature(feature_def)
        
        # Register computer if provided
        if computer:
            self.engine.register_feature(feature_def, computer)
        
        self.logger.info(f"Feature {feature_def.name} registered")
    
    async def get_online(self, namespace: str, key: str) -> Optional[Any]:
        """
        Get feature from online store.
        
        Args:
            namespace: Feature namespace
            key: Feature key
            
        Returns:
            Feature value or None
        """
        # Try cache first
        cache_key = f"{namespace}:{key}"
        cached = await self.cache.get(cache_key)
        
        if cached is not None:
            self.stats['cache_hits'] += 1
            return cached
        
        self.stats['cache_misses'] += 1
        
        # Try online backend
        if 'online' in self.backends:
            value = await self.backends['online'].get(namespace, key)
            if value is not None:
                # Update cache
                await self.cache.set(cache_key, value, ttl=60)  # 1 minute cache
                self.stats['online_gets'] += 1
                return value
        
        return None
    
    async def set_online(self, namespace: str, key: str, value: Any,
                        ttl: Optional[int] = None) -> bool:
        """
        Set feature in online store.
        
        Args:
            namespace: Feature namespace
            key: Feature key
            value: Feature value
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        # Update online backend
        if 'online' in self.backends:
            success = await self.backends['online'].set(namespace, key, value, ttl)
            if success:
                self.stats['online_sets'] += 1
                
                # Update cache
                cache_key = f"{namespace}:{key}"
                await self.cache.set(cache_key, value, ttl=ttl or 60)
                
                return True
        
        return False
    
    async def get_offline(self, namespace: str, key: str) -> Optional[Any]:
        """
        Get feature from offline store.
        
        Args:
            namespace: Feature namespace
            key: Feature key
            
        Returns:
            Feature value or None
        """
        if 'offline' in self.backends:
            self.stats['offline_gets'] += 1
            return await self.backends['offline'].get(namespace, key)
        return None
    
    async def set_offline(self, namespace: str, key: str, value: Any,
                         ttl: Optional[int] = None) -> bool:
        """
        Set feature in offline store.
        
        Args:
            namespace: Feature namespace
            key: Feature key
            value: Feature value
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        if 'offline' in self.backends:
            self.stats['offline_sets'] += 1
            return await self.backends['offline'].set(namespace, key, value, ttl)
        return False
    
    async def get_feature_vector(self, entity_id: str, entity_type: str,
                                feature_names: List[str]) -> FeatureVector:
        """
        Get feature vector for an entity.
        
        Args:
            entity_id: Entity identifier
            entity_type: Type of entity
            feature_names: List of features to retrieve
            
        Returns:
            Feature vector with retrieved values
        """
        features = {}
        
        # Try to get each feature
        for feature_name in feature_names:
            namespace = f"{entity_type}:features"
            key = f"{entity_id}:{feature_name}"
            
            # Try online first
            value = await self.get_online(namespace, key)
            
            # Fall back to offline
            if value is None:
                value = await self.get_offline(namespace, key)
            
            features[feature_name] = value
        
        return FeatureVector(
            entity_id=entity_id,
            entity_type=entity_type,
            features=features,
            source="retrieved"
        )
    
    async def compute_and_store(self, entity_id: str, entity_type: str,
                               entity_data: Dict[str, Any],
                               feature_names: List[str],
                               store_online: bool = True,
                               store_offline: bool = True,
                               ttl: Optional[int] = None) -> FeatureVector:
        """
        Compute features and store them.
        
        Args:
            entity_id: Entity identifier
            entity_type: Type of entity
            entity_data: Entity data for computation
            feature_names: Features to compute
            store_online: Whether to store in online store
            store_offline: Whether to store in offline store
            ttl: Time to live for online store
            
        Returns:
            Computed feature vector
        """
        # Compute features
        vector = await self.engine.compute_feature_vector(
            entity_id, entity_type, entity_data, feature_names
        )
        
        # Store features
        for feature_name, value in vector.features.items():
            if value is not None:
                namespace = f"{entity_type}:features"
                key = f"{entity_id}:{feature_name}"
                
                if store_online:
                    await self.set_online(namespace, key, value, ttl)
                
                if store_offline:
                    await self.set_offline(namespace, key, value)
        
        return vector
    
    async def get_training_data(self, entity_ids: List[str], entity_type: str,
                               feature_names: List[str],
                               start_time: Optional[datetime] = None,
                               end_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get training data for model development.
        
        Args:
            entity_ids: List of entity IDs
            entity_type: Type of entity
            feature_names: Features to retrieve
            start_time: Start time filter
            end_time: End time filter
            
        Returns:
            DataFrame with training data
        """
        if not PANDAS_AVAILABLE:
            self.logger.error("Pandas required for training data")
            return pd.DataFrame()
        
        data = []
        
        for entity_id in entity_ids:
            row = {'entity_id': entity_id}
            
            for feature_name in feature_names:
                namespace = f"{entity_type}:features"
                key = f"{entity_id}:{feature_name}"
                
                value = await self.get_offline(namespace, key)
                row[feature_name] = value
            
            data.append(row)
        
        df = pd.DataFrame(data)
        self.logger.info(f"Retrieved training data with shape {df.shape}")
        
        return df
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get feature store statistics."""
        stats = self.stats.copy()
        
        # Add computation stats
        stats.update(self.engine.stats)
        
        # Add registry stats
        stats['registered_features'] = len(self.registry.features)
        
        # Calculate hit rate
        total_cache_requests = stats['cache_hits'] + stats['cache_misses']
        if total_cache_requests > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / total_cache_requests
        else:
            stats['cache_hit_rate'] = 0
        
        return stats
    
    async def close(self):
        """Close all backend connections."""
        for backend in self.backends.values():
            if hasattr(backend, 'close'):
                await backend.close()
        
        self.logger.info("Feature store closed")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_feature_store(config_path: Optional[str] = None) -> FeatureStore:
    """
    Create and configure feature store.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured FeatureStore instance
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
            'online': {
                'type': 'redis',
                'host': 'localhost',
                'port': 6379,
                'db': 0,
                'key_prefix': 'feature_store:',
                'max_connections': 10,
                'async_enabled': True
            },
            'offline': {
                'type': 'sql',
                'connection_string': 'sqlite:///feature_store.db',
                'table_name': 'feature_store',
                'pool_size': 5,
                'max_overflow': 10
            },
            'cache': {
                'redis_enabled': False,
                'default_ttl_seconds': 60
            },
            'registry': {
                'backend': 'memory'
            },
            'computation': {
                'version': '1.0.0'
            }
        }
    
    return FeatureStore(config)


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    """
    Example usage of feature store.
    """
    
    async def example():
        # Create feature store
        store = create_feature_store()
        
        # Define features
        features = [
            FeatureDefinition(
                name="amount_log",
                feature_type=FeatureType.CONTINUOUS,
                group=FeatureGroup.TRANSACTION,
                description="Log of transaction amount",
                source="amount",
                transformation="log1p",
                validation_rules={'min': 0, 'max': 20}
            ),
            FeatureDefinition(
                name="is_weekend",
                feature_type=FeatureType.BINARY,
                group=FeatureGroup.TEMPORAL,
                description="Whether transaction is on weekend",
                source="timestamp",
                transformation="dayofweek >= 5",
                validation_rules={'allowed_values': [0, 1]}
            ),
            FeatureDefinition(
                name="tx_count_1h",
                feature_type=FeatureType.COUNT,
                group=FeatureGroup.BEHAVIORAL,
                description="Transaction count in last hour",
                source="transactions",
                transformation="rolling_count",
                validation_rules={'min': 0}
            )
        ]
        
        # Register features
        for feature_def in features:
            store.register_feature(feature_def)
        
        # Store some features
        await store.set_online(
            "customer:features",
            "cust123:amount_log",
            4.382,
            ttl=3600
        )
        
        await store.set_offline(
            "customer:features",
            "cust123:amount_log",
            4.382
        )
        
        # Retrieve features
        value = await store.get_online("customer:features", "cust123:amount_log")
        print(f"Retrieved value: {value}")
        
        # Get feature vector
        vector = await store.get_feature_vector(
            "cust123",
            "customer",
            ["amount_log", "is_weekend", "tx_count_1h"]
        )
        print(f"Feature vector: {vector.features}")
        
        # Get statistics
        stats = await store.get_stats()
        print(f"Stats: {stats}")
        
        # Close store
        await store.close()
        
        return store
    
    # Run example
    store = asyncio.run(example())