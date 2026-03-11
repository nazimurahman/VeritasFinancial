# =============================================================================
# VERITASFINANCIAL - FRAUD DETECTION SYSTEM
# Module: deployment/pipeline/realtime_processing.py
# Description: Real-time transaction processing for instant fraud detection
# Author: Data Science Team
# Version: 2.0.0
# Last Updated: 2024-01-15
# =============================================================================

"""
Real-time Processing Pipeline for Fraud Detection
===================================================
This module handles real-time transaction processing:
- Stream ingestion from Kafka/RabbitMQ
- Low-latency feature computation
- Real-time model inference
- Asynchronous processing with async/await
- WebSocket support for live updates
- Circuit breakers and rate limiting
- Connection pooling and retry logic
"""

import asyncio
import json
import time
import uuid
import pickle
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Callable, Awaitable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
from contextlib import asynccontextmanager

# Third-party imports with error handling
try:
    import aiokafka
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logging.warning("aiokafka not available")

try:
    import aio_pika
    RABBITMQ_AVAILABLE = True
except ImportError:
    RABBITMQ_AVAILABLE = False
    logging.warning("aio-pika not available")

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("redis.asyncio not available")

try:
    import numpy as np
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/realtime_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class TransactionStatus(Enum):
    """Status of transaction processing."""
    PENDING = "pending"
    PROCESSING = "processing"
    APPROVED = "approved"
    REJECTED = "rejected"
    FLAGGED = "flagged"
    ERROR = "error"


class RiskLevel(Enum):
    """Risk levels for fraud detection."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Transaction:
    """
    Represents a financial transaction for real-time processing.
    
    Attributes:
        transaction_id: Unique transaction identifier
        customer_id: Customer identifier
        amount: Transaction amount
        currency: Currency code
        merchant_id: Merchant identifier
        merchant_category: Category of merchant
        timestamp: Transaction timestamp
        device_id: Device identifier
        ip_address: IP address of transaction
        location: Location dictionary (country, city, etc.)
        card_present: Whether card was present
        recurring: Whether this is recurring payment
        metadata: Additional transaction metadata
    """
    transaction_id: str
    customer_id: str
    amount: float
    currency: str
    merchant_id: str
    merchant_category: str
    timestamp: datetime
    device_id: Optional[str] = None
    ip_address: Optional[str] = None
    location: Optional[Dict[str, str]] = None
    card_present: bool = False
    recurring: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Transaction':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class FraudPrediction:
    """
    Result of fraud prediction for a transaction.
    
    Attributes:
        transaction_id: Associated transaction ID
        fraud_probability: Probability of fraud (0-1)
        risk_level: Categorized risk level
        model_version: Version of model used
        processing_time_ms: Processing time in milliseconds
        features_used: Features used for prediction
        feature_importance: Feature importance scores
        rules_triggered: Rules that triggered
        explanation: Human-readable explanation
        timestamp: Prediction timestamp
    """
    transaction_id: str
    fraud_probability: float
    risk_level: RiskLevel
    model_version: str
    processing_time_ms: float
    features_used: List[str] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    rules_triggered: List[str] = field(default_factory=list)
    explanation: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['risk_level'] = self.risk_level.value
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FraudPrediction':
        """Create from dictionary."""
        data['risk_level'] = RiskLevel(data['risk_level'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


# =============================================================================
# STREAM CONSUMERS AND PRODUCERS
# =============================================================================

class StreamConsumer(ABC):
    """
    Abstract base class for stream consumers.
    Defines interface for consuming messages from various sources.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize stream consumer.
        
        Args:
            config: Consumer configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.running = False
        self.handlers: List[Callable] = []
    
    @abstractmethod
    async def start(self):
        """Start consuming messages."""
        pass
    
    @abstractmethod
    async def stop(self):
        """Stop consuming messages."""
        pass
    
    @abstractmethod
    async def consume(self):
        """Main consumption loop."""
        pass
    
    def register_handler(self, handler: Callable):
        """
        Register a message handler.
        
        Args:
            handler: Async function that processes messages
        """
        self.handlers.append(handler)
    
    async def _process_message(self, message: Any) -> None:
        """
        Process a message through all handlers.
        
        Args:
            message: Message to process
        """
        for handler in self.handlers:
            try:
                await handler(message)
            except Exception as e:
                self.logger.error(f"Handler error: {e}")


class KafkaConsumer(StreamConsumer):
    """
    Kafka stream consumer implementation.
    Uses aiokafka for async consumption.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Kafka consumer.
        
        Args:
            config: Configuration with bootstrap_servers, topic, etc.
        """
        super().__init__(config)
        
        if not KAFKA_AVAILABLE:
            raise ImportError("aiokafka is required for KafkaConsumer")
        
        self.bootstrap_servers = config.get('bootstrap_servers', 'localhost:9092')
        self.topic = config.get('topic', 'transactions')
        self.group_id = config.get('group_id', 'fraud-detection-group')
        self.auto_offset_reset = config.get('auto_offset_reset', 'latest')
        
        self.consumer = None
    
    async def start(self):
        """Initialize and start Kafka consumer."""
        self.consumer = AIOKafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            auto_offset_reset=self.auto_offset_reset,
            enable_auto_commit=True,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        await self.consumer.start()
        self.running = True
        self.logger.info(f"Kafka consumer started for topic: {self.topic}")
        
        # Start consumption task
        asyncio.create_task(self.consume())
    
    async def stop(self):
        """Stop Kafka consumer."""
        self.running = False
        if self.consumer:
            await self.consumer.stop()
        self.logger.info("Kafka consumer stopped")
    
    async def consume(self):
        """Main consumption loop."""
        try:
            async for msg in self.consumer:
                if not self.running:
                    break
                
                # Process message
                await self._process_message(msg.value)
                
        except Exception as e:
            self.logger.error(f"Consumption error: {e}")
            if self.running:
                # Attempt reconnect
                await asyncio.sleep(5)
                asyncio.create_task(self.consume())


class RabbitMQConsumer(StreamConsumer):
    """
    RabbitMQ stream consumer implementation.
    Uses aio-pika for async consumption.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RabbitMQ consumer.
        
        Args:
            config: Configuration with URL, queue, etc.
        """
        super().__init__(config)
        
        if not RABBITMQ_AVAILABLE:
            raise ImportError("aio-pika is required for RabbitMQConsumer")
        
        self.url = config.get('url', 'amqp://guest:guest@localhost/')
        self.queue_name = config.get('queue', 'transactions')
        self.exchange_name = config.get('exchange', '')
        self.routing_key = config.get('routing_key', '')
        
        self.connection = None
        self.channel = None
        self.queue = None
    
    async def start(self):
        """Initialize and start RabbitMQ consumer."""
        self.connection = await aio_pika.connect_robust(self.url)
        self.channel = await self.connection.channel()
        
        # Declare queue
        self.queue = await self.channel.declare_queue(
            self.queue_name,
            durable=True
        )
        
        self.running = True
        self.logger.info(f"RabbitMQ consumer started for queue: {self.queue_name}")
        
        # Start consumption
        asyncio.create_task(self.consume())
    
    async def stop(self):
        """Stop RabbitMQ consumer."""
        self.running = False
        if self.connection:
            await self.connection.close()
        self.logger.info("RabbitMQ consumer stopped")
    
    async def consume(self):
        """Main consumption loop."""
        try:
            async with self.queue.iterator() as queue_iter:
                async for message in queue_iter:
                    if not self.running:
                        break
                    
                    async with message.process():
                        # Parse message
                        data = json.loads(message.body.decode())
                        
                        # Process
                        await self._process_message(data)
                        
        except Exception as e:
            self.logger.error(f"Consumption error: {e}")
            if self.running:
                # Attempt reconnect
                await asyncio.sleep(5)
                asyncio.create_task(self.consume())


class StreamProducer(ABC):
    """
    Abstract base class for stream producers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize stream producer.
        
        Args:
            config: Producer configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def start(self):
        """Start producer."""
        pass
    
    @abstractmethod
    async def stop(self):
        """Stop producer."""
        pass
    
    @abstractmethod
    async def send(self, topic: str, key: str, value: Any) -> bool:
        """
        Send a message.
        
        Args:
            topic: Destination topic
            key: Message key
            value: Message value
            
        Returns:
            True if sent successfully
        """
        pass


class KafkaProducer(StreamProducer):
    """
    Kafka stream producer implementation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Kafka producer.
        
        Args:
            config: Configuration with bootstrap_servers, etc.
        """
        super().__init__(config)
        
        if not KAFKA_AVAILABLE:
            raise ImportError("aiokafka is required for KafkaProducer")
        
        self.bootstrap_servers = config.get('bootstrap_servers', 'localhost:9092')
        self.producer = None
    
    async def start(self):
        """Initialize and start Kafka producer."""
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: str(k).encode('utf-8')
        )
        
        await self.producer.start()
        self.logger.info("Kafka producer started")
    
    async def stop(self):
        """Stop Kafka producer."""
        if self.producer:
            await self.producer.stop()
        self.logger.info("Kafka producer stopped")
    
    async def send(self, topic: str, key: str, value: Any) -> bool:
        """
        Send message to Kafka topic.
        
        Args:
            topic: Topic name
            key: Message key (for partitioning)
            value: Message value
            
        Returns:
            True if sent successfully
        """
        try:
            await self.producer.send(topic, key=key, value=value)
            return True
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            return False


# =============================================================================
# FEATURE CACHE AND STORE
# =============================================================================

class FeatureCache:
    """
    In-memory cache for real-time features.
    Uses Redis for distributed caching with local fallback.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature cache.
        
        Args:
            config: Cache configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.FeatureCache")
        
        # Local cache (for development/fallback)
        self.local_cache: Dict[str, Any] = {}
        self.local_expiry: Dict[str, datetime] = {}
        self.default_ttl = config.get('default_ttl_seconds', 300)
        
        # Redis client
        self.redis_client = None
        if REDIS_AVAILABLE and config.get('redis_enabled', False):
            self.redis_client = redis.from_url(
                config.get('redis_url', 'redis://localhost:6379'),
                decode_responses=True
            )
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        # Try Redis first
        if self.redis_client:
            try:
                value = await self.redis_client.get(key)
                if value:
                    return json.loads(value)
            except Exception as e:
                self.logger.warning(f"Redis get failed: {e}")
        
        # Fallback to local cache
        if key in self.local_cache:
            # Check expiry
            if key in self.local_expiry:
                if datetime.now() > self.local_expiry[key]:
                    del self.local_cache[key]
                    del self.local_expiry[key]
                    return None
            
            return self.local_cache[key]
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        ttl = ttl or self.default_ttl
        
        # Try Redis first
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    key,
                    ttl,
                    json.dumps(value, default=str)
                )
                return True
            except Exception as e:
                self.logger.warning(f"Redis set failed: {e}")
        
        # Fallback to local cache
        self.local_cache[key] = value
        self.local_expiry[key] = datetime.now() + timedelta(seconds=ttl)
        
        return True
    
    async def delete(self, key: str) -> bool:
        """
        Delete from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted
        """
        if self.redis_client:
            try:
                await self.redis_client.delete(key)
            except:
                pass
        
        if key in self.local_cache:
            del self.local_cache[key]
        if key in self.local_expiry:
            del self.local_expiry[key]
        
        return True
    
    async def increment(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int:
        """
        Increment a counter in cache.
        
        Args:
            key: Counter key
            amount: Amount to increment by
            ttl: Time to live
            
        Returns:
            New counter value
        """
        # Try Redis first
        if self.redis_client:
            try:
                value = await self.redis_client.incrby(key, amount)
                if ttl:
                    await self.redis_client.expire(key, ttl)
                return value
            except Exception as e:
                self.logger.warning(f"Redis increment failed: {e}")
        
        # Fallback to local cache
        current = await self.get(key) or 0
        new_value = current + amount
        await self.set(key, new_value, ttl)
        
        return new_value


class FeatureStore:
    """
    Feature store for real-time feature computation.
    Maintains customer profiles, transaction histories, and aggregated features.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature store.
        
        Args:
            config: Feature store configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.FeatureStore")
        
        # Cache
        self.cache = FeatureCache(config.get('cache', {}))
        
        # In-memory stores for real-time features
        self.customer_profiles: Dict[str, Dict[str, Any]] = {}
        self.recent_transactions: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=config.get('recent_transactions_per_customer', 100))
        )
        self.device_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=config.get('device_history_size', 50))
        )
        
        # Statistics
        self.stats = {
            'features_computed': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    async def get_customer_features(self, customer_id: str) -> Dict[str, Any]:
        """
        Get customer profile features.
        
        Args:
            customer_id: Customer identifier
            
        Returns:
            Dictionary of customer features
        """
        # Try cache first
        cache_key = f"customer:{customer_id}"
        cached = await self.cache.get(cache_key)
        
        if cached:
            self.stats['cache_hits'] += 1
            return cached
        
        self.stats['cache_misses'] += 1
        
        # Get from memory or load from database
        if customer_id in self.customer_profiles:
            features = self.customer_profiles[customer_id]
        else:
            # In production, load from database
            features = await self._load_customer_profile(customer_id)
            self.customer_profiles[customer_id] = features
        
        # Cache for future
        await self.cache.set(cache_key, features, ttl=300)  # 5 minutes
        
        return features
    
    async def compute_transaction_features(self, transaction: Transaction) -> Dict[str, Any]:
        """
        Compute real-time features for a transaction.
        
        Args:
            transaction: Transaction to compute features for
            
        Returns:
            Dictionary of computed features
        """
        features = {}
        start_time = time.time()
        
        # Basic features
        features['amount'] = transaction.amount
        features['currency'] = transaction.currency
        features['merchant_category'] = transaction.merchant_category
        features['card_present'] = int(transaction.card_present)
        features['recurring'] = int(transaction.recurring)
        
        # Time-based features
        hour = transaction.timestamp.hour
        features['hour_of_day'] = hour
        features['day_of_week'] = transaction.timestamp.weekday()
        features['is_weekend'] = int(features['day_of_week'] >= 5)
        
        # Cyclical encoding
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # Customer features
        customer_features = await self.get_customer_features(transaction.customer_id)
        features.update({
            f"customer_{k}": v for k, v in customer_features.items()
            if k not in ['customer_id', 'name']  # Exclude identifiers
        })
        
        # Transaction velocity features
        velocity_features = await self._compute_velocity_features(transaction)
        features.update(velocity_features)
        
        # Merchant features
        merchant_features = await self._get_merchant_features(transaction.merchant_id)
        features.update(merchant_features)
        
        # Device features
        if transaction.device_id:
            device_features = await self._get_device_features(transaction.device_id)
            features.update(device_features)
        
        # Location features
        if transaction.location:
            location_features = await self._compute_location_features(transaction)
            features.update(location_features)
        
        # Update statistics
        self.stats['features_computed'] += 1
        features['feature_computation_time_ms'] = (time.time() - start_time) * 1000
        
        # Store transaction in history
        self.recent_transactions[transaction.customer_id].append(transaction)
        
        return features
    
    async def _compute_velocity_features(self, transaction: Transaction) -> Dict[str, Any]:
        """
        Compute transaction velocity features.
        
        Args:
            transaction: Current transaction
            
        Returns:
            Velocity features dictionary
        """
        features = {}
        customer_id = transaction.customer_id
        now = transaction.timestamp
        
        # Get recent transactions
        recent = list(self.recent_transactions.get(customer_id, []))
        
        if recent:
            # Time since last transaction
            last_tx = recent[-1]
            time_gap = (now - last_tx.timestamp).total_seconds()
            features['time_since_last_tx_seconds'] = time_gap
            
            # Transaction counts in windows
            windows = {
                '1h': timedelta(hours=1),
                '24h': timedelta(hours=24),
                '7d': timedelta(days=7)
            }
            
            for window_name, window_delta in windows.items():
                window_start = now - window_delta
                tx_in_window = [tx for tx in recent if tx.timestamp >= window_start]
                
                features[f'tx_count_{window_name}'] = len(tx_in_window)
                features[f'tx_amount_sum_{window_name}'] = sum(tx.amount for tx in tx_in_window)
                
                if tx_in_window:
                    features[f'tx_amount_avg_{window_name}'] = np.mean([tx.amount for tx in tx_in_window])
                    features[f'tx_amount_std_{window_name}'] = np.std([tx.amount for tx in tx_in_window]) if len(tx_in_window) > 1 else 0
                else:
                    features[f'tx_amount_avg_{window_name}'] = 0
                    features[f'tx_amount_std_{window_name}'] = 0
            
            # Amount deviation
            if features.get('tx_amount_avg_24h', 0) > 0:
                features['amount_deviation_from_avg'] = transaction.amount / features['tx_amount_avg_24h']
            else:
                features['amount_deviation_from_avg'] = 1.0
            
            # Velocity of transactions (tx per hour)
            if time_gap > 0:
                features['tx_velocity_per_hour'] = 3600 / time_gap
            else:
                features['tx_velocity_per_hour'] = float('inf')
        else:
            # First transaction for customer
            features['time_since_last_tx_seconds'] = -1
            features['tx_count_1h'] = 0
            features['tx_count_24h'] = 0
            features['tx_count_7d'] = 0
            features['amount_deviation_from_avg'] = 1.0
            features['tx_velocity_per_hour'] = 0
        
        return features
    
    async def _get_merchant_features(self, merchant_id: str) -> Dict[str, Any]:
        """
        Get merchant risk features.
        
        Args:
            merchant_id: Merchant identifier
            
        Returns:
            Merchant features dictionary
        """
        # Try cache first
        cache_key = f"merchant:{merchant_id}"
        cached = await self.cache.get(cache_key)
        
        if cached:
            return cached
        
        # In production, load from database
        # For now, return default features
        features = {
            'merchant_risk_score': 0.1,  # Default low risk
            'merchant_fraud_rate': 0.01,
            'merchant_avg_amount': 100.0,
            'merchant_transaction_count': 1000
        }
        
        await self.cache.set(cache_key, features, ttl=3600)  # 1 hour cache
        
        return features
    
    async def _get_device_features(self, device_id: str) -> Dict[str, Any]:
        """
        Get device fingerprint features.
        
        Args:
            device_id: Device identifier
            
        Returns:
            Device features dictionary
        """
        features = {
            'device_risk_score': 0.1,  # Default low risk
            'device_age_days': 30,
            'transactions_on_device': len(self.device_history.get(device_id, [])),
            'is_new_device': device_id not in self.device_history
        }
        
        return features
    
    async def _compute_location_features(self, transaction: Transaction) -> Dict[str, Any]:
        """
        Compute location-based features.
        
        Args:
            transaction: Current transaction
            
        Returns:
            Location features dictionary
        """
        features = {}
        
        if transaction.location:
            country = transaction.location.get('country', 'unknown')
            
            # Country risk score (would come from external source)
            country_risk = {
                'US': 0.1,
                'GB': 0.1,
                'CA': 0.1,
                'unknown': 0.3
            }
            features['location_country_risk'] = country_risk.get(country, 0.3)
            
            # Distance from home (would need home location)
            features['distance_from_home_km'] = 0  # Placeholder
            
            # Is foreign transaction
            customer_profile = await self.get_customer_features(transaction.customer_id)
            home_country = customer_profile.get('home_country', 'unknown')
            features['is_foreign_transaction'] = int(country != home_country)
        
        return features
    
    async def _load_customer_profile(self, customer_id: str) -> Dict[str, Any]:
        """
        Load customer profile from database.
        
        Args:
            customer_id: Customer identifier
            
        Returns:
            Customer profile dictionary
        """
        # In production, this would query a database
        # For now, return default profile
        return {
            'customer_id': customer_id,
            'account_age_days': 365,
            'avg_transaction_amount': 75.0,
            'std_transaction_amount': 50.0,
            'transaction_count_total': 100,
            'home_country': 'US',
            'risk_tier': 'standard',
            'credit_score': 700,
            'income_level': 'medium'
        }


# =============================================================================
# MODEL INFERENCE ENGINE
# =============================================================================

class InferenceEngine:
    """
    Real-time model inference engine.
    Supports multiple models with versioning and A/B testing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize inference engine.
        
        Args:
            config: Inference configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.InferenceEngine")
        
        # Models
        self.models: Dict[str, Any] = {}
        self.model_versions: Dict[str, str] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Feature store
        self.feature_store = FeatureStore(config.get('feature_store', {}))
        
        # Load models
        self._load_models()
        
        # Metrics
        self.metrics = {
            'predictions': 0,
            'avg_latency_ms': 0,
            'total_latency_ms': 0
        }
    
    def _load_models(self):
        """Load models from disk."""
        model_paths = self.config.get('model_paths', {})
        
        for model_name, model_path in model_paths.items():
            try:
                with open(model_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                
                # Store version info
                self.model_versions[model_name] = self.config.get(f'version_{model_name}', '1.0.0')
                self.model_metadata[model_name] = {
                    'path': model_path,
                    'version': self.model_versions[model_name],
                    'features': self.config.get(f'features_{model_name}', []),
                    'threshold': self.config.get(f'threshold_{model_name}', 0.5)
                }
                
                self.logger.info(f"Loaded model: {model_name} v{self.model_versions[model_name]}")
                
            except Exception as e:
                self.logger.error(f"Failed to load model {model_name}: {e}")
    
    async def predict(self, transaction: Transaction, 
                     model_name: str = 'default') -> FraudPrediction:
        """
        Make real-time prediction for a transaction.
        
        Args:
            transaction: Transaction to predict
            model_name: Name of model to use
            
        Returns:
            FraudPrediction object
        """
        start_time = time.time()
        
        # Get model
        if model_name not in self.models:
            self.logger.warning(f"Model {model_name} not found, using default")
            model_name = 'default'
        
        model = self.models.get(model_name)
        metadata = self.model_metadata.get(model_name, {})
        
        if not model:
            raise ValueError(f"No model available: {model_name}")
        
        # Compute features
        features = await self.feature_store.compute_transaction_features(transaction)
        
        # Prepare feature vector
        feature_names = metadata.get('features', list(features.keys()))
        feature_vector = []
        
        # Ensure all required features exist
        for feature_name in feature_names:
            if feature_name in features:
                feature_vector.append(features[feature_name])
            else:
                feature_vector.append(0.0)  # Default value
        
        feature_array = np.array(feature_vector).reshape(1, -1)
        
        # Make prediction
        try:
            if hasattr(model, 'predict_proba'):
                fraud_probability = model.predict_proba(feature_array)[0, 1]
            else:
                fraud_probability = model.predict(feature_array)[0]
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            fraud_probability = 0.0
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Determine risk level
        risk_level = self._get_risk_level(fraud_probability, metadata.get('threshold', 0.5))
        
        # Get feature importance (if available)
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            for i, name in enumerate(feature_names[:len(model.feature_importances_)]):
                feature_importance[name] = float(model.feature_importances_[i])
        
        # Update metrics
        self.metrics['predictions'] += 1
        self.metrics['total_latency_ms'] += processing_time_ms
        self.metrics['avg_latency_ms'] = (
            self.metrics['total_latency_ms'] / self.metrics['predictions']
        )
        
        # Create prediction
        prediction = FraudPrediction(
            transaction_id=transaction.transaction_id,
            fraud_probability=float(fraud_probability),
            risk_level=risk_level,
            model_version=metadata.get('version', 'unknown'),
            processing_time_ms=processing_time_ms,
            features_used=feature_names,
            feature_importance=feature_importance,
            explanation=self._generate_explanation(features, feature_importance, fraud_probability)
        )
        
        return prediction
    
    def _get_risk_level(self, probability: float, threshold: float) -> RiskLevel:
        """
        Convert probability to risk level.
        
        Args:
            probability: Fraud probability
            threshold: Decision threshold
            
        Returns:
            RiskLevel enum
        """
        if probability < 0.3:
            return RiskLevel.LOW
        elif probability < 0.6:
            return RiskLevel.MEDIUM
        elif probability < 0.8:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _generate_explanation(self, features: Dict[str, Any], 
                             importance: Dict[str, float],
                             probability: float) -> str:
        """
        Generate human-readable explanation.
        
        Args:
            features: Feature dictionary
            importance: Feature importance scores
            probability: Fraud probability
            
        Returns:
            Explanation string
        """
        explanations = []
        
        if probability > 0.7:
            explanations.append(f"High fraud probability: {probability:.2%}")
            
            # Find top contributing factors
            if importance:
                top_factors = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
                
                for factor, score in top_factors:
                    if factor in features:
                        explanations.append(f"- {factor}: {features[factor]:.2f} (importance: {score:.3f})")
        
        return "\n".join(explanations) if explanations else "Normal transaction pattern"


# =============================================================================
# REAL-TIME PROCESSOR
# =============================================================================

class RealtimeProcessor:
    """
    Main real-time processing pipeline.
    Coordinates stream consumption, feature computation, inference, and output.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize real-time processor.
        
        Args:
            config: Processor configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.RealtimeProcessor")
        
        # Components
        self.consumer = None
        self.producer = None
        self.inference_engine = InferenceEngine(config.get('inference', {}))
        
        # Configuration
        self.input_topic = config.get('input_topic', 'transactions')
        self.output_topic = config.get('output_topic', 'fraud_predictions')
        self.alert_topic = config.get('alert_topic', 'fraud_alerts')
        
        # Rate limiting
        self.max_processing_rate = config.get('max_processing_rate', 1000)  # tx per second
        self.rate_limiter = RateLimiter(self.max_processing_rate)
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.get('circuit_breaker_threshold', 10),
            recovery_timeout=config.get('circuit_breaker_timeout', 30)
        )
        
        # Statistics
        self.stats = {
            'transactions_processed': 0,
            'fraud_detected': 0,
            'errors': 0,
            'start_time': datetime.now(),
            'processing_rate': 0
        }
        
        # Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            self.prom_metrics = {
                'transactions': Counter('realtime_transactions_total', 'Total transactions processed'),
                'fraud': Counter('realtime_fraud_total', 'Total fraud detected'),
                'latency': Histogram('realtime_latency_seconds', 'Processing latency'),
                'errors': Counter('realtime_errors_total', 'Total errors')
            }
    
    async def start(self):
        """Start the real-time processor."""
        # Initialize consumer
        consumer_type = self.config.get('consumer_type', 'kafka')
        if consumer_type == 'kafka':
            self.consumer = KafkaConsumer(self.config.get('consumer', {}))
        elif consumer_type == 'rabbitmq':
            self.consumer = RabbitMQConsumer(self.config.get('consumer', {}))
        else:
            raise ValueError(f"Unknown consumer type: {consumer_type}")
        
        # Initialize producer
        producer_type = self.config.get('producer_type', 'kafka')
        if producer_type == 'kafka':
            self.producer = KafkaProducer(self.config.get('producer', {}))
        else:
            raise ValueError(f"Unknown producer type: {producer_type}")
        
        # Register handler
        self.consumer.register_handler(self.process_transaction)
        
        # Start components
        await self.consumer.start()
        await self.producer.start()
        
        self.logger.info("Real-time processor started")
    
    async def stop(self):
        """Stop the real-time processor."""
        if self.consumer:
            await self.consumer.stop()
        if self.producer:
            await self.producer.stop()
        
        self.logger.info("Real-time processor stopped")
    
    async def process_transaction(self, message: Dict[str, Any]) -> None:
        """
        Process a single transaction message.
        
        Args:
            message: Transaction message from stream
        """
        start_time = time.time()
        
        try:
            # Apply rate limiting
            await self.rate_limiter.acquire()
            
            # Check circuit breaker
            if not self.circuit_breaker.allow_request():
                self.logger.warning("Circuit breaker open, skipping processing")
                return
            
            # Parse transaction
            transaction = Transaction.from_dict(message)
            
            self.logger.debug(f"Processing transaction: {transaction.transaction_id}")
            
            # Get prediction
            prediction = await self.inference_engine.predict(transaction)
            
            # Take action based on risk level
            await self._handle_prediction(transaction, prediction)
            
            # Send result to output topic
            await self.producer.send(
                self.output_topic,
                key=transaction.transaction_id,
                value=prediction.to_dict()
            )
            
            # Update statistics
            self.stats['transactions_processed'] += 1
            
            if prediction.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                self.stats['fraud_detected'] += 1
                if PROMETHEUS_AVAILABLE:
                    self.prom_metrics['fraud'].inc()
            
            # Update processing rate
            elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
            if elapsed > 0:
                self.stats['processing_rate'] = self.stats['transactions_processed'] / elapsed
            
            # Record metrics
            processing_time = time.time() - start_time
            if PROMETHEUS_AVAILABLE:
                self.prom_metrics['transactions'].inc()
                self.prom_metrics['latency'].observe(processing_time)
            
            self.logger.debug(
                f"Processed {transaction.transaction_id} in {processing_time*1000:.2f}ms, "
                f"risk: {prediction.risk_level.value}"
            )
            
        except Exception as e:
            self.stats['errors'] += 1
            self.circuit_breaker.record_failure()
            
            if PROMETHEUS_AVAILABLE:
                self.prom_metrics['errors'].inc()
            
            self.logger.error(f"Error processing transaction: {e}", exc_info=True)
    
    async def _handle_prediction(self, transaction: Transaction, 
                                 prediction: FraudPrediction) -> None:
        """
        Handle fraud prediction based on risk level.
        
        Args:
            transaction: Original transaction
            prediction: Fraud prediction
        """
        # Log high-risk transactions
        if prediction.risk_level == RiskLevel.CRITICAL:
            self.logger.warning(
                f"CRITICAL FRAUD: {transaction.transaction_id} - "
                f"probability: {prediction.fraud_probability:.2%}"
            )
            
            # Send to alert topic
            alert = {
                'transaction': transaction.to_dict(),
                'prediction': prediction.to_dict(),
                'alert_type': 'critical_fraud',
                'timestamp': datetime.now().isoformat()
            }
            
            await self.producer.send(
                self.alert_topic,
                key=transaction.transaction_id,
                value=alert
            )
            
        elif prediction.risk_level == RiskLevel.HIGH:
            self.logger.info(
                f"High risk transaction: {transaction.transaction_id} - "
                f"probability: {prediction.fraud_probability:.2%}"
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.stats.copy()
        stats['uptime_seconds'] = (datetime.now() - self.stats['start_time']).total_seconds()
        return stats


# =============================================================================
# UTILITY CLASSES
# =============================================================================

class RateLimiter:
    """
    Token bucket rate limiter for controlling processing rate.
    """
    
    def __init__(self, rate: float):
        """
        Initialize rate limiter.
        
        Args:
            rate: Maximum requests per second
        """
        self.rate = rate
        self.tokens = rate
        self.last_refill = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire tokens for processing.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if acquired
        """
        async with self.lock:
            # Refill tokens
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
            self.last_refill = now
            
            # Check if enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            # Not enough tokens, wait
            wait_time = (tokens - self.tokens) / self.rate
            await asyncio.sleep(wait_time)
            
            # Refill after wait
            self.tokens = min(self.rate, self.tokens + wait_time * self.rate)
            self.tokens -= tokens
            
            return True


class CircuitBreaker:
    """
    Circuit breaker pattern for fault tolerance.
    Prevents cascading failures by opening circuit when errors exceed threshold.
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 30):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying to close circuit
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        
        self.failure_count = 0
        self.state = 'closed'  # closed, open, half-open
        self.last_failure_time = None
        self.lock = asyncio.Lock()
    
    def allow_request(self) -> bool:
        """
        Check if request is allowed.
        
        Returns:
            True if request can proceed
        """
        if self.state == 'closed':
            return True
        
        elif self.state == 'open':
            # Check if recovery timeout elapsed
            if self.last_failure_time:
                elapsed = time.time() - self.last_failure_time
                if elapsed > self.recovery_timeout:
                    self.state = 'half-open'
                    return True
            return False
        
        elif self.state == 'half-open':
            # Allow one test request
            return True
        
        return False
    
    def record_failure(self):
        """Record a failure."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'open'
    
    def record_success(self):
        """Record a success (resets circuit)."""
        if self.state == 'half-open':
            self.state = 'closed'
            self.failure_count = 0


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_realtime_processor(config_path: Optional[str] = None) -> RealtimeProcessor:
    """
    Create and configure real-time processor.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured RealtimeProcessor instance
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
            'consumer_type': 'kafka',
            'producer_type': 'kafka',
            'input_topic': 'transactions',
            'output_topic': 'fraud_predictions',
            'alert_topic': 'fraud_alerts',
            'max_processing_rate': 1000,
            'circuit_breaker_threshold': 10,
            'circuit_breaker_timeout': 30,
            
            'consumer': {
                'bootstrap_servers': 'localhost:9092',
                'topic': 'transactions',
                'group_id': 'fraud-detection-group'
            },
            
            'producer': {
                'bootstrap_servers': 'localhost:9092'
            },
            
            'inference': {
                'model_paths': {
                    'default': 'artifacts/models/fraud_model.pkl'
                },
                'version_default': '1.0.0',
                'features_default': [
                    'amount', 'hour_of_day', 'day_of_week',
                    'tx_count_1h', 'tx_count_24h',
                    'amount_deviation_from_avg'
                ],
                'threshold_default': 0.5,
                
                'feature_store': {
                    'cache': {
                        'redis_enabled': False,
                        'default_ttl_seconds': 300
                    },
                    'recent_transactions_per_customer': 100
                }
            }
        }
    
    return RealtimeProcessor(config)


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    """
    Example usage of real-time processing pipeline.
    """
    
    async def main():
        # Create processor
        processor = create_realtime_processor()
        
        # Start processor
        await processor.start()
        
        # Run for a while
        try:
            while True:
                # Print stats every 10 seconds
                await asyncio.sleep(10)
                stats = processor.get_stats()
                logger.info(f"Stats: {stats}")
                
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        
        finally:
            await processor.stop()
    
    # Run the async main function
    asyncio.run(main())