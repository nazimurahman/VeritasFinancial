"""
Stream Consumers Module for VeritasFinancial Banking Fraud Detection System

This module handles real-time data ingestion from various banking data streams including:
- Transaction streams from core banking systems
- Customer profile updates
- Device fingerprinting data
- External risk intelligence feeds

Author: VeritasFinancial Data Science Team
Version: 1.0.0
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Generator, Union
from dataclasses import dataclass, asdict
import asyncio
from collections import deque
import hashlib
import uuid

import pandas as pd
import numpy as np
from kafka import KafkaConsumer, KafkaProducer, TopicPartition
from kafka.errors import KafkaError, NoBrokersAvailable
import redis
import aiokafka
from confluent_kafka import Consumer, Producer, KafkaException
import pyarrow as pa
import pyarrow.parquet as pq
import avro.schema
import avro.io
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TransactionMessage:
    """
    Data class representing a raw transaction message from banking stream.
    
    This class defines the structure of incoming transaction data with
    proper type hints and validation rules. It serves as the canonical
    data model for all incoming transaction events.
    
    Attributes:
        transaction_id: Unique identifier for the transaction
        account_id: Customer account identifier
        amount: Transaction amount (positive for credit, negative for debit)
        currency: Three-letter currency code (ISO 4217)
        transaction_type: Type of transaction (PURCHASE, WITHDRAWAL, TRANSFER, etc.)
        merchant_id: Merchant identifier (if applicable)
        merchant_category: MCC code or category
        timestamp: Transaction timestamp (ISO format)
        location: Dictionary containing location data (country, city, coordinates)
        device_id: Unique device identifier
        ip_address: IP address of the transaction origin
        card_present: Boolean indicating if card was physically present
        is_international: Boolean indicating cross-border transaction
        channel: Transaction channel (ONLINE, POS, ATM, MOBILE)
        metadata: Additional transaction metadata (free-form dictionary)
    """
    transaction_id: str
    account_id: str
    amount: float
    currency: str
    transaction_type: str
    merchant_id: Optional[str] = None
    merchant_category: Optional[str] = None
    timestamp: Optional[str] = None
    location: Optional[Dict[str, Any]] = None
    device_id: Optional[str] = None
    ip_address: Optional[str] = None
    card_present: bool = False
    is_international: bool = False
    channel: str = "UNKNOWN"
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """
        Post-initialization validation and data cleaning.
        
        This method runs automatically after __init__ to ensure data quality
        and set default values for missing fields.
        """
        # Set default timestamp if not provided
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
        
        # Ensure amount is float and properly signed
        self.amount = float(self.amount)
        
        # Set default metadata if None
        if self.metadata is None:
            self.metadata = {}
        
        # Validate required fields
        self._validate_message()
    
    def _validate_message(self):
        """
        Validate transaction message fields against business rules.
        
        Raises:
            ValueError: If any validation rule is violated
        """
        # Transaction ID must be present and valid
        if not self.transaction_id or len(self.transaction_id) < 5:
            raise ValueError(f"Invalid transaction ID: {self.transaction_id}")
        
        # Account ID must be present
        if not self.account_id:
            raise ValueError("Account ID cannot be empty")
        
        # Amount must be non-zero and within reasonable limits
        if abs(self.amount) <= 0.01:
            raise ValueError(f"Transaction amount too small: {self.amount}")
        if abs(self.amount) > 1000000:
            logger.warning(f"Large transaction amount detected: {self.amount}")
        
        # Currency must be valid ISO code
        valid_currencies = {'USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'CNY'}
        if self.currency not in valid_currencies:
            logger.warning(f"Unusual currency detected: {self.currency}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to dictionary for serialization."""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransactionMessage':
        """Create TransactionMessage instance from dictionary."""
        return cls(**data)


@dataclass
class CustomerProfileMessage:
    """
    Data class for customer profile updates from CRM systems.
    
    Attributes:
        customer_id: Unique customer identifier
        account_id: Primary account identifier
        name: Customer name (encrypted/hashed for privacy)
        email: Email address (encrypted/hashed)
        phone: Phone number (encrypted/hashed)
        date_of_birth: Customer date of birth
        country_of_residence: Country code
        risk_rating: Internal risk rating (1-10)
        account_age_days: Number of days since account opening
        total_balance: Total account balance
        average_balance: Average balance over last 30 days
        transaction_volume_30d: Number of transactions in last 30 days
        kyc_status: KYC verification status
        kyc_completed_date: Date of last KYC completion
        watchlist_flag: Boolean indicating if on any watchlist
        fraud_history: List of previous fraud incidents
        device_history: List of known devices
        last_updated: Profile last update timestamp
    """
    customer_id: str
    account_id: str
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    date_of_birth: Optional[str] = None
    country_of_residence: Optional[str] = None
    risk_rating: int = 5
    account_age_days: int = 0
    total_balance: float = 0.0
    average_balance: float = 0.0
    transaction_volume_30d: int = 0
    kyc_status: str = "PENDING"
    kyc_completed_date: Optional[str] = None
    watchlist_flag: bool = False
    fraud_history: Optional[List[Dict[str, Any]]] = None
    device_history: Optional[List[str]] = None
    last_updated: Optional[str] = None


@dataclass
class DeviceFingerprintMessage:
    """
    Data class for device fingerprinting data.
    
    Attributes:
        device_id: Unique device identifier
        device_type: Type of device (MOBILE, TABLET, DESKTOP)
        os_name: Operating system name
        os_version: Operating system version
        browser_name: Browser name (if applicable)
        browser_version: Browser version
        screen_resolution: Screen resolution
        timezone: Device timezone
        language: Device language setting
        ip_address: Last known IP address
        location_history: List of locations from this device
        first_seen: First time this device was seen
        last_seen: Last time this device was seen
        fraud_score: Calculated fraud risk for this device
        associated_accounts: List of accounts using this device
    """
    device_id: str
    device_type: str
    os_name: Optional[str] = None
    os_version: Optional[str] = None
    browser_name: Optional[str] = None
    browser_version: Optional[str] = None
    screen_resolution: Optional[str] = None
    timezone: Optional[str] = None
    language: Optional[str] = None
    ip_address: Optional[str] = None
    location_history: Optional[List[Dict[str, Any]]] = None
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    fraud_score: float = 0.0
    associated_accounts: Optional[List[str]] = None


class KafkaStreamConsumer:
    """
    Kafka-based stream consumer for real-time banking data ingestion.
    
    This class handles connection to Kafka clusters, message consumption,
    deserialization, validation, and routing to appropriate processing pipelines.
    
    Features:
    - Automatic reconnection with exponential backoff
    - Message validation and schema enforcement
    - Dead letter queue for failed messages
    - Partition assignment and rebalancing
    - Offset management (automatic/manual)
    - Metrics collection for monitoring
    - Batch processing support
    - Error handling with retry logic
    """
    
    def __init__(
        self,
        bootstrap_servers: List[str],
        group_id: str,
        topics: List[str],
        auto_offset_reset: str = 'earliest',
        enable_auto_commit: bool = True,
        max_poll_records: int = 500,
        max_poll_interval_ms: int = 300000,
        session_timeout_ms: int = 45000,
        heartbeat_interval_ms: int = 3000,
        security_protocol: str = 'PLAINTEXT',
        sasl_mechanism: Optional[str] = None,
        sasl_plain_username: Optional[str] = None,
        sasl_plain_password: Optional[str] = None,
        redis_host: str = 'localhost',
        redis_port: int = 6379,
        redis_db: int = 0,
        dead_letter_topic: str = 'fraud-detection-dlq',
        metrics_topic: str = 'fraud-detection-metrics'
    ):
        """
        Initialize Kafka consumer with comprehensive configuration.
        
        Args:
            bootstrap_servers: List of Kafka broker addresses
            group_id: Consumer group ID for coordination
            topics: List of topics to subscribe to
            auto_offset_reset: Where to start if no offset exists ('earliest'/'latest')
            enable_auto_commit: Whether to auto-commit offsets
            max_poll_records: Maximum records per poll
            max_poll_interval_ms: Maximum time between polls
            session_timeout_ms: Session timeout for consumer group
            heartbeat_interval_ms: Heartbeat interval to broker
            security_protocol: Security protocol (PLAINTEXT, SSL, SASL_SSL)
            sasl_mechanism: SASL mechanism (PLAIN, SCRAM-SHA-256, etc.)
            sasl_plain_username: SASL username if using authentication
            sasl_plain_password: SASL password if using authentication
            redis_host: Redis host for state management
            redis_port: Redis port
            redis_db: Redis database number
            dead_letter_topic: Topic for failed messages
            metrics_topic: Topic for consumer metrics
        """
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.topics = topics
        self.dead_letter_topic = dead_letter_topic
        self.metrics_topic = metrics_topic
        
        # Connection state tracking
        self.is_running = False
        self.consumer = None
        self.producer = None
        self.redis_client = None
        
        # Metrics and monitoring
        self.metrics = {
            'messages_consumed': 0,
            'messages_processed': 0,
            'messages_failed': 0,
            'messages_dlq': 0,
            'last_poll_time': None,
            'total_processing_time': 0,
            'average_processing_time': 0,
            'errors_by_type': {},
            'consumer_lag': {}
        }
        
        # Processing state
        self.processing_stats = deque(maxlen=1000)  # Keep last 1000 processing times
        self.partition_assignments = {}
        
        # Build Kafka consumer configuration
        self.consumer_config = {
            'bootstrap_servers': bootstrap_servers,
            'group_id': group_id,
            'auto_offset_reset': auto_offset_reset,
            'enable_auto_commit': enable_auto_commit,
            'max_poll_records': max_poll_records,
            'max_poll_interval_ms': max_poll_interval_ms,
            'session_timeout_ms': session_timeout_ms,
            'heartbeat_interval_ms': heartbeat_interval_ms,
            'security_protocol': security_protocol,
            # Add error handling and retry configuration
            'retry_backoff_ms': 100,
            'reconnect_backoff_ms': 50,
            'reconnect_backoff_max_ms': 1000,
            'request_timeout_ms': 305000,  # Slightly less than session timeout
            # Enable idempotence for exactly-once semantics
            'enable_idempotence': True,
            'max_in_flight_requests_per_connection': 5,
            # Compression for efficient network usage
            'compression_type': 'snappy'
        }
        
        # Add SASL authentication if configured
        if sasl_mechanism and sasl_plain_username and sasl_plain_password:
            self.consumer_config.update({
                'sasl_mechanism': sasl_mechanism,
                'sasl_plain_username': sasl_plain_username,
                'sasl_plain_password': sasl_plain_password
            })
        
        # Initialize Redis client for state management
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Connected to Redis successfully")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def connect(self) -> bool:
        """
        Establish connections to Kafka broker and initialize consumer/producer.
        
        Returns:
            bool: True if connection successful, False otherwise
            
        Raises:
            NoBrokersAvailable: If unable to connect to any Kafka broker
        """
        try:
            # Initialize Kafka consumer
            logger.info(f"Connecting to Kafka brokers: {self.bootstrap_servers}")
            self.consumer = KafkaConsumer(
                *self.topics,
                **self.consumer_config,
                # Add custom deserializers
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda m: m.decode('utf-8') if m else None
            )
            
            # Initialize Kafka producer for DLQ and metrics
            producer_config = {
                'bootstrap_servers': self.bootstrap_servers,
                'security_protocol': self.consumer_config['security_protocol'],
                'acks': 'all',  # Wait for all replicas to acknowledge
                'retries': 10,  # Retry on failure
                'max_in_flight_requests_per_connection': 5,
                'compression_type': 'snappy',
                'linger_ms': 10,  # Batch messages for efficiency
                'batch_size': 16384  # 16KB batches
            }
            
            if sasl_mechanism:
                producer_config.update({
                    'sasl_mechanism': sasl_mechanism,
                    'sasl_plain_username': sasl_plain_username,
                    'sasl_plain_password': sasl_plain_password
                })
            
            self.producer = KafkaProducer(
                **producer_config,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None
            )
            
            # Get initial partition assignment
            self._update_partition_assignment()
            
            # Start metrics collection thread
            self._start_metrics_collection()
            
            logger.info(f"Successfully connected to Kafka and subscribed to topics: {self.topics}")
            return True
            
        except NoBrokersAvailable as e:
            logger.error(f"No Kafka brokers available at {self.bootstrap_servers}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to Kafka: {e}")
            return False
    
    def _update_partition_assignment(self):
        """Update partition assignment information for monitoring."""
        if self.consumer:
            try:
                # Get current assignment
                assignment = self.consumer.assignment()
                for tp in assignment:
                    # Get end offsets for lag calculation
                    end_offsets = self.consumer.end_offsets([tp])
                    position = self.consumer.position(tp)
                    
                    self.partition_assignments[f"{tp.topic}-{tp.partition}"] = {
                        'topic': tp.topic,
                        'partition': tp.partition,
                        'current_offset': position,
                        'end_offset': end_offsets.get(tp, 0),
                        'lag': end_offsets.get(tp, 0) - position
                    }
            except Exception as e:
                logger.error(f"Error updating partition assignment: {e}")
    
    def _start_metrics_collection(self):
        """Start background thread for metrics collection and reporting."""
        import threading
        
        def metrics_collector():
            """Collect and report consumer metrics periodically."""
            while self.is_running:
                try:
                    # Update consumer lag
                    self._update_partition_assignment()
                    
                    # Calculate aggregate lag
                    total_lag = sum(
                        p['lag'] for p in self.partition_assignments.values()
                    )
                    self.metrics['consumer_lag']['total'] = total_lag
                    
                    # Send metrics to monitoring topic
                    if self.producer:
                        metrics_message = {
                            'timestamp': datetime.utcnow().isoformat(),
                            'group_id': self.group_id,
                            'metrics': self.metrics,
                            'partition_assignments': self.partition_assignments
                        }
                        self.producer.send(
                            self.metrics_topic,
                            key=f"consumer-{self.group_id}",
                            value=metrics_message
                        )
                    
                    # Store metrics in Redis if available
                    if self.redis_client:
                        self.redis_client.hset(
                            f"consumer_metrics:{self.group_id}",
                            mapping={
                                k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                                for k, v in self.metrics.items()
                            }
                        )
                        self.redis_client.expire(f"consumer_metrics:{self.group_id}", 3600)
                    
                except Exception as e:
                    logger.error(f"Error in metrics collection: {e}")
                
                # Sleep for 30 seconds between collections
                time.sleep(30)
        
        self.is_running = True
        collector_thread = threading.Thread(target=metrics_collector, daemon=True)
        collector_thread.start()
        logger.info("Metrics collection thread started")
    
    def consume_messages(
        self,
        batch_size: int = 100,
        timeout_ms: int = 1000,
        validate: bool = True,
        process_async: bool = False
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Consume messages from Kafka topics with comprehensive processing.
        
        This method yields batches of messages for processing, handling
        validation, error management, and offset tracking.
        
        Args:
            batch_size: Number of messages to accumulate before yielding
            timeout_ms: Poll timeout in milliseconds
            validate: Whether to validate messages before yielding
            process_async: Whether to process messages asynchronously
            
        Yields:
            Generator yielding lists of processed messages
            
        Example:
            consumer = KafkaStreamConsumer(...)
            consumer.connect()
            
            for message_batch in consumer.consume_messages(batch_size=100):
                for message in message_batch:
                    # Process each message
                    fraud_score = fraud_detection_model.predict(message)
                    
                # Commit offsets after batch processing
                consumer.commit()
        """
        if not self.consumer or not self.producer:
            raise ConnectionError("Consumer not connected. Call connect() first.")
        
        logger.info(f"Starting message consumption from topics: {self.topics}")
        
        message_buffer = []
        last_commit_time = time.time()
        commit_interval = 60  # Commit every 60 seconds if auto-commit disabled
        
        try:
            while self.is_running:
                try:
                    # Poll for messages
                    poll_start = time.time()
                    raw_messages = self.consumer.poll(timeout_ms=timeout_ms)
                    poll_time = time.time() - poll_start
                    
                    # Update metrics
                    self.metrics['last_poll_time'] = poll_time
                    
                    # Process each topic partition
                    for topic_partition, messages in raw_messages.items():
                        for message in messages:
                            # Track processing start time
                            process_start = time.time()
                            
                            try:
                                # Validate message
                                if validate:
                                    self._validate_message(message)
                                
                                # Process message (can be sync or async)
                                if process_async:
                                    # Submit for async processing
                                    asyncio.create_task(
                                        self._process_message_async(message)
                                    )
                                else:
                                    # Process synchronously
                                    processed_message = self._process_message(message)
                                    
                                    if processed_message:
                                        message_buffer.append(processed_message)
                                        
                                        # Update processing metrics
                                        process_time = time.time() - process_start
                                        self.processing_stats.append(process_time)
                                        self.metrics['messages_processed'] += 1
                                
                            except Exception as e:
                                # Handle processing error
                                self._handle_processing_error(message, e)
                            
                            # Update total messages consumed
                            self.metrics['messages_consumed'] += 1
                    
                    # Yield batch if ready
                    if len(message_buffer) >= batch_size:
                        yield message_buffer
                        message_buffer = []
                        
                        # Commit offsets if not using auto-commit
                        if not self.consumer_config['enable_auto_commit']:
                            current_time = time.time()
                            if current_time - last_commit_time >= commit_interval:
                                self.consumer.commit()
                                last_commit_time = current_time
                    
                    # Update average processing time
                    if self.processing_stats:
                        self.metrics['average_processing_time'] = np.mean(
                            self.processing_stats
                        )
                    
                except KafkaError as e:
                    logger.error(f"Kafka error during message consumption: {e}")
                    self.metrics['errors_by_type']['kafka_error'] = \
                        self.metrics['errors_by_type'].get('kafka_error', 0) + 1
                    
                    # Attempt to reconnect
                    time.sleep(5)
                    self.reconnect()
                    
                except Exception as e:
                    logger.error(f"Unexpected error in consume loop: {e}")
                    self.metrics['errors_by_type']['unexpected_error'] = \
                        self.metrics['errors_by_type'].get('unexpected_error', 0) + 1
                    
        except KeyboardInterrupt:
            logger.info("Consumer stopped by user")
        finally:
            # Yield any remaining messages
            if message_buffer:
                yield message_buffer
            
            self.close()
    
    def _validate_message(self, message):
        """
        Validate message format and content.
        
        Args:
            message: Kafka message object to validate
            
        Raises:
            ValueError: If message validation fails
        """
        # Check for null message
        if message is None:
            raise ValueError("Received null message")
        
        # Check value exists
        if message.value is None:
            raise ValueError("Message value is null")
        
        # Basic structure validation
        required_keys = {'transaction_id', 'account_id', 'amount', 'timestamp'}
        message_dict = message.value
        
        # Check required fields
        missing_keys = required_keys - set(message_dict.keys())
        if missing_keys:
            raise ValueError(f"Missing required fields: {missing_keys}")
        
        # Data type validation
        if not isinstance(message_dict.get('amount'), (int, float)):
            raise ValueError("Amount must be numeric")
        
        # Timestamp format validation
        try:
            datetime.fromisoformat(message_dict['timestamp'])
        except (ValueError, TypeError):
            raise ValueError("Invalid timestamp format")
    
    def _process_message(self, message) -> Optional[Dict[str, Any]]:
        """
        Process a single message synchronously.
        
        Args:
            message: Raw Kafka message
            
        Returns:
            Processed message dictionary or None if skipped
        """
        try:
            # Convert to standard format
            processed = {
                'raw_message': message.value,
                'topic': message.topic,
                'partition': message.partition,
                'offset': message.offset,
                'key': message.key,
                'timestamp': message.timestamp,
                'timestamp_type': message.timestamp_type,
                'processing_time': datetime.utcnow().isoformat(),
                'message_id': str(uuid.uuid4())  # Generate unique ID for tracking
            }
            
            # Add enrichment if available
            processed = self._enrich_message(processed)
            
            # Apply business rules
            processed = self._apply_business_rules(processed)
            
            # Store in Redis for real-time features
            self._store_in_redis(processed)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.metrics['messages_failed'] += 1
            return None
    
    async def _process_message_async(self, message):
        """
        Process a message asynchronously for better throughput.
        
        Args:
            message: Raw Kafka message
        """
        try:
            # Simulate async processing (could be database writes, API calls, etc.)
            processed = self._process_message(message)
            
            if processed:
                # Store in Redis asynchronously
                if self.redis_client:
                    await self._async_redis_store(processed)
                
                # Could send to another Kafka topic for further processing
                await self._async_kafka_produce(processed)
                
        except Exception as e:
            logger.error(f"Error in async processing: {e}")
            self.metrics['messages_failed'] += 1
    
    def _enrich_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich message with additional data from various sources.
        
        Args:
            message: Processed message to enrich
            
        Returns:
            Enriched message with additional fields
        """
        raw_data = message.get('raw_message', {})
        
        # Extract location from IP address if available
        ip_address = raw_data.get('ip_address')
        if ip_address and self.redis_client:
            # Check IP cache in Redis
            cached_location = self.redis_client.get(f"ip_geo:{ip_address}")
            if cached_location:
                message['geo_data'] = json.loads(cached_location)
        
        # Calculate time-based features
        transaction_time = datetime.fromisoformat(raw_data.get('timestamp', ''))
        current_time = datetime.utcnow()
        
        message['time_features'] = {
            'hour_of_day': transaction_time.hour,
            'day_of_week': transaction_time.weekday(),
            'is_weekend': transaction_time.weekday() >= 5,
            'processing_delay_seconds': (current_time - transaction_time).total_seconds()
        }
        
        # Add risk indicators
        message['risk_indicators'] = {
            'high_amount': raw_data.get('amount', 0) > 10000,
            'unusual_hour': transaction_time.hour in range(0, 5),  # 12 AM - 5 AM
            'international': raw_data.get('is_international', False)
        }
        
        return message
    
    def _apply_business_rules(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply business rules to flag potential fraud indicators.
        
        Args:
            message: Enriched message
            
        Returns:
            Message with business rule flags
        """
        raw_data = message.get('raw_message', {})
        
        # Initialize business rules
        business_rules = {
            'rule_flags': [],
            'rule_scores': {},
            'priority': 'NORMAL'
        }
        
        # Rule 1: Amount threshold check
        amount = raw_data.get('amount', 0)
        if amount > 50000:
            business_rules['rule_flags'].append('HIGH_VALUE_TRANSACTION')
            business_rules['rule_scores']['high_value'] = min(amount / 100000, 1.0)
        
        # Rule 2: Velocity check (can be enhanced with Redis counters)
        account_id = raw_data.get('account_id')
        if account_id and self.redis_client:
            tx_count = self.redis_client.get(f"tx_count:{account_id}:last_hour")
            if tx_count and int(tx_count) > 10:
                business_rules['rule_flags'].append('HIGH_VELOCITY')
                business_rules['rule_scores']['velocity'] = 0.8
        
        # Rule 3: Unusual location check
        if raw_data.get('is_international') and not raw_data.get('card_present'):
            business_rules['rule_flags'].append('INTERNATIONAL_CARD_NOT_PRESENT')
            business_rules['rule_scores']['location'] = 0.7
        
        # Rule 4: Device risk check
        device_id = raw_data.get('device_id')
        if device_id and self.redis_client:
            device_risk = self.redis_client.get(f"device_risk:{device_id}")
            if device_risk:
                business_rules['rule_scores']['device'] = float(device_risk)
                if float(device_risk) > 0.8:
                    business_rules['rule_flags'].append('HIGH_RISK_DEVICE')
        
        # Set priority based on number of flags
        if len(business_rules['rule_flags']) >= 3:
            business_rules['priority'] = 'HIGH'
        elif len(business_rules['rule_flags']) >= 1:
            business_rules['priority'] = 'MEDIUM'
        
        message['business_rules'] = business_rules
        
        return message
    
    def _store_in_redis(self, message: Dict[str, Any]):
        """
        Store message data in Redis for real-time feature calculation.
        
        Args:
            message: Processed message to store
        """
        if not self.redis_client:
            return
        
        try:
            raw_data = message.get('raw_message', {})
            account_id = raw_data.get('account_id')
            transaction_id = raw_data.get('transaction_id')
            
            if account_id and transaction_id:
                # Store transaction in Redis with TTL
                tx_key = f"transaction:{transaction_id}"
                self.redis_client.hset(tx_key, mapping={
                    'account_id': account_id,
                    'amount': str(raw_data.get('amount', 0)),
                    'timestamp': raw_data.get('timestamp', ''),
                    'processed_time': message.get('processing_time', '')
                })
                self.redis_client.expire(tx_key, 86400)  # 24 hours
                
                # Update account counters
                current_hour = datetime.utcnow().strftime('%Y%m%d%H')
                hour_key = f"account_hour:{account_id}:{current_hour}"
                self.redis_client.incr(hour_key)
                self.redis_client.expire(hour_key, 7200)  # 2 hours
                
                # Store transaction in sorted set for time-based queries
                timestamp = datetime.fromisoformat(raw_data.get('timestamp', '')).timestamp()
                self.redis_client.zadd(
                    f"account_transactions:{account_id}",
                    {transaction_id: timestamp}
                )
                self.redis_client.expire(f"account_transactions:{account_id}", 604800)  # 7 days
                
        except Exception as e:
            logger.error(f"Error storing in Redis: {e}")
    
    async def _async_redis_store(self, message: Dict[str, Any]):
        """Async version of Redis storage."""
        # Implement async Redis operations if using aioredis
        pass
    
    async def _async_kafka_produce(self, message: Dict[str, Any]):
        """Async Kafka production for further processing."""
        # Could implement with aiokafka
        pass
    
    def _handle_processing_error(self, message, error: Exception):
        """
        Handle message processing errors by sending to Dead Letter Queue.
        
        Args:
            message: Failed message
            error: Exception that occurred
        """
        logger.error(f"Processing error for message: {error}")
        self.metrics['messages_failed'] += 1
        
        try:
            # Prepare DLQ message
            dlq_message = {
                'original_message': message.value if hasattr(message, 'value') else message,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'timestamp': datetime.utcnow().isoformat(),
                'topic': message.topic if hasattr(message, 'topic') else 'unknown',
                'partition': message.partition if hasattr(message, 'partition') else -1,
                'offset': message.offset if hasattr(message, 'offset') else -1,
                'consumer_group': self.group_id
            }
            
            # Send to DLQ topic
            if self.producer:
                future = self.producer.send(
                    self.dead_letter_topic,
                    key=dlq_message['original_message'].get('transaction_id', 'unknown'),
                    value=dlq_message
                )
                future.get(timeout=10)  # Wait for send to complete
                self.metrics['messages_dlq'] += 1
                
                logger.info(f"Message sent to DLQ. Error: {error}")
                
        except Exception as e:
            logger.error(f"Failed to send message to DLQ: {e}")
            self.metrics['errors_by_type']['dlq_error'] = \
                self.metrics['errors_by_type'].get('dlq_error', 0) + 1
    
    def reconnect(self):
        """
        Reconnect to Kafka with exponential backoff.
        
        Implements exponential backoff strategy for reconnection attempts.
        """
        backoff = 1
        max_backoff = 60
        attempts = 0
        
        while self.is_running and attempts < 10:
            try:
                logger.info(f"Attempting to reconnect (attempt {attempts + 1})...")
                
                # Close existing connections
                self.close()
                
                # Attempt reconnection
                if self.connect():
                    logger.info("Successfully reconnected to Kafka")
                    return True
                
            except Exception as e:
                logger.error(f"Reconnection attempt {attempts + 1} failed: {e}")
            
            # Exponential backoff
            time.sleep(min(backoff, max_backoff))
            backoff *= 2
            attempts += 1
        
        logger.error("Failed to reconnect after maximum attempts")
        return False
    
    def commit(self):
        """Manually commit offsets."""
        if self.consumer and not self.consumer_config['enable_auto_commit']:
            try:
                self.consumer.commit()
                logger.debug("Offsets committed successfully")
            except Exception as e:
                logger.error(f"Error committing offsets: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current consumer metrics."""
        return {
            **self.metrics,
            'partition_assignments': self.partition_assignments,
            'processing_stats': {
                'mean': float(np.mean(self.processing_stats)) if self.processing_stats else 0,
                'median': float(np.median(self.processing_stats)) if self.processing_stats else 0,
                'p95': float(np.percentile(self.processing_stats, 95)) if self.processing_stats else 0,
                'p99': float(np.percentile(self.processing_stats, 99)) if self.processing_stats else 0
            }
        }
    
    def close(self):
        """Clean up resources and close connections."""
        logger.info("Closing consumer connections...")
        
        self.is_running = False
        
        if self.consumer:
            try:
                self.consumer.close()
                logger.info("Kafka consumer closed")
            except Exception as e:
                logger.error(f"Error closing consumer: {e}")
        
        if self.producer:
            try:
                self.producer.close()
                logger.info("Kafka producer closed")
            except Exception as e:
                logger.error(f"Error closing producer: {e}")
        
        if self.redis_client:
            try:
                self.redis_client.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")


class AsyncKafkaStreamConsumer:
    """
    AsyncIO-based Kafka consumer for high-throughput, non-blocking consumption.
    
    This class uses aiokafka for asynchronous message processing, allowing
    for better scalability in I/O-bound scenarios.
    """
    
    def __init__(self, bootstrap_servers: List[str], group_id: str, topics: List[str]):
        """
        Initialize async Kafka consumer.
        
        Args:
            bootstrap_servers: List of Kafka broker addresses
            group_id: Consumer group ID
            topics: List of topics to subscribe to
        """
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.topics = topics
        self.consumer = None
        
    async def start(self):
        """Start the async consumer."""
        self.consumer = aiokafka.AIOKafkaConsumer(
            *self.topics,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            auto_offset_reset='earliest',
            enable_auto_commit=False
        )
        await self.consumer.start()
        
    async def consume(self):
        """Consume messages asynchronously."""
        async for message in self.consumer:
            # Process message without blocking
            yield message
            
    async def stop(self):
        """Stop the async consumer."""
        if self.consumer:
            await self.consumer.stop()


class RabbitMQStreamConsumer:
    """
    RabbitMQ consumer for banking data streams.
    
    Alternative message broker implementation for systems using RabbitMQ.
    """
    
    def __init__(self, host: str, port: int, queue_name: str, username: str, password: str):
        """
        Initialize RabbitMQ consumer.
        
        Args:
            host: RabbitMQ host
            port: RabbitMQ port
            queue_name: Queue to consume from
            username: Authentication username
            password: Authentication password
        """
        self.host = host
        self.port = port
        self.queue_name = queue_name
        self.username = username
        self.password = password
        self.connection = None
        self.channel = None
        
    def connect(self):
        """Establish connection to RabbitMQ."""
        import pika
        
        credentials = pika.PlainCredentials(self.username, self.password)
        parameters = pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            credentials=credentials,
            heartbeat=600,
            blocked_connection_timeout=300
        )
        
        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=self.queue_name, durable=True)
        
        logger.info(f"Connected to RabbitMQ at {self.host}:{self.port}")
        
    def consume_messages(self, callback=None):
        """
        Start consuming messages from RabbitMQ.
        
        Args:
            callback: Function to call for each message
        """
        def default_callback(ch, method, properties, body):
            """Default message callback."""
            try:
                message = json.loads(body)
                logger.debug(f"Received message: {message.get('transaction_id')}")
                
                # Acknowledge message
                ch.basic_ack(delivery_tag=method.delivery_tag)
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                # Reject and requeue
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
        
        callback = callback or default_callback
        
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=callback
        )
        
        logger.info(f"Started consuming from queue: {self.queue_name}")
        self.channel.start_consuming()
        
    def close(self):
        """Close RabbitMQ connection."""
        if self.connection:
            self.connection.close()
            logger.info("RabbitMQ connection closed")


class WebSocketStreamConsumer:
    """
    WebSocket consumer for real-time data streams.
    
    Useful for consuming from WebSocket APIs that provide real-time
    transaction data (e.g., payment gateways, crypto exchanges).
    """
    
    def __init__(self, url: str, reconnect: bool = True):
        """
        Initialize WebSocket consumer.
        
        Args:
            url: WebSocket endpoint URL
            reconnect: Whether to auto-reconnect on disconnection
        """
        self.url = url
        self.reconnect = reconnect
        self.ws = None
        
    async def connect(self):
        """Establish WebSocket connection."""
        import websockets
        
        self.ws = await websockets.connect(
            self.url,
            ping_interval=20,
            ping_timeout=60
        )
        logger.info(f"Connected to WebSocket: {self.url}")
        
    async def consume(self):
        """Consume messages from WebSocket."""
        while True:
            try:
                message = await self.ws.recv()
                data = json.loads(message)
                yield data
                
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed")
                if self.reconnect:
                    await self.connect()
                else:
                    break
                    
    async def close(self):
        """Close WebSocket connection."""
        if self.ws:
            await self.ws.close()
            logger.info("WebSocket connection closed")


class FileStreamConsumer:
    """
    File-based stream consumer for batch processing and testing.
    
    Reads data from files (CSV, Parquet, JSON) as if they were streams,
    useful for development, testing, and batch processing scenarios.
    """
    
    def __init__(self, file_path: str, file_type: str = 'auto', chunk_size: int = 1000):
        """
        Initialize file consumer.
        
        Args:
            file_path: Path to input file
            file_type: Type of file ('csv', 'parquet', 'json', 'auto')
            chunk_size: Number of records to read per chunk
        """
        self.file_path = file_path
        self.file_type = file_type if file_type != 'auto' else self._detect_file_type(file_path)
        self.chunk_size = chunk_size
        self.reader = None
        
    def _detect_file_type(self, file_path: str) -> str:
        """Detect file type from extension."""
        extension = file_path.split('.')[-1].lower()
        
        if extension == 'csv':
            return 'csv'
        elif extension == 'parquet':
            return 'parquet'
        elif extension in ['json', 'jsonl']:
            return 'json'
        else:
            raise ValueError(f"Unsupported file type: {extension}")
    
    def connect(self):
        """Initialize file reader based on file type."""
        logger.info(f"Opening file: {self.file_path} (type: {self.file_type})")
        
        if self.file_type == 'csv':
            self.reader = pd.read_csv(
                self.file_path,
                chunksize=self.chunk_size,
                parse_dates=True,
                infer_datetime_format=True
            )
            
        elif self.file_type == 'parquet':
            # For Parquet, we'll use PyArrow for chunked reading
            self.reader = pq.ParquetFile(self.file_path)
            
        elif self.file_type == 'json':
            self.reader = self._json_line_reader()
    
    def _json_line_reader(self):
        """Create generator for JSON line-delimited files."""
        def generator():
            with open(self.file_path, 'r') as f:
                batch = []
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        batch.append(data)
                        
                        if len(batch) >= self.chunk_size:
                            yield batch
                            batch = []
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing JSON: {e}")
                        
                if batch:
                    yield batch
                    
        return generator()
    
    def consume_messages(self) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Consume messages from file in chunks.
        
        Yields:
            Batches of messages read from file
        """
        if not self.reader:
            self.connect()
        
        if self.file_type == 'csv':
            for chunk in self.reader:
                yield chunk.to_dict('records')
                
        elif self.file_type == 'parquet':
            for batch in self.reader.iter_batches(batch_size=self.chunk_size):
                df = batch.to_pandas()
                yield df.to_dict('records')
                
        elif self.file_type == 'json':
            for batch in self.reader:
                yield batch
    
    def close(self):
        """Close file reader."""
        logger.info("File consumer closed")