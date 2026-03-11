"""
VeritasFinancial - Data Acquisition Module
=========================================
This module handles all data ingestion from various banking data sources including:
- Core banking transaction systems
- Customer relationship management (CRM) systems
- Device fingerprinting services
- External fraud intelligence feeds

The module ensures data quality, consistency, and real-time processing capabilities
for fraud detection at scale.
"""

from .api_clients import (
    BankingAPIClient,           # Main client for banking system APIs
    FraudIntelligenceClient,     # External fraud data provider client
    MerchantDataClient,          # Merchant information and risk scoring
    GeoIPClient,                 # IP geolocation and risk assessment
    DeviceFingerprintClient      # Device identification and tracking
)

from .database_connectors import (
    PostgreSQLConnector,         # Main transaction database connector
    MongoDBConnector,            # Document store for device fingerprints
    RedisConnector,              # In-memory cache for real-time features
    CassandraConnector           # Time-series data for transaction history
)

from .stream_consumers import (
    KafkaConsumer,               # Real-time transaction stream consumer
    KinesisConsumer,             # AWS Kinesis stream consumer
    RabbitMQConsumer,            # Message queue consumer for events
    WebSocketConsumer            # Real-time push notifications
)

from .data_validators import (
    TransactionValidator,        # Validates transaction data structure
    CustomerValidator,           # Validates customer profile data
    DeviceValidator,             # Validates device fingerprint data
    SchemaValidator,             # Generic schema validation framework
    DataQualityChecker           # Comprehensive data quality checks
)

__all__ = [
    # API Clients
    'BankingAPIClient',
    'FraudIntelligenceClient',
    'MerchantDataClient',
    'GeoIPClient',
    'DeviceFingerprintClient',
    
    # Database Connectors
    'PostgreSQLConnector',
    'MongoDBConnector',
    'RedisConnector',
    'CassandraConnector',
    
    # Stream Consumers
    'KafkaConsumer',
    'KinesisConsumer',
    'RabbitMQConsumer',
    'WebSocketConsumer',
    
    # Data Validators
    'TransactionValidator',
    'CustomerValidator',
    'DeviceValidator',
    'SchemaValidator',
    'DataQualityChecker'
]