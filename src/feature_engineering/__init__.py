"""
VeritasFinancial Feature Engineering Module
===========================================
This module provides comprehensive feature engineering capabilities for banking fraud detection.
It transforms raw transaction data into sophisticated features that capture fraud patterns.

The module is organized into several submodules:
- domain_features: Business-specific features (transactions, customers, devices, behavior)
- temporal_features: Time-based features (rolling statistics, seasonality, time gaps)
- aggregate_features: Aggregated statistics (customer, merchant, device levels)
- graph_features: Network-based features (connections, communities)
- embedding_features: Learned representations (transactions, categories)

Author: VeritasFinancial Data Science Team
Version: 1.0.0
"""

# Import main classes and functions for easy access
from .domain_features.transaction_features import (
    TransactionFeatureEngineer,  # Creates transaction-level features
    AmountFeatureExtractor,      # Extracts amount-based features
    LocationFeatureExtractor,    # Extracts location-based features
    MerchantFeatureExtractor,    # Extracts merchant-based features
)

from .domain_features.customer_features import (
    CustomerFeatureEngineer,     # Creates customer-level features
    DemographicsFeatureExtractor,# Extracts demographic features
    AccountFeatureExtractor,     # Extracts account-based features
    RiskProfileFeatureExtractor, # Extracts risk profile features
)

from .domain_features.device_features import (
    DeviceFeatureEngineer,       # Creates device-level features
    FingerprintFeatureExtractor, # Extracts device fingerprint features
    BehavioralFeatureExtractor,  # Extracts device behavioral features
)

from .domain_features.behavioral_features import (
    BehavioralFeatureEngineer,   # Creates behavioral pattern features
    SpendingPatternAnalyzer,     # Analyzes spending patterns
    DeviationDetector,           # Detects deviations from normal behavior
    VelocityCalculator,          # Calculates transaction velocities
)

from .temporal_features.rolling_statistics import (
    RollingStatisticsCalculator, # Calculates rolling window statistics
    WindowFeatureExtractor,      # Extracts window-based features
    ExpandingFeatureExtractor,   # Extracts expanding window features
)

from .temporal_features.seasonality import (
    SeasonalityAnalyzer,         # Analyzes seasonal patterns
    TimeSeriesDecomposer,        # Decomposes time series components
    CyclicalFeatureEncoder,      # Encodes cyclical time features
)

from .temporal_features.time_gaps import (
    TimeGapAnalyzer,             # Analyzes time gaps between transactions
    IrregularIntervalHandler,    # Handles irregular time intervals
    SessionDetector,             # Detects transaction sessions
)

from .aggregate_features.customer_aggregates import (
    CustomerAggregator,          # Aggregates customer-level statistics
    CustomerHistoryAnalyzer,     # Analyzes customer transaction history
    CustomerSegmentation,        # Segments customers based on behavior
)

from .aggregate_features.merchant_aggregates import (
    MerchantAggregator,          # Aggregates merchant-level statistics
    MerchantRiskAnalyzer,        # Analyzes merchant risk patterns
    MerchantCategoryFeatures,    # Creates merchant category features
)

from .aggregate_features.device_aggregates import (
    DeviceAggregator,            # Aggregates device-level statistics
    DeviceHistoryAnalyzer,       # Analyzes device transaction history
    DeviceRiskScorer,            # Scores device risk levels
)

from .graph_features.network_analysis import (
    NetworkAnalyzer,             # Analyzes transaction networks
    GraphBuilder,                # Builds transaction graphs
    CentralityCalculator,        # Calculates network centrality measures
    PathAnalyzer,                # Analyzes transaction paths
)

from .graph_features.community_detection import (
    CommunityDetector,           # Detects communities in transaction networks
    ClusterAnalyzer,             # Analyzes transaction clusters
    FraudRingDetector,           # Detects potential fraud rings
)

from .embedding_features.transaction_embeddings import (
    TransactionEmbedder,         # Creates transaction embeddings
    SequenceEncoder,             # Encodes transaction sequences
    ContextualEmbedder,          # Creates contextual embeddings
)

from .embedding_features.categorical_embeddings import (
    CategoricalEmbedder,         # Creates embeddings for categorical variables
    EntityEncoder,               # Encodes entities (customers, merchants)
    LearnedRepresentation,       # Manages learned representations
)

# Define the public API
__all__ = [
    # Domain Features
    'TransactionFeatureEngineer',
    'AmountFeatureExtractor',
    'LocationFeatureExtractor',
    'MerchantFeatureExtractor',
    'CustomerFeatureEngineer',
    'DemographicsFeatureExtractor',
    'AccountFeatureExtractor',
    'RiskProfileFeatureExtractor',
    'DeviceFeatureEngineer',
    'FingerprintFeatureExtractor',
    'BehavioralFeatureExtractor',
    'BehavioralFeatureEngineer',
    'SpendingPatternAnalyzer',
    'DeviationDetector',
    'VelocityCalculator',
    
    # Temporal Features
    'RollingStatisticsCalculator',
    'WindowFeatureExtractor',
    'ExpandingFeatureExtractor',
    'SeasonalityAnalyzer',
    'TimeSeriesDecomposer',
    'CyclicalFeatureEncoder',
    'TimeGapAnalyzer',
    'IrregularIntervalHandler',
    'SessionDetector',
    
    # Aggregate Features
    'CustomerAggregator',
    'CustomerHistoryAnalyzer',
    'CustomerSegmentation',
    'MerchantAggregator',
    'MerchantRiskAnalyzer',
    'MerchantCategoryFeatures',
    'DeviceAggregator',
    'DeviceHistoryAnalyzer',
    'DeviceRiskScorer',
    
    # Graph Features
    'NetworkAnalyzer',
    'GraphBuilder',
    'CentralityCalculator',
    'PathAnalyzer',
    'CommunityDetector',
    'ClusterAnalyzer',
    'FraudRingDetector',
    
    # Embedding Features
    'TransactionEmbedder',
    'SequenceEncoder',
    'ContextualEmbedder',
    'CategoricalEmbedder',
    'EntityEncoder',
    'LearnedRepresentation',
]

# Module metadata
__version__ = '1.0.0'
__author__ = 'VeritasFinancial Data Science Team'
__email__ = 'datascience@veritasfinancial.com'
__description__ = 'Advanced feature engineering module for banking fraud detection'

# Module-level configuration
DEFAULT_CONFIG = {
    'rolling_windows': ['1H', '24H', '7D', '30D'],
    'aggregation_functions': ['mean', 'std', 'min', 'max', 'count', 'sum'],
    'embedding_dimensions': {
        'merchant': 32,
        'customer': 64,
        'device': 16,
        'category': 8
    },
    'graph_parameters': {
        'max_depth': 3,
        'min_community_size': 5,
        'centrality_measures': ['degree', 'betweenness', 'closeness']
    }
}

def get_feature_engineering_pipeline(config=None):
    """
    Factory function to create a complete feature engineering pipeline.
    
    This function assembles all feature engineering components into a single
    pipeline that can process raw transaction data and generate all features.
    
    Args:
        config (dict, optional): Configuration dictionary. If None, uses DEFAULT_CONFIG.
        
    Returns:
        dict: Dictionary containing all initialized feature engineering components
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    pipeline = {
        'transaction_features': TransactionFeatureEngineer(config),
        'customer_features': CustomerFeatureEngineer(config),
        'device_features': DeviceFeatureEngineer(config),
        'behavioral_features': BehavioralFeatureEngineer(config),
        'rolling_stats': RollingStatisticsCalculator(config),
        'seasonality': SeasonalityAnalyzer(config),
        'time_gaps': TimeGapAnalyzer(config),
        'customer_aggregates': CustomerAggregator(config),
        'merchant_aggregates': MerchantAggregator(config),
        'device_aggregates': DeviceAggregator(config),
        'network_analysis': NetworkAnalyzer(config),
        'community_detection': CommunityDetector(config),
        'transaction_embeddings': TransactionEmbedder(config),
        'categorical_embeddings': CategoricalEmbedder(config)
    }
    
    return pipeline