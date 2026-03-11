"""
VeritasFinancial - Exploratory Analysis Module
=============================================
This module provides comprehensive exploratory data analysis (EDA) tools
for banking fraud detection. It includes statistical analysis, visualization,
correlation studies, temporal pattern analysis, and anomaly detection.

The module is designed to help data scientists understand:
- Data quality and missing patterns
- Statistical properties of transactions
- Fraud patterns across different dimensions
- Temporal and seasonal fraud trends
- Correlations and feature relationships
- Anomalies and outliers in financial data

Author: VeritasFinancial Data Science Team
Version: 1.0.0
"""

# Import main classes and functions to make them available at package level
from .statistical_analysis import (
    FraudStatisticalAnalysis,
    DescriptiveStatistics,
    HypothesisTester,
    DistributionAnalyzer,
    QualityAssessor
)

from .visualizations import (
    FraudVisualizer,
    InteractiveDashboard,
    DistributionPlotter,
    TimeSeriesPlotter,
    GeographicMapper
)

from .correlation_studies import (
    CorrelationAnalyzer,
    FeatureSelector,
    MulticollinearityDetector,
    AssociationMiner
)

from .temporal_analysis import (
    TemporalAnalyzer,
    SeasonalityDetector,
    TrendAnalyzer,
    TimeGapAnalyzer,
    VelocityCalculator
)

from .anomaly_detection import (
    AnomalyDetector,
    OutlierAnalyzer,
    IsolationForestDetector,
    StatisticalOutlierDetector,
    DensityBasedDetector
)

# Package metadata
__version__ = '1.0.0'
__author__ = 'VeritasFinancial'
__all__ = [
    # Statistical Analysis
    'FraudStatisticalAnalysis',
    'DescriptiveStatistics',
    'HypothesisTester',
    'DistributionAnalyzer',
    'QualityAssessor',
    
    # Visualizations
    'FraudVisualizer',
    'InteractiveDashboard',
    'DistributionPlotter',
    'TimeSeriesPlotter',
    'GeographicMapper',
    
    # Correlation Studies
    'CorrelationAnalyzer',
    'FeatureSelector',
    'MulticollinearityDetector',
    'AssociationMiner',
    
    # Temporal Analysis
    'TemporalAnalyzer',
    'SeasonalityDetector',
    'TrendAnalyzer',
    'TimeGapAnalyzer',
    'VelocityCalculator',
    
    # Anomaly Detection
    'AnomalyDetector',
    'OutlierAnalyzer',
    'IsolationForestDetector',
    'StatisticalOutlierDetector',
    'DensityBasedDetector'
]

# Configure logging for the module
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Module-level constants
DEFAULT_FIGURE_SIZE = (12, 8)
DEFAULT_COLOR_PALETTE = {
    'fraud': '#FF6B6B',      # Red for fraud
    'normal': '#4ECDC4',      # Teal for normal
    'warning': '#FFE66D',     # Yellow for warnings
    'info': '#95E1D3'         # Light teal for information
}

# Configure plotting style for consistency
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')