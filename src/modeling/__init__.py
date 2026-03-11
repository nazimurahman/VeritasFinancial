"""
VeritasFinancial - Banking Fraud Detection System
Modeling Module Initialization
==================================================
This module contains all machine learning models and training utilities
for fraud detection in banking transactions.

The module is organized into four main submodules:
1. classical_ml/ - Traditional ML algorithms (XGBoost, LightGBM, Isolation Forest)
2. deep_learning/ - Neural network architectures (Autoencoders, LSTM, Transformers)
3. training/ - Training utilities (CV, hyperparameter tuning, early stopping)
4. evaluation/ - Model evaluation metrics and interpretability tools

Author: VeritasFinancial Team
Version: 1.0.0
"""

# Version information
__version__ = "1.0.0"
__author__ = "VeritasFinancial Team"

# Import from classical_ml submodule
from .classical_ml.isolation_forest import (
    IsolationForestDetector,        # Unsupervised anomaly detection
    OnlineIsolationForest,           # Streaming anomaly detection
    IsolationForestConfig            # Configuration class
)

from .classical_ml.xgboost_model import (
    FraudXGBoostClassifier,          # XGBoost with fraud-specific features
    XGBoostHyperparameterTuner,      # Automated hyperparameter optimization
    XGBoostModelConfig,              # Configuration management
    XGBoostModelArtifact              # Model serialization utilities
)

from .classical_ml.lightgbm_model import (
    FraudLightGBMClassifier,         # LightGBM with fraud-specific features
    LightGBMHyperparameterTuner,      # Automated hyperparameter optimization
    LightGBMModelConfig,              # Configuration management
    LightGBMModelArtifact              # Model serialization utilities
)

from .classical_ml.ensemble_methods import (
    FraudEnsembleClassifier,          # Ensemble of multiple models
    StackingEnsemble,                 # Stacking ensemble with meta-learner
    VotingEnsemble,                    # Weighted voting ensemble
    EnsembleConfig,                    # Ensemble configuration
    ModelWeightOptimizer                # Optimize ensemble weights
)

# Import from deep_learning submodule
from .deep_learning.neural_networks import (
    FraudNeuralNetwork,               # Multi-layer perceptron for fraud
    DeepFraudDetector,                 # Deep neural network architecture
    NeuralNetworkConfig,                # NN configuration
    LayerConfig,                        # Individual layer configuration
    ActivationFunctions                  # Custom activation functions
)

from .deep_learning.autoencoders import (
    FraudAutoencoder,                  # Autoencoder for anomaly detection
    VariationalAutoencoder,             # VAE for probabilistic fraud detection
    DenoisingAutoencoder,                # Robust feature learning
    AutoencoderConfig,                    # Autoencoder configuration
    ReconstructionLoss,                    # Custom loss functions
    AnomalyScoreCalculator                  # Score from reconstruction error
)

from .deep_learning.lstm_models import (
    FraudLSTMClassifier,                # LSTM for sequence fraud detection
    BiLSTMAttention,                     # BiLSTM with attention mechanism
    LSTMAutoencoder,                      # LSTM autoencoder for sequences
    LSTMConfig,                            # LSTM configuration
    SequenceDataLoader,                     # Sequence data utilities
    TemporalFeatureExtractor                  # Extract temporal features
)

from .deep_learning.transformers import (
    FraudTransformerEncoder,              # Transformer for fraud detection
    MultiHeadAttention,                     # Multi-head attention mechanism
    PositionalEncoding,                       # Positional embeddings
    TransformerBlock,                           # Single transformer block
    GQAAttention,                                 # Grouped Query Attention
    FlashAttentionKernel,                         # Efficient attention
    RotaryPositionalEmbedding,                      # RoPE embeddings
    TransformerConfig                                 # Transformer configuration
)

# Import from training submodule
from .training.cross_validation import (
    FraudCrossValidator,                   # Cross-validation for fraud
    TimeSeriesSplitter,                      # Time-based CV splits
    StratifiedKFoldFraud,                      # Stratified CV for imbalanced data
    PurgedGroupTimeSeriesSplit,                  # Purged CV for time series
    CVResults,                                       # CV results storage
    CVVisualizer                                        # CV visualization tools
)

from .training.hyperparameter_tuning import (
    FraudHyperparameterOptimizer,            # Hyperparameter optimization
    BayesianOptimizer,                          # Bayesian optimization
    GridSearchOptimizer,                          # Grid search
    RandomSearchOptimizer,                          # Random search
    HyperbandOptimizer,                               # Hyperband optimization
    OptunaOptimizer,                                     # Optuna integration
    HyperparameterSpace,                                    # Parameter space definition
    TuningResults,                                             # Tuning results storage
    ParallelTuner                                                 # Parallel tuning support
)

from .training.early_stopping import (
    FraudEarlyStopping,                         # Early stopping for training
    MetricBasedStopping,                           # Stop based on metric
    GradientBasedStopping,                           # Stop based on gradient
    PatienceStopping,                                   # Stop based on patience
    EnsembleStopping,                                      # Ensemble of stopping criteria
    StoppingConfig,                                           # Stopping configuration
    TrainingMonitor                                          # Real-time training monitoring
)

# Import from evaluation submodule
from .evaluation.metrics import (
    FraudMetricsCalculator,                      # Comprehensive metrics
    PrecisionRecallCalculator,                       # Precision-recall calculations
    ROCCurveCalculator,                                 # ROC curve calculations
    CostSensitiveMetrics,                                  # Business cost metrics
    MetricsVisualizer,                                          # Metrics visualization
    MetricConfig,                                                 # Metrics configuration
    ConfidenceIntervalCalculator                                      # Statistical significance
)

from .evaluation.thresholds import (
    FraudThresholdOptimizer,                     # Optimal threshold selection
    ThresholdFinder,                                 # Multiple threshold strategies
    BusinessCostOptimizer,                              # Cost-based threshold
    F1ThresholdOptimizer,                                   # F1-based threshold
    YoudensIndexCalculator,                                     # Youden's index
    ThresholdConfig,                                                # Threshold configuration
    DynamicThresholdAdjuster                                         # Adaptive thresholds
)

from .evaluation.interpretability import (
    FraudModelExplainer,                           # Model interpretability
    SHAPExplainer,                                      # SHAP values
    LIMEExplainer,                                        # LIME explanations
    FeatureImportanceAnalyzer,                              # Feature importance
    PartialDependencePlot,                                       # PDP plots
    IndividualConditionalExpectation,                             # ICE plots
    CounterfactualExplanations,                                      # Counterfactuals
    ExplainabilityConfig                                               # Explainability config
)

from .evaluation.business_metrics import (
    BusinessMetricsCalculator,                    # Business impact metrics
    FinancialImpactAnalyzer,                          # Cost/benefit analysis
    RegulatoryComplianceChecker,                        # Compliance metrics
    OperationalMetrics,                                     # Operational efficiency
    RiskAdjustedMetrics,                                        # Risk-based metrics
    BusinessKPIFormatter,                                          # KPI formatting
    ROICalculator,                                                     # ROI calculation
    BusinessThresholds                                              # Business decision thresholds
)

# Module exports - list all public classes
__all__ = [
    # Classical ML exports
    'IsolationForestDetector',
    'OnlineIsolationForest',
    'IsolationForestConfig',
    'FraudXGBoostClassifier',
    'XGBoostHyperparameterTuner',
    'XGBoostModelConfig',
    'XGBoostModelArtifact',
    'FraudLightGBMClassifier',
    'LightGBMHyperparameterTuner',
    'LightGBMModelConfig',
    'LightGBMModelArtifact',
    'FraudEnsembleClassifier',
    'StackingEnsemble',
    'VotingEnsemble',
    'EnsembleConfig',
    'ModelWeightOptimizer',
    
    # Deep Learning exports
    'FraudNeuralNetwork',
    'DeepFraudDetector',
    'NeuralNetworkConfig',
    'LayerConfig',
    'ActivationFunctions',
    'FraudAutoencoder',
    'VariationalAutoencoder',
    'DenoisingAutoencoder',
    'AutoencoderConfig',
    'ReconstructionLoss',
    'AnomalyScoreCalculator',
    'FraudLSTMClassifier',
    'BiLSTMAttention',
    'LSTMAutoencoder',
    'LSTMConfig',
    'SequenceDataLoader',
    'TemporalFeatureExtractor',
    'FraudTransformerEncoder',
    'MultiHeadAttention',
    'PositionalEncoding',
    'TransformerBlock',
    'GQAAttention',
    'FlashAttentionKernel',
    'RotaryPositionalEmbedding',
    'TransformerConfig',
    
    # Training exports
    'FraudCrossValidator',
    'TimeSeriesSplitter',
    'StratifiedKFoldFraud',
    'PurgedGroupTimeSeriesSplit',
    'CVResults',
    'CVVisualizer',
    'FraudHyperparameterOptimizer',
    'BayesianOptimizer',
    'GridSearchOptimizer',
    'RandomSearchOptimizer',
    'HyperbandOptimizer',
    'OptunaOptimizer',
    'HyperparameterSpace',
    'TuningResults',
    'ParallelTuner',
    'FraudEarlyStopping',
    'MetricBasedStopping',
    'GradientBasedStopping',
    'PatienceStopping',
    'EnsembleStopping',
    'StoppingConfig',
    'TrainingMonitor',
    
    # Evaluation exports
    'FraudMetricsCalculator',
    'PrecisionRecallCalculator',
    'ROCCurveCalculator',
    'CostSensitiveMetrics',
    'MetricsVisualizer',
    'MetricConfig',
    'ConfidenceIntervalCalculator',
    'FraudThresholdOptimizer',
    'ThresholdFinder',
    'BusinessCostOptimizer',
    'F1ThresholdOptimizer',
    'YoudensIndexCalculator',
    'ThresholdConfig',
    'DynamicThresholdAdjuster',
    'FraudModelExplainer',
    'SHAPExplainer',
    'LIMEExplainer',
    'FeatureImportanceAnalyzer',
    'PartialDependencePlot',
    'IndividualConditionalExpectation',
    'CounterfactualExplanations',
    'ExplainabilityConfig',
    'BusinessMetricsCalculator',
    'FinancialImpactAnalyzer',
    'RegulatoryComplianceChecker',
    'OperationalMetrics',
    'RiskAdjustedMetrics',
    'BusinessKPIFormatter',
    'ROICalculator',
    'BusinessThresholds'
]

# Module-level docstring
__doc__ = """
VeritasFinancial Modeling Module
================================

This module provides a comprehensive suite of machine learning models and utilities
specifically designed for banking fraud detection. It includes:

1. Classical ML Models:
   - XGBoost and LightGBM with fraud-specific optimizations
   - Isolation Forest for unsupervised anomaly detection
   - Ensemble methods combining multiple models

2. Deep Learning Models:
   - Neural networks with fraud-specific architectures
   - Autoencoders for anomaly detection
   - LSTM models for sequence fraud detection
   - Transformers with advanced attention mechanisms (GQA, Flash Attention)

3. Training Utilities:
   - Cross-validation strategies for imbalanced data
   - Hyperparameter optimization (Bayesian, Grid, Random, Hyperband)
   - Advanced early stopping criteria

4. Evaluation Tools:
   - Comprehensive metrics for fraud detection
   - Optimal threshold selection
   - Model interpretability (SHAP, LIME)
   - Business impact analysis

All models are production-ready with proper error handling, logging, and
configuration management. They are designed to handle the unique challenges
of fraud detection: extreme class imbalance, temporal patterns, and the
need for explainability.
"""

# Check dependencies
import importlib
import logging

# Configure module-level logger
logger = logging.getLogger(__name__)

def _check_dependencies():
    """
    Check if all required dependencies are installed.
    
    Raises:
        ImportError: If any required dependency is missing.
    """
    required_packages = [
        ('numpy', '1.24.0'),
        ('pandas', '2.0.0'),
        ('scikit-learn', '1.3.0'),
        ('xgboost', '2.0.0'),
        ('lightgbm', '4.0.0'),
        ('torch', '2.0.0'),
        ('transformers', '4.30.0'),
        ('shap', '0.42.0'),
        ('optuna', '3.3.0')
    ]
    
    missing_packages = []
    version_issues = []
    
    for package, min_version in required_packages:
        try:
            module = importlib.import_module(package.replace('-', '_'))
            if hasattr(module, '__version__'):
                current_version = module.__version__
                # Simple version comparison (can be enhanced)
                if current_version < min_version:
                    version_issues.append(
                        f"{package} {current_version} < required {min_version}"
                    )
            logger.info(f"Found {package} {getattr(module, '__version__', 'unknown')}")
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages or version_issues:
        error_msg = []
        if missing_packages:
            error_msg.append(f"Missing packages: {', '.join(missing_packages)}")
        if version_issues:
            error_msg.append(f"Version issues: {', '.join(version_issues)}")
        raise ImportError(" | ".join(error_msg))
    
    logger.info("All dependencies satisfied")

# Run dependency check on import
try:
    _check_dependencies()
except ImportError as e:
    logger.warning(f"Dependency check failed: {e}")
    logger.warning("Some functionality may not be available")