"""
VeritasFinancial - Banking Fraud Detection System
Test Suite Initialization Module

This module initializes the test suite for the VeritasFinancial fraud detection system.
It provides common testing utilities, fixtures, and configuration settings used across
all test modules.

The test suite follows industry best practices for data science testing:
- Unit tests for individual components
- Integration tests for pipelines
- Property-based testing for data validation
- Performance tests for model evaluation
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
import random
import tempfile
import shutil
import json
import yaml
from pathlib import Path

# Add the project root to Python path for imports
# This ensures we can import modules from the src directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# Import common testing utilities
from src.utils.logger import setup_test_logger
from src.utils.config_manager import ConfigManager
from src.data_preprocessing.pipelines.preprocessing_pipeline import FraudPreprocessingPipeline

# Set up test logging
# Test logger helps in debugging test failures
logger = setup_test_logger(__name__)

# Test configuration constants
# These control the behavior of the test suite
TEST_SEED = 42  # Fixed seed for reproducible tests
TEST_DATA_SIZE = 1000  # Number of samples in synthetic test data
TEST_BATCH_SIZE = 32  # Batch size for model tests
TEST_EPOCHS = 2  # Number of epochs for quick model tests
TEST_LEARNING_RATE = 0.001  # Learning rate for model tests
TEST_TOLERANCE = 1e-6  # Numerical tolerance for assertions
TEST_FRAUD_RATIO = 0.1  # Ratio of fraud cases in synthetic data (10% is realistic for fraud detection)

# Feature names for testing
# These match the actual feature names used in production
TEST_FEATURES = {
    'numerical': [
        'amount', 
        'customer_age', 
        'account_balance', 
        'credit_score',
        'transaction_velocity_1h',
        'avg_transaction_amount_7d',
        'std_transaction_amount_7d',
        'distance_from_home',
        'device_risk_score',
        'ip_risk_score'
    ],
    'categorical': [
        'merchant_category',
        'country_code',
        'device_type',
        'transaction_type',
        'currency',
        'customer_segment'
    ],
    'temporal': [
        'hour_of_day',
        'day_of_week',
        'month',
        'is_weekend',
        'time_since_last_transaction'
    ],
    'target': ['is_fraud']
}

# Test configuration dictionary
TEST_CONFIG = {
    'random_seed': TEST_SEED,
    'data': {
        'train_size': 0.7,
        'val_size': 0.15,
        'test_size': 0.15,
        'fraud_ratio': TEST_FRAUD_RATIO
    },
    'preprocessing': {
        'missing_threshold': 0.3,
        'outlier_method': 'iqr',
        'iqr_multiplier': 1.5,
        'scaling_method': 'robust'
    },
    'features': TEST_FEATURES,
    'model': {
        'type': 'xgboost',
        'params': {
            'max_depth': 6,
            'learning_rate': TEST_LEARNING_RATE,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': (1 - TEST_FRAUD_RATIO) / TEST_FRAUD_RATIO  # Handle imbalance
        }
    }
}


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """
    Pytest configuration hook.
    This runs once when pytest starts to set up the test environment.
    
    Args:
        config: Pytest configuration object
    """
    # Register custom markers for test categorization
    config.addinivalue_line(
        "markers",
        "unit: Mark test as unit test (fast, isolated component tests)"
    )
    config.addinivalue_line(
        "markers",
        "integration: Mark test as integration test (tests multiple components)"
    )
    config.addinivalue_line(
        "markers",
        "slow: Mark test as slow (runs longer, skip with -m 'not slow')"
    )
    config.addinivalue_line(
        "markers",
        "gpu: Mark test as requiring GPU (skip if no GPU available)"
    )
    config.addinivalue_line(
        "markers",
        "data_dependent: Mark test as depending on external data"
    )
    
    # Set up test environment variables
    os.environ['TESTING'] = 'True'
    os.environ['TEST_DATA_PATH'] = str(Path(PROJECT_ROOT) / 'tests' / 'test_data')
    
    logger.info("Pytest configured for VeritasFinancial test suite")


# =============================================================================
# Fixture Factory Functions
# =============================================================================

def create_synthetic_transaction_data(
    n_samples: int = TEST_DATA_SIZE,
    fraud_ratio: float = TEST_FRAUD_RATIO,
    seed: int = TEST_SEED
) -> pd.DataFrame:
    """
    Create synthetic transaction data for testing.
    
    This function generates realistic-looking transaction data that mimics
    real banking transactions. It's crucial for testing without accessing
    sensitive production data.
    
    Args:
        n_samples: Number of samples to generate
        fraud_ratio: Proportion of fraudulent transactions (0-1)
        seed: Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Synthetic transaction data
        
    Example:
        >>> df = create_synthetic_transaction_data(1000, 0.1)
        >>> len(df)
        1000
        >>> df['is_fraud'].mean()
        0.1
    """
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    
    # Generate timestamps over last 30 days
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)
    timestamps = [
        start_time + timedelta(
            seconds=random.randint(0, int((end_time - start_time).total_seconds()))
        )
        for _ in range(n_samples)
    ]
    
    # Create base DataFrame
    df = pd.DataFrame({
        # Transaction identifiers
        'transaction_id': [f'TXN_{i:08d}' for i in range(n_samples)],
        'customer_id': [f'CUST_{random.randint(1, 1000):05d}' for _ in range(n_samples)],
        'merchant_id': [f'MERCH_{random.randint(1, 500):05d}' for _ in range(n_samples)],
        'device_id': [f'DEV_{random.randint(1, 2000):05d}' for _ in range(n_samples)],
        
        # Transaction details
        'amount': np.random.lognormal(mean=4, sigma=1.5, size=n_samples),
        'currency': np.random.choice(['USD', 'EUR', 'GBP', 'JPY'], n_samples),
        'transaction_time': timestamps,
        'transaction_type': np.random.choice(['purchase', 'transfer', 'withdrawal', 'payment'], n_samples),
        
        # Location data
        'country': np.random.choice(['US', 'UK', 'DE', 'FR', 'JP', 'CA'], n_samples),
        'city': np.random.choice(['New York', 'London', 'Berlin', 'Paris', 'Tokyo', 'Toronto'], n_samples),
        'ip_address': [f'192.168.{random.randint(1, 255)}.{random.randint(1, 255)}' for _ in range(n_samples)],
        
        # Customer data
        'customer_age': np.random.randint(18, 80, n_samples),
        'account_balance': np.random.lognormal(mean=8, sigma=2, size=n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'account_age_days': np.random.randint(1, 3650, n_samples),
        
        # Device data
        'device_type': np.random.choice(['mobile', 'tablet', 'desktop', 'unknown'], n_samples),
        'os_type': np.random.choice(['iOS', 'Android', 'Windows', 'macOS', 'Linux'], n_samples),
        
        # Merchant data
        'merchant_category': np.random.choice(['retail', 'travel', 'entertainment', 'utilities', 'food'], n_samples),
        
        # Target variable (initialize as 0 for all)
        'is_fraud': 0
    })
    
    # Introduce fraud patterns
    # Fraudulent transactions are not random - they have specific patterns
    n_fraud = int(n_samples * fraud_ratio)
    fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
    
    for idx in fraud_indices:
        # Fraud patterns:
        # 1. Higher amounts
        df.loc[idx, 'amount'] *= np.random.uniform(2, 10)
        
        # 2. Unusual times (late night)
        hour = pd.to_datetime(df.loc[idx, 'transaction_time']).hour
        if hour not in [0, 1, 2, 3, 4]:  # Force to late night
            df.loc[idx, 'transaction_time'] = df.loc[idx, 'transaction_time'].replace(
                hour=np.random.choice([0, 1, 2, 3, 4])
            )
        
        # 3. Unusual locations
        if random.random() > 0.3:
            df.loc[idx, 'country'] = np.random.choice(['RU', 'CN', 'NG', 'BR', 'IN'])
        
        # 4. Unusual amounts (round numbers)
        if random.random() > 0.5:
            df.loc[idx, 'amount'] = round(df.loc[idx, 'amount'] / 100) * 100
        
        # 5. New device
        df.loc[idx, 'device_id'] = f'DEV_NEW_{random.randint(10000, 99999)}'
        
        # Set fraud flag
        df.loc[idx, 'is_fraud'] = 1
    
    # Add some missing values to test robustness (5% missing in some columns)
    for col in ['device_type', 'os_type', 'city']:
        missing_idx = np.random.choice(n_samples, int(n_samples * 0.05), replace=False)
        df.loc[missing_idx, col] = np.nan
    
    logger.info(f"Created synthetic dataset with {n_samples} samples, {n_fraud} fraud cases")
    
    return df


def create_test_config_file(temp_dir: str) -> str:
    """
    Create a test configuration file.
    
    Args:
        temp_dir: Temporary directory path
        
    Returns:
        str: Path to created config file
    """
    config_path = os.path.join(temp_dir, 'test_config.yaml')
    
    with open(config_path, 'w') as f:
        yaml.dump(TEST_CONFIG, f, default_flow_style=False)
    
    return config_path


def create_test_feature_store(temp_dir: str) -> str:
    """
    Create a test feature store directory with sample features.
    
    Args:
        temp_dir: Temporary directory path
        
    Returns:
        str: Path to feature store
    """
    feature_store_path = os.path.join(temp_dir, 'feature_store')
    os.makedirs(feature_store_path, exist_ok=True)
    
    # Create sample feature files
    df = create_synthetic_transaction_data(500)
    
    # Save features in parquet format
    df.to_parquet(os.path.join(feature_store_path, 'transaction_features.parquet'))
    
    return feature_store_path


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def test_data_path() -> str:
    """
    Fixture providing path to test data directory.
    Scope: session - created once per test session
    
    Returns:
        str: Path to test data directory
    """
    return os.path.join(PROJECT_ROOT, 'tests', 'test_data')


@pytest.fixture(scope="function")
def synthetic_transaction_data() -> pd.DataFrame:
    """
    Fixture providing synthetic transaction data for testing.
    Scope: function - new data for each test function
    
    Returns:
        pd.DataFrame: Synthetic transaction data
    """
    return create_synthetic_transaction_data()


@pytest.fixture(scope="function")
def synthetic_fraud_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Fixture providing features and target separately.
    
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features and target
    """
    df = create_synthetic_transaction_data()
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    return X, y


@pytest.fixture(scope="function")
def train_val_test_data() -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Fixture providing train/validation/test splits.
    
    Returns:
        Dict with keys 'train', 'val', 'test' containing (X, y) tuples
    """
    from sklearn.model_selection import train_test_split
    
    # Generate data
    df = create_synthetic_transaction_data(5000)
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    
    # Split into train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=TEST_SEED, stratify=y
    )
    
    # Split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=TEST_SEED, stratify=y_temp
    )
    
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }


@pytest.fixture(scope="function")
def preprocessed_data(train_val_test_data, preprocessing_pipeline):
    """
    Fixture providing preprocessed data.
    
    Args:
        train_val_test_data: Train/val/test data fixture
        preprocessing_pipeline: Preprocessing pipeline fixture
        
    Returns:
        Dict with preprocessed data
    """
    preprocessed = {}
    
    for split_name, (X, y) in train_val_test_data.items():
        # Fit on training data only
        if split_name == 'train':
            X_processed = preprocessing_pipeline.fit_transform(X)
        else:
            X_processed = preprocessing_pipeline.transform(X)
        
        preprocessed[split_name] = (X_processed, y)
    
    return preprocessed


@pytest.fixture(scope="function")
def preprocessing_pipeline():
    """
    Fixture providing a preprocessing pipeline instance.
    
    Returns:
        FraudPreprocessingPipeline: Configured pipeline
    """
    pipeline = FraudPreprocessingPipeline()
    
    # Add preprocessing steps
    pipeline.add_step('missing_handler', MissingValueHandler(threshold=0.3))
    pipeline.add_step('outlier_handler', OutlierHandler(method='iqr'))
    pipeline.add_step('scaler', RobustScaler())
    
    return pipeline


@pytest.fixture(scope="function")
def config_manager():
    """
    Fixture providing a config manager with test configuration.
    
    Returns:
        ConfigManager: Configured config manager
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = create_test_config_file(temp_dir)
        manager = ConfigManager(config_path)
        yield manager


@pytest.fixture(scope="function")
def temp_workspace():
    """
    Fixture providing a temporary workspace directory.
    Automatically cleaned up after test.
    
    Returns:
        str: Path to temporary workspace
    """
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="function")
def feature_store(temp_workspace):
    """
    Fixture providing a feature store path with sample features.
    
    Args:
        temp_workspace: Temporary workspace fixture
        
    Returns:
        str: Path to feature store
    """
    return create_test_feature_store(temp_workspace)


@pytest.fixture(scope="session")
def gpu_available() -> bool:
    """
    Fixture checking if GPU is available for testing.
    
    Returns:
        bool: True if GPU available
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# =============================================================================
# Custom Assertions and Helper Functions
# =============================================================================

def assert_dataframes_equal(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    check_dtypes: bool = True,
    rtol: float = TEST_TOLERANCE,
    atol: float = TEST_TOLERANCE
):
    """
    Custom assertion for DataFrame equality with tolerance.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        check_dtypes: Whether to check data types
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Raises:
        AssertionError: If DataFrames are not equal
    """
    # Check shape
    assert df1.shape == df2.shape, f"Shapes differ: {df1.shape} vs {df2.shape}"
    
    # Check columns
    assert set(df1.columns) == set(df2.columns), f"Columns differ: {set(df1.columns)} vs {set(df2.columns)}"
    
    # Check dtypes if requested
    if check_dtypes:
        for col in df1.columns:
            assert df1[col].dtype == df2[col].dtype, f"dtype differs for {col}: {df1[col].dtype} vs {df2[col].dtype}"
    
    # Check numeric columns with tolerance
    numeric_cols = df1.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        np.testing.assert_allclose(
            df1[col].values, df2[col].values,
            rtol=rtol, atol=atol,
            err_msg=f"Values differ for column {col}"
        )
    
    # Check non-numeric columns exactly
    non_numeric_cols = df1.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        # Handle NaN values
        mask1 = df1[col].isna()
        mask2 = df2[col].isna()
        assert (mask1 == mask2).all(), f"NaN pattern differs for {col}"
        
        # Check non-NaN values
        assert (df1[col][~mask1] == df2[col][~mask2]).all(), f"Values differ for column {col}"


def assert_model_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    min_precision: float = 0.7,
    min_recall: float = 0.5,
    min_f1: float = 0.6,
    min_auc: float = 0.8
):
    """
    Custom assertion for model performance metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        min_precision: Minimum acceptable precision
        min_recall: Minimum acceptable recall
        min_f1: Minimum acceptable F1 score
        min_auc: Minimum acceptable AUC-ROC
        
    Raises:
        AssertionError: If performance below thresholds
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Assert minimum performance
    assert precision >= min_precision, f"Precision {precision:.3f} < {min_precision}"
    assert recall >= min_recall, f"Recall {recall:.3f} < {min_recall}"
    assert f1 >= min_f1, f"F1 {f1:.3f} < {min_f1}"
    
    # Check AUC if probabilities provided
    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob)
        assert auc >= min_auc, f"AUC {auc:.3f} < {min_auc}"


def assert_feature_importance(
    importance_dict: Dict[str, float],
    top_features: List[str],
    min_importance: float = 0.01
):
    """
    Custom assertion for feature importance.
    
    Args:
        importance_dict: Dictionary mapping features to importance scores
        top_features: Expected top features
        min_importance: Minimum importance for any feature
        
    Raises:
        AssertionError: If importance criteria not met
    """
    # Check that all features have non-negative importance
    for feature, importance in importance_dict.items():
        assert importance >= 0, f"Feature {feature} has negative importance {importance}"
    
    # Check that top features are present with reasonable importance
    for feature in top_features:
        assert feature in importance_dict, f"Top feature {feature} not in importance dict"
        assert importance_dict[feature] >= min_importance, f"Feature {feature} importance {importance_dict[feature]} < {min_importance}"


def assert_data_quality(
    df: pd.DataFrame,
    max_missing_ratio: float = 0.1,
    max_infinite_ratio: float = 0.01,
    allowed_columns: Optional[List[str]] = None
):
    """
    Custom assertion for data quality.
    
    Args:
        df: DataFrame to check
        max_missing_ratio: Maximum allowed ratio of missing values
        max_infinite_ratio: Maximum allowed ratio of infinite values
        allowed_columns: List of allowed column names
        
    Raises:
        AssertionError: If data quality checks fail
    """
    # Check column names if specified
    if allowed_columns:
        extra_cols = set(df.columns) - set(allowed_columns)
        assert not extra_cols, f"Unexpected columns: {extra_cols}"
    
    # Check missing values
    missing_ratios = df.isnull().sum() / len(df)
    for col, ratio in missing_ratios.items():
        assert ratio <= max_missing_ratio, f"Column {col} has {ratio:.1%} missing values > {max_missing_ratio:.1%}"
    
    # Check infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        inf_ratio = np.isinf(df[col]).sum() / len(df)
        assert inf_ratio <= max_infinite_ratio, f"Column {col} has {inf_ratio:.1%} infinite values > {max_infinite_ratio:.1%}"


def assert_pipeline_consistency(
    pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
):
    """
    Custom assertion for pipeline consistency.
    Ensures that transform works after fit and doesn't modify input.
    
    Args:
        pipeline: Fitted pipeline
        X_train: Training data
        X_test: Test data
        
    Raises:
        AssertionError: If pipeline behavior is inconsistent
    """
    # Save original data
    X_train_original = X_train.copy()
    X_test_original = X_test.copy()
    
    # Transform data
    X_train_transformed = pipeline.transform(X_train)
    X_test_transformed = pipeline.transform(X_test)
    
    # Check that original data wasn't modified
    pd.testing.assert_frame_equal(X_train, X_train_original)
    pd.testing.assert_frame_equal(X_test, X_test_original)
    
    # Check that transform produces consistent output shape
    assert X_train_transformed.shape[0] == X_train.shape[0]
    assert X_test_transformed.shape[0] == X_test.shape[0]
    
    # Check that all transformed values are finite
    assert np.isfinite(X_train_transformed).all().all()
    assert np.isfinite(X_test_transformed).all().all()


# =============================================================================
# Test Data Generation Utilities
# =============================================================================

class TestDataGenerator:
    """
    Utility class for generating test data with specific patterns.
    Useful for testing edge cases and specific fraud scenarios.
    """
    
    def __init__(self, seed: int = TEST_SEED):
        """
        Initialize generator with random seed.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
    
    def generate_high_amount_fraud(self, n_samples: int = 100) -> pd.DataFrame:
        """
        Generate fraud cases with high transaction amounts.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            pd.DataFrame: Fraud data
        """
        df = create_synthetic_transaction_data(n_samples, fraud_ratio=1.0, seed=self.seed)
        
        # Make amounts extremely high
        df['amount'] = df['amount'] * np.random.uniform(10, 100, n_samples)
        
        return df
    
    def generate_time_anomaly_fraud(self, n_samples: int = 100) -> pd.DataFrame:
        """
        Generate fraud cases with unusual transaction times.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            pd.DataFrame: Fraud data
        """
        df = create_synthetic_transaction_data(n_samples, fraud_ratio=1.0, seed=self.seed)
        
        # Set all transactions to unusual hours (3-5 AM)
        for i in range(n_samples):
            df.loc[i, 'transaction_time'] = df.loc[i, 'transaction_time'].replace(
                hour=np.random.choice([3, 4, 5])
            )
        
        return df
    
    def generate_location_anomaly_fraud(self, n_samples: int = 100) -> pd.DataFrame:
        """
        Generate fraud cases with unusual locations.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            pd.DataFrame: Fraud data
        """
        df = create_synthetic_transaction_data(n_samples, fraud_ratio=1.0, seed=self.seed)
        
        # Set all transactions to high-risk countries
        df['country'] = np.random.choice(['RU', 'CN', 'NG', 'BR', 'IN', 'VE'], n_samples)
        
        return df
    
    def generate_new_device_fraud(self, n_samples: int = 100) -> pd.DataFrame:
        """
        Generate fraud cases with new, never-seen-before devices.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            pd.DataFrame: Fraud data
        """
        df = create_synthetic_transaction_data(n_samples, fraud_ratio=1.0, seed=self.seed)
        
        # Generate new device IDs
        df['device_id'] = [f'DEV_NEW_{random.randint(100000, 999999)}' for _ in range(n_samples)]
        
        return df
    
    def generate_mixed_fraud_patterns(self, n_samples: int = 100) -> pd.DataFrame:
        """
        Generate fraud cases with mixed patterns.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            pd.DataFrame: Fraud data
        """
        dfs = []
        
        # Generate different fraud patterns
        patterns = [
            self.generate_high_amount_fraud(n_samples // 4),
            self.generate_time_anomaly_fraud(n_samples // 4),
            self.generate_location_anomaly_fraud(n_samples // 4),
            self.generate_new_device_fraud(n_samples // 4)
        ]
        
        # Combine all patterns
        df = pd.concat(patterns, ignore_index=True)
        
        # Add noise
        noise_df = create_synthetic_transaction_data(len(df), fraud_ratio=0, seed=self.seed)
        for col in noise_df.columns:
            if col in df.columns and col not in ['transaction_id', 'customer_id']:
                # Add small amount of noise to numeric columns
                if col in df.select_dtypes(include=[np.number]).columns:
                    df[col] += np.random.normal(0, df[col].std() * 0.01, len(df))
        
        return df


# =============================================================================
# Module Level Variables
# =============================================================================

# Create global test data generator
test_data_generator = TestDataGenerator()

# Export commonly used functions and classes
__all__ = [
    # Fixtures
    'synthetic_transaction_data',
    'synthetic_fraud_data',
    'train_val_test_data',
    'preprocessing_pipeline',
    'preprocessed_data',
    'config_manager',
    'temp_workspace',
    'feature_store',
    'gpu_available',
    
    # Test data generation
    'TestDataGenerator',
    'test_data_generator',
    'create_synthetic_transaction_data',
    
    # Custom assertions
    'assert_dataframes_equal',
    'assert_model_performance',
    'assert_feature_importance',
    'assert_data_quality',
    'assert_pipeline_consistency',
    
    # Constants
    'TEST_SEED',
    'TEST_DATA_SIZE',
    'TEST_BATCH_SIZE',
    'TEST_EPOCHS',
    'TEST_LEARNING_RATE',
    'TEST_TOLERANCE',
    'TEST_FRAUD_RATIO',
    'TEST_FEATURES',
    'TEST_CONFIG',
    
    # Path utilities
    'PROJECT_ROOT'
]

# Log module initialization
logger.info("VeritasFinancial test suite initialized successfully")
logger.info(f"Test seed: {TEST_SEED}")
logger.info(f"Test data size: {TEST_DATA_SIZE}")
logger.info(f"Test fraud ratio: {TEST_FRAUD_RATIO}")