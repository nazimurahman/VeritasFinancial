# tests/test_data_preprocessing.py
"""
Unit tests for data preprocessing module.
Tests cover data cleaning, transformation, validation, and pipeline execution.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing.cleaners.transaction_cleaner import TransactionCleaner
from src.data_preprocessing.cleaners.customer_cleaner import CustomerCleaner
from src.data_preprocessing.cleaners.device_cleaner import DeviceCleaner
from src.data_preprocessing.transformers.categorical_encoder import CategoricalEncoder
from src.data_preprocessing.transformers.numerical_scaler import NumericalScaler
from src.data_preprocessing.transformers.datetime_processor import DateTimeProcessor
from src.data_preprocessing.handlers.missing_values import MissingValueHandler
from src.data_preprocessing.handlers.outliers import OutlierHandler
from src.data_preprocessing.handlers.imbalance import ImbalanceHandler
from src.data_preprocessing.pipelines.preprocessing_pipeline import PreprocessingPipeline


class TestTransactionCleaner:
    """
    Test suite for TransactionCleaner class.
    Tests cleaning operations specific to transaction data.
    """
    
    @pytest.fixture
    def sample_transaction_data(self):
        """
        Fixture providing sample transaction data for testing.
        Returns a DataFrame with various transaction scenarios.
        """
        return pd.DataFrame({
            'transaction_id': ['T1', 'T2', 'T3', 'T4', 'T5'],
            'customer_id': ['C1', 'C2', 'C1', 'C3', 'C2'],
            'amount': [100.50, -50.00, 1000.00, 25.00, 999999.99],
            'currency': ['USD', 'USD', 'EUR', 'GBP', 'USD'],
            'transaction_time': [
                datetime.now() - timedelta(hours=2),
                datetime.now() - timedelta(hours=1),
                datetime.now() - timedelta(minutes=30),
                datetime.now() - timedelta(minutes=15),
                datetime.now()
            ],
            'merchant_id': ['M1', 'M2', 'M1', 'M3', 'M4'],
            'merchant_category': ['retail', 'travel', 'retail', 'food', 'luxury'],
            'country': ['US', 'US', 'FR', 'GB', 'US'],
            'city': ['NYC', 'LA', 'Paris', 'London', 'SF'],
            'device_id': ['D1', 'D2', 'D1', 'D3', 'D4'],
            'device_type': ['mobile', 'desktop', 'mobile', 'tablet', 'mobile'],
            'ip_address': ['192.168.1.1', '10.0.0.1', '172.16.0.1', '192.168.1.2', '10.0.0.2'],
            'is_fraud': [0, 1, 0, 0, 1]
        })
    
    @pytest.fixture
    def cleaner_with_duplicates(self):
        """
        Fixture providing transaction data with duplicates for testing deduplication.
        """
        return TransactionCleaner(config={'remove_duplicates': True, 'dedup_subset': ['transaction_id']})
    
    def test_initialization(self):
        """
        Test TransactionCleaner initialization with various configs.
        Verifies that the cleaner properly sets up with default and custom configurations.
        """
        # Test with default config
        cleaner = TransactionCleaner()
        assert cleaner.config is not None
        assert 'remove_duplicates' in cleaner.config
        assert 'amount_min' in cleaner.config
        assert 'amount_max' in cleaner.config
        
        # Test with custom config
        custom_config = {
            'remove_duplicates': True,
            'amount_min': 0.01,
            'amount_max': 100000,
            'dedup_subset': ['transaction_id', 'customer_id']
        }
        cleaner = TransactionCleaner(config=custom_config)
        assert cleaner.config['amount_min'] == 0.01
        assert cleaner.config['amount_max'] == 100000
        assert cleaner.config['dedup_subset'] == ['transaction_id', 'customer_id']
    
    def test_remove_duplicates(self, sample_transaction_data, cleaner_with_duplicates):
        """
        Test duplicate removal functionality.
        Verifies that duplicates are correctly identified and removed.
        """
        # Create data with duplicates
        data_with_dupes = pd.concat([sample_transaction_data, sample_transaction_data.iloc[[0]]])
        
        # Apply cleaning
        cleaned_data = cleaner_with_duplicates._remove_duplicates(data_with_dupes)
        
        # Assertions
        assert len(cleaned_data) == len(sample_transaction_data)
        assert cleaned_data['transaction_id'].is_unique
        assert cleaned_data.equals(sample_transaction_data)
    
    def test_validate_amount(self, sample_transaction_data):
        """
        Test amount validation functionality.
        Verifies that amount values are within acceptable ranges.
        """
        cleaner = TransactionCleaner(config={'amount_min': 0.01, 'amount_max': 100000})
        
        # Test negative amount (should be flagged)
        assert not cleaner._validate_amount(-50.00)
        
        # Test amount below minimum
        assert not cleaner._validate_amount(0.001)
        
        # Test amount above maximum
        assert not cleaner._validate_amount(999999.99)
        
        # Test valid amount
        assert cleaner._validate_amount(100.50)
        
        # Test edge cases
        assert cleaner._validate_amount(0.01)  # Minimum boundary
        assert cleaner._validate_amount(100000)  # Maximum boundary
    
    def test_standardize_formats(self, sample_transaction_data):
        """
        Test format standardization for various fields.
        Verifies that currency, country codes, and other fields are properly standardized.
        """
        cleaner = TransactionCleaner()
        
        # Add non-standard formats
        data = sample_transaction_data.copy()
        data.loc[0, 'currency'] = 'usd'  # Lowercase
        data.loc[1, 'currency'] = '$'    # Symbol
        data.loc[2, 'country'] = 'FRA'   # 3-letter code
        data.loc[3, 'country'] = 'UK'    # Non-standard
        
        standardized = cleaner._standardize_formats(data)
        
        # Check currency standardization
        assert standardized.loc[0, 'currency'] == 'USD'
        assert standardized.loc[1, 'currency'] == 'USD'  # $ should map to USD
        
        # Check country standardization
        assert standardized.loc[2, 'country'] == 'FR'
        assert standardized.loc[3, 'country'] == 'GB'
    
    def test_full_clean_pipeline(self, sample_transaction_data):
        """
        Test the complete cleaning pipeline.
        Verifies that all cleaning steps work together correctly.
        """
        cleaner = TransactionCleaner(config={
            'remove_duplicates': True,
            'amount_min': 0.01,
            'amount_max': 100000,
            'standardize_currency': True,
            'standardize_country': True
        })
        
        # Add issues to clean
        dirty_data = sample_transaction_data.copy()
        dirty_data.loc[0, 'currency'] = 'usd'
        dirty_data.loc[1, 'country'] = 'FRA'
        
        # Add duplicate
        dirty_data = pd.concat([dirty_data, dirty_data.iloc[[0]]])
        
        # Clean data
        cleaned_data = cleaner.clean(dirty_data)
        
        # Verify cleaning
        assert len(cleaned_data) == len(sample_transaction_data)  # Duplicate removed
        assert cleaned_data.loc[0, 'currency'] == 'USD'  # Currency standardized
        assert cleaned_data.loc[1, 'country'] == 'FR'    # Country standardized
        assert not any(cleaned_data['amount'] < 0.01)    # No invalid amounts


class TestMissingValueHandler:
    """
    Test suite for MissingValueHandler class.
    Tests various strategies for handling missing values.
    """
    
    @pytest.fixture
    def data_with_missing(self):
        """
        Fixture providing data with various missing value patterns.
        """
        np.random.seed(42)
        n_samples = 100
        
        data = pd.DataFrame({
            'amount': np.random.normal(100, 30, n_samples),
            'customer_age': np.random.randint(18, 80, n_samples),
            'credit_score': np.random.normal(700, 50, n_samples),
            'transaction_category': np.random.choice(['A', 'B', 'C'], n_samples),
            'is_fraud': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
        })
        
        # Introduce missing values
        data.loc[0:10, 'amount'] = np.nan
        data.loc[5:15, 'customer_age'] = np.nan
        data.loc[20:30, 'credit_score'] = np.nan
        data.loc[15:25, 'transaction_category'] = np.nan
        
        return data
    
    def test_initialization(self):
        """
        Test MissingValueHandler initialization with different strategies.
        """
        # Test mean strategy
        handler = MissingValueHandler(strategy='mean')
        assert handler.strategy == 'mean'
        assert handler.fitted == False
        
        # Test median strategy
        handler = MissingValueHandler(strategy='median')
        assert handler.strategy == 'median'
        
        # Test mode strategy
        handler = MissingValueHandler(strategy='mode')
        assert handler.strategy == 'mode'
        
        # Test constant strategy
        handler = MissingValueHandler(strategy='constant', fill_value=0)
        assert handler.strategy == 'constant'
        assert handler.fill_value == 0
        
        # Test interpolation strategy
        handler = MissingValueHandler(strategy='interpolate', method='linear')
        assert handler.strategy == 'interpolate'
        assert handler.method == 'linear'
    
    def test_mean_imputation(self, data_with_missing):
        """
        Test mean imputation strategy.
        Verifies that missing values are replaced with column means.
        """
        handler = MissingValueHandler(strategy='mean')
        
        # Fit and transform
        handler.fit(data_with_missing)
        imputed_data = handler.transform(data_with_missing)
        
        # Check that no missing values remain
        assert imputed_data.isnull().sum().sum() == 0
        
        # Check numerical columns
        for col in ['amount', 'customer_age', 'credit_score']:
            if col in imputed_data.select_dtypes(include=[np.number]).columns:
                # Check that imputed values match column means
                col_mean = data_with_missing[col].mean()
                assert abs(imputed_data[col].iloc[0] - col_mean) < 0.1
    
    def test_median_imputation(self, data_with_missing):
        """
        Test median imputation strategy.
        Verifies that missing values are replaced with column medians.
        """
        handler = MissingValueHandler(strategy='median')
        
        # Fit and transform
        handler.fit(data_with_missing)
        imputed_data = handler.transform(data_with_missing)
        
        # Check numerical columns
        for col in ['amount', 'customer_age', 'credit_score']:
            if col in imputed_data.select_dtypes(include=[np.number]).columns:
                col_median = data_with_missing[col].median()
                assert abs(imputed_data[col].iloc[0] - col_median) < 0.1
    
    def test_mode_imputation(self, data_with_missing):
        """
        Test mode imputation strategy for categorical data.
        """
        handler = MissingValueHandler(strategy='mode')
        
        # Fit and transform
        handler.fit(data_with_missing)
        imputed_data = handler.transform(data_with_missing)
        
        # Check categorical column
        if 'transaction_category' in imputed_data.columns:
            # Mode should be 'A', 'B', or 'C'
            mode_value = data_with_missing['transaction_category'].mode()[0]
            assert imputed_data['transaction_category'].iloc[15] == mode_value
    
    def test_constant_imputation(self, data_with_missing):
        """
        Test constant value imputation.
        """
        fill_value = -999
        handler = MissingValueHandler(strategy='constant', fill_value=fill_value)
        
        # Fit and transform
        handler.fit(data_with_missing)
        imputed_data = handler.transform(data_with_missing)
        
        # Check that missing values replaced with constant
        missing_mask = data_with_missing['amount'].isnull()
        assert (imputed_data.loc[missing_mask, 'amount'] == fill_value).all()
    
    def test_interpolation_imputation(self, data_with_missing):
        """
        Test interpolation imputation for time-series like data.
        """
        # Sort data to simulate time series
        data_with_missing = data_with_missing.sort_index()
        
        handler = MissingValueHandler(strategy='interpolate', method='linear')
        
        # Fit and transform
        handler.fit(data_with_missing)
        imputed_data = handler.transform(data_with_missing)
        
        # Check that values are interpolated between known points
        # This is harder to test directly, so we'll check no NaNs remain
        assert imputed_data.isnull().sum().sum() == 0
    
    def test_fit_transform_integration(self, data_with_missing):
        """
        Test fit_transform method for end-to-end missing value handling.
        """
        handler = MissingValueHandler(strategy='mean')
        
        # Use fit_transform directly
        imputed_data = handler.fit_transform(data_with_missing)
        
        # Verify handler is fitted
        assert handler.fitted == True
        
        # Verify no missing values
        assert imputed_data.isnull().sum().sum() == 0
        
        # Verify statistics were stored
        assert hasattr(handler, 'statistics_')
        assert 'amount' in handler.statistics_
        assert handler.statistics_['amount']['mean'] is not None
    
    def test_invalid_strategy(self):
        """
        Test that invalid strategies raise appropriate errors.
        """
        with pytest.raises(ValueError, match="Strategy must be one of"):
            MissingValueHandler(strategy='invalid_strategy')
    
    def test_transform_without_fit(self, data_with_missing):
        """
        Test that transform raises error if called before fit.
        """
        handler = MissingValueHandler(strategy='mean')
        
        with pytest.raises(ValueError, match="must be fitted before transform"):
            handler.transform(data_with_missing)


class TestOutlierHandler:
    """
    Test suite for OutlierHandler class.
    Tests various outlier detection and handling methods.
    """
    
    @pytest.fixture
    def data_with_outliers(self):
        """
        Fixture providing data with outliers for testing.
        """
        np.random.seed(42)
        n_samples = 1000
        
        # Generate normal data
        data = pd.DataFrame({
            'amount': np.random.normal(100, 20, n_samples),
            'transaction_count': np.random.poisson(5, n_samples),
            'customer_age': np.random.normal(40, 15, n_samples),
            'credit_score': np.random.normal(700, 50, n_samples)
        })
        
        # Add outliers
        data.loc[0, 'amount'] = 10000  # Extreme high value
        data.loc[1, 'amount'] = -1000  # Extreme low value
        data.loc[2, 'transaction_count'] = 100  # Unusually high
        data.loc[3, 'customer_age'] = 150  # Impossible age
        data.loc[4, 'credit_score'] = 50  # Extremely low credit score
        
        return data
    
    def test_initialization(self):
        """
        Test OutlierHandler initialization with different methods.
        """
        # Test IQR method
        handler = OutlierHandler(method='iqr', threshold=1.5)
        assert handler.method == 'iqr'
        assert handler.threshold == 1.5
        
        # Test Z-score method
        handler = OutlierHandler(method='zscore', threshold=3)
        assert handler.method == 'zscore'
        assert handler.threshold == 3
        
        # Test Isolation Forest method
        handler = OutlierHandler(method='isolation_forest', contamination=0.1)
        assert handler.method == 'isolation_forest'
        assert handler.contamination == 0.1
        
        # Test with custom action
        handler = OutlierHandler(method='iqr', action='cap')
        assert handler.action == 'cap'
    
    def test_iqr_outlier_detection(self, data_with_outliers):
        """
        Test IQR-based outlier detection.
        """
        handler = OutlierHandler(method='iqr', threshold=1.5, action='flag')
        
        # Fit and transform
        handler.fit(data_with_outliers)
        result = handler.transform(data_with_outliers)
        
        # Check that outliers are flagged
        assert 'outlier_flag' in result.columns
        
        # Known outliers should be flagged
        assert result.loc[0, 'outlier_flag'] == 1  # amount outlier
        assert result.loc[1, 'outlier_flag'] == 1  # amount outlier
        assert result.loc[3, 'outlier_flag'] == 1  # age outlier
        
        # Most normal points should not be flagged
        assert result.loc[10:, 'outlier_flag'].sum() < 50  # Less than 5% outliers
    
    def test_iqr_outlier_capping(self, data_with_outliers):
        """
        Test outlier capping with IQR method.
        """
        handler = OutlierHandler(method='iqr', threshold=1.5, action='cap')
        
        # Fit and transform
        handler.fit(data_with_outliers)
        result = handler.transform(data_with_outliers)
        
        # Calculate IQR bounds manually for amount
        q1 = data_with_outliers['amount'].quantile(0.25)
        q3 = data_with_outliers['amount'].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        lower_bound = q1 - 1.5 * iqr
        
        # Check that outliers are capped
        assert result.loc[0, 'amount'] <= upper_bound
        assert result.loc[1, 'amount'] >= lower_bound
        
        # Original values should be capped, not removed
        assert len(result) == len(data_with_outliers)
    
    def test_zscore_outlier_detection(self, data_with_outliers):
        """
        Test Z-score based outlier detection.
        """
        handler = OutlierHandler(method='zscore', threshold=3, action='flag')
        
        # Fit and transform
        handler.fit(data_with_outliers)
        result = handler.transform(data_with_outliers)
        
        # Calculate Z-scores for amount
        mean_amount = data_with_outliers['amount'].mean()
        std_amount = data_with_outliers['amount'].std()
        zscore_0 = abs((data_with_outliers.loc[0, 'amount'] - mean_amount) / std_amount)
        
        # Check that extreme outlier is flagged
        assert zscore_0 > 3
        assert result.loc[0, 'outlier_flag'] == 1
    
    def test_isolation_forest_detection(self, data_with_outliers):
        """
        Test Isolation Forest based outlier detection.
        """
        handler = OutlierHandler(method='isolation_forest', contamination=0.05, action='flag')
        
        # Fit and transform
        handler.fit(data_with_outliers)
        result = handler.transform(data_with_outliers)
        
        # Check that outliers are flagged
        assert 'outlier_flag' in result.columns
        
        # Approximately 5% of data should be flagged
        flagged_pct = result['outlier_flag'].mean()
        assert 0.03 <= flagged_pct <= 0.07
    
    def test_outlier_removal(self, data_with_outliers):
        """
        Test outlier removal action.
        """
        handler = OutlierHandler(method='iqr', threshold=1.5, action='remove')
        
        # Fit and transform
        handler.fit(data_with_outliers)
        result = handler.transform(data_with_outliers)
        
        # Check that outliers were removed
        assert len(result) < len(data_with_outliers)
        
        # Recalculate IQR bounds for result
        q1 = result['amount'].quantile(0.25)
        q3 = result['amount'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # All values should be within bounds
        assert (result['amount'] >= lower_bound).all()
        assert (result['amount'] <= upper_bound).all()
    
    def test_per_column_outlier_handling(self, data_with_outliers):
        """
        Test outlier handling with per-column thresholds.
        """
        handler = OutlierHandler(
            method='iqr',
            threshold={
                'amount': 3.0,  # More lenient for amount
                'customer_age': 1.5  # Stricter for age
            },
            action='flag'
        )
        
        # Fit and transform
        handler.fit(data_with_outliers)
        result = handler.transform(data_with_outliers)
        
        # Check column-specific thresholds were applied
        # Amount outlier might not be flagged with higher threshold
        # Age outlier should be flagged with lower threshold
        assert 'outlier_flag' in result.columns
    
    def test_invalid_method(self):
        """
        Test that invalid methods raise appropriate errors.
        """
        with pytest.raises(ValueError, match="Method must be one of"):
            OutlierHandler(method='invalid_method')
    
    def test_invalid_action(self):
        """
        Test that invalid actions raise appropriate errors.
        """
        handler = OutlierHandler(method='iqr', action='invalid')
        
        with pytest.raises(ValueError, match="Action must be one of"):
            handler._handle_outliers(pd.DataFrame(), {'outlier_flag': [1]})


class TestImbalanceHandler:
    """
    Test suite for ImbalanceHandler class.
    Tests various techniques for handling class imbalance.
    """
    
    @pytest.fixture
    def imbalanced_data(self):
        """
        Fixture providing imbalanced classification data.
        """
        np.random.seed(42)
        n_samples = 10000
        
        # Generate features
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.normal(0, 1, n_samples),
            'feature4': np.random.choice(['A', 'B', 'C'], n_samples)
        })
        
        # Generate imbalanced target (1% positive class)
        y = np.random.choice([0, 1], n_samples, p=[0.99, 0.01])
        
        return X, y
    
    def test_initialization(self):
        """
        Test ImbalanceHandler initialization with different techniques.
        """
        # Test SMOTE
        handler = ImbalanceHandler(technique='smote', random_state=42)
        assert handler.technique == 'smote'
        assert handler.random_state == 42
        
        # Test Random Under-sampling
        handler = ImbalanceHandler(technique='random_under', sampling_strategy=0.5)
        assert handler.technique == 'random_under'
        assert handler.sampling_strategy == 0.5
        
        # Test Random Over-sampling
        handler = ImbalanceHandler(technique='random_over', sampling_strategy=0.8)
        assert handler.technique == 'random_over'
        assert handler.sampling_strategy == 0.8
        
        # Test ADASYN
        handler = ImbalanceHandler(technique='adasyn', n_neighbors=5)
        assert handler.technique == 'adasyn'
        assert handler.n_neighbors == 5
    
    def test_smote_balancing(self, imbalanced_data):
        """
        Test SMOTE oversampling technique.
        """
        X, y = imbalanced_data
        handler = ImbalanceHandler(technique='smote', random_state=42)
        
        # Apply SMOTE
        X_resampled, y_resampled = handler.fit_resample(X, y)
        
        # Check that classes are balanced
        unique, counts = np.unique(y_resampled, return_counts=True)
        assert len(unique) == 2
        assert abs(counts[0] - counts[1]) / counts[0] < 0.1  # Within 10% balance
        
        # Check that minority class size increased
        original_minority = sum(y == 1)
        resampled_minority = sum(y_resampled == 1)
        assert resampled_minority > original_minority
        
        # Check that features preserve structure (roughly)
        assert X_resampled.shape[1] == X.shape[1]
    
    def test_random_under_sampling(self, imbalanced_data):
        """
        Test random under-sampling technique.
        """
        X, y = imbalanced_data
        target_ratio = 0.5  # Minority should be 50% of majority
        handler = ImbalanceHandler(technique='random_under', sampling_strategy=target_ratio)
        
        # Apply under-sampling
        X_resampled, y_resampled = handler.fit_resample(X, y)
        
        # Check that classes are balanced according to target ratio
        unique, counts = np.unique(y_resampled, return_counts=True)
        assert len(unique) == 2
        actual_ratio = counts[1] / counts[0]
        assert abs(actual_ratio - target_ratio) < 0.1
        
        # Check that majority class size decreased
        original_majority = sum(y == 0)
        resampled_majority = sum(y_resampled == 0)
        assert resampled_majority < original_majority
    
    def test_random_over_sampling(self, imbalanced_data):
        """
        Test random over-sampling technique.
        """
        X, y = imbalanced_data
        handler = ImbalanceHandler(technique='random_over', random_state=42)
        
        # Apply over-sampling
        X_resampled, y_resampled = handler.fit_resample(X, y)
        
        # Check that classes are balanced
        unique, counts = np.unique(y_resampled, return_counts=True)
        assert len(unique) == 2
        assert abs(counts[0] - counts[1]) / counts[0] < 0.1
        
        # Check that minority class size increased
        original_minority = sum(y == 1)
        resampled_minority = sum(y_resampled == 1)
        assert resampled_minority > original_minority
    
    def test_adasyn_balancing(self, imbalanced_data):
        """
        Test ADASYN adaptive synthetic sampling.
        """
        X, y = imbalanced_data
        handler = ImbalanceHandler(technique='adasyn', n_neighbors=5, random_state=42)
        
        # Apply ADASYN
        X_resampled, y_resampled = handler.fit_resample(X, y)
        
        # Check that classes are balanced
        unique, counts = np.unique(y_resampled, return_counts=True)
        assert len(unique) == 2
        assert abs(counts[0] - counts[1]) / counts[0] < 0.1
        
        # ADASYN should create synthetic samples
        assert len(X_resampled) > len(X)
    
    def test_combined_techniques(self, imbalanced_data):
        """
        Test combination of over and under sampling.
        """
        handler = ImbalanceHandler(
            technique='combined',
            over_ratio=0.5,
            under_ratio=0.8,
            random_state=42
        )
        
        # Apply combined sampling
        X_resampled, y_resampled = handler.fit_resample(imbalanced_data[0], imbalanced_data[1])
        
        # Check results
        unique, counts = np.unique(y_resampled, return_counts=True)
        assert len(unique) == 2
        assert counts[0] > 0 and counts[1] > 0
    
    def test_invalid_technique(self):
        """
        Test that invalid techniques raise appropriate errors.
        """
        with pytest.raises(ValueError, match="Technique must be one of"):
            ImbalanceHandler(technique='invalid_technique')
    
    def test_categorical_feature_handling(self, imbalanced_data):
        """
        Test that categorical features are handled properly.
        """
        X, y = imbalanced_data
        
        # X has a categorical column 'feature4'
        handler = ImbalanceHandler(technique='smote', random_state=42)
        
        # This should work - SMOTE should handle categorical via encoding
        X_resampled, y_resampled = handler.fit_resample(X, y)
        
        # Check that categorical column still exists
        assert 'feature4' in X_resampled.columns
        
        # Check that categories are preserved (some may be new from SMOTE)
        original_cats = set(X['feature4'].unique())
        new_cats = set(X_resampled['feature4'].unique())
        assert original_cats.intersection(new_cats) == original_cats


class TestPreprocessingPipeline:
    """
    Test suite for PreprocessingPipeline class.
    Tests the end-to-end preprocessing pipeline functionality.
    """
    
    @pytest.fixture
    def raw_data(self):
        """
        Fixture providing raw data for pipeline testing.
        """
        np.random.seed(42)
        n_samples = 1000
        
        return pd.DataFrame({
            'transaction_id': [f'T{i}' for i in range(n_samples)],
            'customer_id': [f'C{np.random.randint(1, 100)}' for _ in range(n_samples)],
            'amount': np.random.normal(100, 30, n_samples),
            'currency': np.random.choice(['USD', 'EUR', 'GBP', 'usd', 'eur'], n_samples),
            'transaction_time': [datetime.now() - timedelta(hours=np.random.randint(0, 720)) 
                                for _ in range(n_samples)],
            'merchant_category': np.random.choice(['retail', 'travel', 'food', 'luxury'], n_samples),
            'country': np.random.choice(['US', 'GB', 'FR', 'DE', 'usa', 'uk'], n_samples),
            'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], n_samples),
            'is_fraud': np.random.choice([0, 1], n_samples, p=[0.98, 0.02])
        })
    
    def test_pipeline_initialization(self):
        """
        Test pipeline initialization and configuration.
        """
        pipeline = PreprocessingPipeline(config={
            'cleaners': ['transaction', 'customer', 'device'],
            'handlers': ['missing', 'outliers', 'imbalance'],
            'transformers': ['categorical', 'numerical', 'datetime']
        })
        
        assert pipeline.config is not None
        assert len(pipeline.steps) == 0
        assert pipeline.fitted == False
    
    def test_add_step(self):
        """
        Test adding steps to pipeline.
        """
        pipeline = PreprocessingPipeline()
        
        # Add a cleaning step
        cleaner = TransactionCleaner()
        pipeline.add_step('clean_transactions', cleaner)
        
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0]['name'] == 'clean_transactions'
        assert pipeline.steps[0]['transformer'] == cleaner
    
    def test_build_default_pipeline(self, raw_data):
        """
        Test building default preprocessing pipeline.
        """
        pipeline = PreprocessingPipeline()
        
        # Build default pipeline
        pipeline.build_default_pipeline()
        
        assert len(pipeline.steps) > 0
        
        # Steps should include cleaning, handling, and transformation
        step_names = [step['name'] for step in pipeline.steps]
        assert any('clean' in name for name in step_names)
        assert any('missing' in name for name in step_names)
        assert any('outlier' in name for name in step_names)
        assert any('encode' in name for name in step_names)
    
    def test_fit_transform_pipeline(self, raw_data):
        """
        Test end-to-end fit_transform on pipeline.
        """
        pipeline = PreprocessingPipeline()
        
        # Build pipeline with specific steps for testing
        pipeline.add_step('clean', TransactionCleaner())
        pipeline.add_step('handle_missing', MissingValueHandler(strategy='mean'))
        pipeline.add_step('encode_categorical', CategoricalEncoder(encoding_type='label'))
        pipeline.add_step('scale_numerical', NumericalScaler(method='standard'))
        
        # Fit and transform
        processed_data = pipeline.fit_transform(raw_data)
        
        # Check that pipeline is fitted
        assert pipeline.fitted == True
        
        # Check that data was transformed
        assert processed_data is not None
        assert len(processed_data) > 0
        
        # Check that categorical columns were encoded (should be numeric now)
        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) > 0
    
    def test_pipeline_with_missing_data(self, raw_data):
        """
        Test pipeline handling of missing data.
        """
        # Introduce missing values
        raw_data.loc[0:50, 'amount'] = np.nan
        raw_data.loc[20:70, 'customer_id'] = np.nan
        
        pipeline = PreprocessingPipeline()
        
        # Add missing value handler
        pipeline.add_step('handle_missing', MissingValueHandler(strategy='mean'))
        
        # Process data
        processed_data = pipeline.fit_transform(raw_data)
        
        # Check no missing values remain
        assert processed_data.isnull().sum().sum() == 0
    
    def test_pipeline_with_outliers(self, raw_data):
        """
        Test pipeline handling of outliers.
        """
        # Add outliers
        raw_data.loc[0, 'amount'] = 1000000
        
        pipeline = PreprocessingPipeline()
        
        # Add outlier handler
        pipeline.add_step('handle_outliers', OutlierHandler(method='iqr', action='cap'))
        
        # Process data
        processed_data = pipeline.fit_transform(raw_data)
        
        # Check that outlier was capped
        max_amount = processed_data['amount'].max()
        assert max_amount < 1000000
    
    def test_pipeline_with_imbalance(self, raw_data):
        """
        Test pipeline handling of class imbalance.
        """
        # Separate features and target
        X = raw_data.drop('is_fraud', axis=1)
        y = raw_data['is_fraud']
        
        # This test would need a pipeline that handles target separately
        # For now, just verify that imbalance handler works with features and target
        handler = ImbalanceHandler(technique='smote')
        
        # Apply to features and target
        X_resampled, y_resampled = handler.fit_resample(X, y)
        
        # Check balance improved
        original_ratio = sum(y == 1) / len(y)
        new_ratio = sum(y_resampled == 1) / len(y_resampled)
        
        assert new_ratio > original_ratio
    
    def test_pipeline_save_load(self, raw_data, tmp_path):
        """
        Test saving and loading pipeline.
        """
        import joblib
        
        pipeline = PreprocessingPipeline()
        pipeline.build_default_pipeline()
        
        # Fit pipeline
        pipeline.fit_transform(raw_data)
        
        # Save pipeline
        save_path = tmp_path / "pipeline.pkl"
        joblib.dump(pipeline, save_path)
        
        # Load pipeline
        loaded_pipeline = joblib.load(save_path)
        
        # Check that loaded pipeline works
        assert loaded_pipeline.fitted == True
        assert len(loaded_pipeline.steps) == len(pipeline.steps)
    
    def test_pipeline_error_handling(self):
        """
        Test pipeline error handling for invalid operations.
        """
        pipeline = PreprocessingPipeline()
        
        # Transform without fit should raise error
        with pytest.raises(ValueError, match="must be fitted before transform"):
            pipeline.transform(pd.DataFrame())
        
        # Add invalid transformer
        class InvalidTransformer:
            def fit(self, X):
                raise ValueError("Invalid transformer")
        
        pipeline.add_step('invalid', InvalidTransformer())
        
        # Fit should fail gracefully
        with pytest.raises(Exception):
            pipeline.fit_transform(pd.DataFrame({'col': [1, 2, 3]}))
    
    def test_pipeline_step_dependencies(self, raw_data):
        """
        Test step dependencies in pipeline.
        """
        pipeline = PreprocessingPipeline()
        
        # Add steps with dependencies
        pipeline.add_step('clean', TransactionCleaner())
        pipeline.add_step('encode_categorical', CategoricalEncoder(), depends_on=['clean'])
        pipeline.add_step('scale', NumericalScaler(), depends_on=['clean', 'encode_categorical'])
        
        # This should work - dependencies are satisfied
        processed_data = pipeline.fit_transform(raw_data)
        assert processed_data is not None
        
        # Try adding step with unmet dependency
        pipeline = PreprocessingPipeline()
        pipeline.add_step('scale', NumericalScaler(), depends_on=['nonexistent'])
        
        # Should raise error about missing dependency
        with pytest.raises(ValueError, match="Dependency not found"):
            pipeline.fit_transform(raw_data)