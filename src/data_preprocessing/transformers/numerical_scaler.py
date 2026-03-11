"""
Numerical Scaler Module

This module provides various scaling and normalization techniques for
numerical features in fraud detection. Proper scaling is crucial for
many machine learning algorithms.

Scaling techniques implemented:
1. StandardScaler - Zero mean, unit variance
2. MinMaxScaler - Scale to [0, 1] range
3. RobustScaler - Robust to outliers using IQR
4. QuantileTransformer - Maps to uniform or normal distribution
5. PowerTransformer - Makes data more Gaussian-like
6. Log Transformer - Log transformation for skewed data
7. Winsorization - Capping extreme values
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    QuantileTransformer, PowerTransformer, Normalizer
)
from scipy import stats
import logging
from typing import Dict, List, Optional, Union, Any
import warnings

logger = logging.getLogger(__name__)


class NumericalScaler(BaseEstimator, TransformerMixin):
    """
    Comprehensive numerical scaler for fraud detection.
    
    This transformer provides multiple scaling strategies and automatically
    selects appropriate methods based on feature distributions.
    
    Attributes:
        scaling_methods (dict): Mapping of column names to scaling methods
        fitted_scalers (dict): Fitted scalers for each column
        outlier_handling (str): Method for handling outliers
        distribution_tests (dict): Results of normality tests
    """
    
    def __init__(
        self,
        scaling_method: str = 'auto',
        outlier_handling: str = 'clip',
        clip_range: Optional[tuple] = None,
        winsorize_limits: tuple = (0.01, 0.99),
        power_method: str = 'yeo-johnson',
        handle_zeros: str = 'add_one',  # For log transform
        **kwargs
    ):
        """
        Initialize the NumericalScaler.
        
        Args:
            scaling_method: 'auto', 'standard', 'minmax', 'robust', 'quantile',
                          'power', 'log', 'none', or dict mapping columns to methods
            outlier_handling: 'clip', 'winsorize', 'remove', 'ignore'
            clip_range: Min/max values for clipping (if None, use percentiles)
            winsorize_limits: Percentile limits for winsorization
            power_method: 'yeo-johnson' or 'box-cox'
            handle_zeros: How to handle zeros for log transform
            **kwargs: Additional arguments for specific scalers
        """
        self.scaling_method = scaling_method
        self.outlier_handling = outlier_handling
        self.clip_range = clip_range
        self.winsorize_limits = winsorize_limits
        self.power_method = power_method
        self.handle_zeros = handle_zeros
        self.kwargs = kwargs
        
        # Storage for fitted scalers and statistics
        self.fitted_scalers = {}
        self.selected_methods = {}
        self.feature_stats = {}
        self.distribution_tests = {}
        self.numerical_columns = []
        self.feature_names_ = []
        
        logger.info(f"NumericalScaler initialized with method: {scaling_method}")
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the scaler to the data.
        
        Args:
            X: Input DataFrame with numerical columns
            y: Ignored, included for compatibility
            
        Returns:
            self
        """
        logger.info(f"Fitting NumericalScaler on {len(X)} samples")
        
        # Identify numerical columns
        self.numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if not self.numerical_columns:
            logger.warning("No numerical columns found")
            return self
        
        # Calculate statistics for each column
        for col in self.numerical_columns:
            self.feature_stats[col] = self._calculate_statistics(X[col])
        
        # Determine scaling method for each column
        if isinstance(self.scaling_method, dict):
            # Use provided mapping
            self.selected_methods = self.scaling_method
        elif self.scaling_method == 'auto':
            # Auto-select based on distribution
            self.selected_methods = self._auto_select_methods(X)
        else:
            # Use same method for all columns
            self.selected_methods = {col: self.scaling_method for col in self.numerical_columns}
        
        logger.info(f"Selected scaling methods: {self.selected_methods}")
        
        # Fit scalers for each column
        for col, method in self.selected_methods.items():
            if method != 'none':
                self.fitted_scalers[col] = self._fit_scaler(
                    col, X[col], method
                )
        
        # Generate feature names
        self.feature_names_ = self.numerical_columns.copy()
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform numerical columns using fitted scalers.
        
        Args:
            X: Input DataFrame with numerical columns
            
        Returns:
            DataFrame with scaled features
        """
        logger.info(f"Transforming {len(X)} samples")
        
        result = X.copy()
        
        # Transform each numerical column
        for col in self.numerical_columns:
            if col not in result.columns:
                logger.warning(f"Column {col} not found in transform data")
                continue
            
            # Handle outliers first
            if self.outlier_handling != 'ignore':
                result[col] = self._handle_outliers(result[col], col)
            
            # Apply scaling
            method = self.selected_methods.get(col, 'none')
            if method != 'none':
                scaler = self.fitted_scalers.get(col)
                if scaler is not None:
                    result[col] = self._transform_column(scaler, result[col], method)
            
            # Add transformed column (may create new columns for some methods)
            if method == 'log' or method == 'power':
                result[f"{col}_transformed"] = result[col]
        
        logger.info(f"Transformation complete")
        return result
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit to data, then transform it.
        
        Args:
            X: Input DataFrame
            y: Ignored
            
        Returns:
            Transformed DataFrame
        """
        return self.fit(X, y).transform(X)
    
    def _calculate_statistics(self, series: pd.Series) -> Dict:
        """
        Calculate statistical properties of a column.
        
        Args:
            series: Input series
            
        Returns:
            Dictionary with statistics
        """
        stats_dict = {
            'mean': series.mean(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'q1': series.quantile(0.25),
            'median': series.median(),
            'q3': series.quantile(0.75),
            'iqr': series.quantile(0.75) - series.quantile(0.25),
            'skewness': series.skew(),
            'kurtosis': series.kurtosis(),
            'zeros_pct': (series == 0).mean() * 100,
            'negatives_pct': (series < 0).mean() * 100
        }
        
        # Add normality test
        if len(series) >= 8:  # Minimum for Shapiro
            try:
                shapiro_stat, shapiro_p = stats.shapiro(series.sample(min(5000, len(series))))
                stats_dict['is_normal'] = shapiro_p > 0.05
            except:
                stats_dict['is_normal'] = False
        
        return stats_dict
    
    def _auto_select_methods(self, X: pd.DataFrame) -> Dict[str, str]:
        """
        Automatically select scaling methods based on feature distributions.
        
        Args:
            X: Input DataFrame
            
        Returns:
            Dictionary mapping columns to scaling methods
        """
        methods = {}
        
        for col in self.numerical_columns:
            series = X[col].dropna()
            
            if len(series) == 0:
                methods[col] = 'none'
                continue
            
            # Check for binary features (0/1)
            if series.nunique() == 2:
                methods[col] = 'none'  # Don't scale binary features
                continue
            
            # Check for count data (positive, integer-like)
            is_count_data = (series.min() >= 0) and (series.dtype in ['int64', 'int32'])
            
            # Check for heavy skewness
            skewness = abs(series.skew())
            
            # Check for outliers
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            has_outliers = ((series < q1 - 3*iqr) | (series > q3 + 3*iqr)).any()
            
            # Decision logic
            if is_count_data and skewness > 1:
                # Count data with high skew - log transform
                methods[col] = 'log'
            elif skewness > 2:
                # Highly skewed - power transform
                methods[col] = 'power'
            elif has_outliers:
                # Has outliers - robust scaling
                methods[col] = 'robust'
            elif self.feature_stats[col].get('is_normal', False):
                # Normally distributed - standard scaling
                methods[col] = 'standard'
            else:
                # Default - quantile transform for non-normal
                methods[col] = 'quantile'
        
        return methods
    
    def _fit_scaler(self, col: str, series: pd.Series, method: str) -> Any:
        """
        Fit a scaler for a single column.
        
        Args:
            col: Column name
            series: Column data
            method: Scaling method
            
        Returns:
            Fitted scaler
        """
        logger.debug(f"Fitting {method} scaler for column {col}")
        
        # Prepare data (remove NaN for fitting)
        clean_series = series.dropna().values.reshape(-1, 1)
        
        if method == 'standard':
            scaler = StandardScaler(**self.kwargs)
            scaler.fit(clean_series)
            
        elif method == 'minmax':
            scaler = MinMaxScaler(**self.kwargs)
            scaler.fit(clean_series)
            
        elif method == 'robust':
            scaler = RobustScaler(**self.kwargs)
            scaler.fit(clean_series)
            
        elif method == 'quantile':
            scaler = QuantileTransformer(
                output_distribution='normal',
                **self.kwargs
            )
            scaler.fit(clean_series)
            
        elif method == 'power':
            scaler = PowerTransformer(
                method=self.power_method,
                **self.kwargs
            )
            scaler.fit(clean_series)
            
        elif method == 'log':
            # Log transform doesn't need fitting, just store parameters
            scaler = {
                'min_val': series.min(),
                'has_zeros': (series == 0).any(),
                'has_negatives': (series < 0).any()
            }
            
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        return scaler
    
    def _transform_column(self, scaler: Any, series: pd.Series, method: str) -> pd.Series:
        """
        Transform a single column using fitted scaler.
        
        Args:
            scaler: Fitted scaler
            series: Column data
            method: Scaling method
            
        Returns:
            Transformed series
        """
        # Handle NaN values
        nan_mask = series.isna()
        result = series.copy()
        
        if method == 'log':
            # Log transformation
            if isinstance(scaler, dict):
                # Add small constant to handle zeros/negatives
                if scaler['has_zeros'] or scaler['has_negatives']:
                    if self.handle_zeros == 'add_one':
                        result = np.log1p(result - scaler['min_val'] + 1)
                    else:
                        result = np.log(result - scaler['min_val'] + 1e-10)
                else:
                    result = np.log(result)
            else:
                result = np.log(result + 1e-10)
                
        elif method == 'power':
            # Power transform
            if hasattr(scaler, 'transform'):
                transformed = scaler.transform(result.values.reshape(-1, 1))
                result = pd.Series(transformed.flatten(), index=result.index)
                
        else:
            # Standard scaling methods
            if hasattr(scaler, 'transform'):
                transformed = scaler.transform(result.values.reshape(-1, 1))
                result = pd.Series(transformed.flatten(), index=result.index)
        
        # Restore NaN values
        result[nan_mask] = np.nan
        
        return result
    
    def _handle_outliers(self, series: pd.Series, col: str) -> pd.Series:
        """
        Handle outliers in a column.
        
        Args:
            series: Input series
            col: Column name
            
        Returns:
            Series with outliers handled
        """
        result = series.copy()
        
        if self.outlier_handling == 'ignore':
            return result
        
        # Calculate outlier bounds
        if self.clip_range is not None:
            lower, upper = self.clip_range
        else:
            # Use IQR method
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 3 * iqr
            upper = q3 + 3 * iqr
        
        # Count outliers
        outliers = ((series < lower) | (series > upper)) & series.notna()
        outlier_count = outliers.sum()
        
        if outlier_count > 0:
            logger.debug(f"Found {outlier_count} outliers in column {col}")
            
            if self.outlier_handling == 'clip':
                # Clip to bounds
                result = result.clip(lower, upper)
                
            elif self.outlier_handling == 'winsorize':
                # Winsorize (replace with percentiles)
                lower_pct, upper_pct = self.winsorize_limits
                lower_val = series.quantile(lower_pct)
                upper_val = series.quantile(upper_pct)
                result = result.clip(lower_val, upper_val)
                
            elif self.outlier_handling == 'remove':
                # Set outliers to NaN (they will be handled by missing value handler)
                result[outliers] = np.nan
        
        return result
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform scaled data back to original scale.
        
        Args:
            X: Scaled DataFrame
            
        Returns:
            DataFrame with original scale
        """
        result = X.copy()
        
        for col in self.numerical_columns:
            method = self.selected_methods.get(col, 'none')
            scaler = self.fitted_scalers.get(col)
            
            if method != 'none' and scaler is not None:
                if hasattr(scaler, 'inverse_transform'):
                    transformed = scaler.inverse_transform(
                        result[col].values.reshape(-1, 1)
                    )
                    result[col] = transformed.flatten()
                elif method == 'log':
                    # Inverse of log transform
                    result[col] = np.exp(result[col]) - 1
        
        return result
    
    def get_scaling_info(self) -> Dict:
        """
        Get information about scaling methods used.
        
        Returns:
            Dictionary with scaling information
        """
        return {
            'numerical_columns': self.numerical_columns,
            'selected_methods': self.selected_methods,
            'feature_stats': self.feature_stats,
            'outlier_handling': self.outlier_handling
        }