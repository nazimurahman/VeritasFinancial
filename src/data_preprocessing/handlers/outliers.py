"""
Outlier Handler Module

This module provides comprehensive strategies for detecting and handling
outliers in banking fraud detection data. Outliers can be genuine fraud
or data errors, so careful handling is required.

Detection methods:
1. Z-Score - Standard deviation based
2. IQR - Interquartile range based
3. Modified Z-Score - Robust to extreme outliers
4. Percentile - Based on percentiles
5. Isolation Forest - ML-based detection
6. DBSCAN - Density-based clustering

Handling strategies:
1. Remove - Delete outlier rows
2. Cap - Clip to threshold values
3. Winsorize - Replace with percentiles
4. Transform - Apply transformation
5. Flag - Create indicator and keep
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

logger = logging.getLogger(__name__)


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Comprehensive outlier handler for fraud detection.
    
    This transformer detects and handles outliers using multiple methods,
    with special consideration for fraud detection where outliers may be
    the target of interest.
    
    Attributes:
        detection_method (str): Method for outlier detection
        handling_method (str): Method for handling outliers
        contamination (float): Expected proportion of outliers
        threshold (float): Threshold for outlier detection
        add_outlier_flag (bool): Whether to add outlier indicator columns
        preserve_fraud (bool): Whether to preserve potential fraud cases
    """
    
    def __init__(
        self,
        detection_method: str = 'iqr',
        handling_method: str = 'cap',
        contamination: float = 0.05,
        threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        lower_percentile: float = 1,
        upper_percentile: float = 99,
        add_outlier_flag: bool = True,
        preserve_fraud: bool = True,
        fraud_column: Optional[str] = None,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize the OutlierHandler.
        
        Args:
            detection_method: 'zscore', 'iqr', 'modified_zscore', 'percentile',
                            'isolation_forest', 'dbscan'
            handling_method: 'remove', 'cap', 'winsorize', 'transform', 'flag'
            contamination: Expected proportion of outliers (for ML methods)
            threshold: Threshold for z-score methods
            iqr_multiplier: Multiplier for IQR method
            lower_percentile: Lower percentile for percentile method
            upper_percentile: Upper percentile for percentile method
            add_outlier_flag: Whether to add outlier indicator columns
            preserve_fraud: Whether to preserve potential fraud cases
            fraud_column: Name of fraud label column
            random_state: Random state for reproducibility
            **kwargs: Additional arguments
        """
        self.detection_method = detection_method
        self.handling_method = handling_method
        self.contamination = contamination
        self.threshold = threshold
        self.iqr_multiplier = iqr_multiplier
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.add_outlier_flag = add_outlier_flag
        self.preserve_fraud = preserve_fraud
        self.fraud_column = fraud_column
        self.random_state = random_state
        self.kwargs = kwargs
        
        # Storage for fitted data
        self.outlier_bounds = {}
        self.outlier_stats = {}
        self.outlier_detector = None
        self.numerical_columns = []
        self.feature_names_ = []
        
        logger.info(f"OutlierHandler initialized with detection: {detection_method}, handling: {handling_method}")
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the outlier handler to the data.
        
        Args:
            X: Input DataFrame
            y: Optional target values (for preserving fraud)
            
        Returns:
            self
        """
        logger.info(f"Fitting OutlierHandler on {len(X)} samples")
        
        # Identify numerical columns
        self.numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Store target if provided
        if y is not None and self.fraud_column is None:
            self.fraud_values = y
        elif self.fraud_column in X.columns:
            self.fraud_values = X[self.fraud_column]
        else:
            self.fraud_values = None
        
        # Detect outliers and calculate bounds
        self.outlier_bounds = self._calculate_outlier_bounds(X)
        
        # Calculate outlier statistics
        self.outlier_stats = self._calculate_outlier_stats(X)
        
        # Fit ML-based detector if needed
        if self.detection_method in ['isolation_forest', 'dbscan']:
            self._fit_ml_detector(X)
        
        # Generate feature names
        self._generate_feature_names(X)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by handling outliers.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with outliers handled
        """
        logger.info(f"Transforming {len(X)} samples")
        
        result = X.copy()
        
        # Detect outliers
        outlier_masks = self._detect_outliers(result)
        
        # Add outlier flags if requested
        if self.add_outlier_flag:
            result = self._add_outlier_flags(result, outlier_masks)
        
        # Handle outliers
        result = self._handle_outliers(result, outlier_masks)
        
        logger.info(f"Outlier handling complete")
        return result
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit to data, then transform it.
        
        Args:
            X: Input DataFrame
            y: Optional target values
            
        Returns:
            Transformed DataFrame
        """
        return self.fit(X, y).transform(X)
    
    def _calculate_outlier_bounds(self, df: pd.DataFrame) -> Dict:
        """
        Calculate outlier detection bounds for each column.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with outlier bounds
        """
        bounds = {}
        
        for col in self.numerical_columns:
            series = df[col].dropna()
            
            if len(series) == 0:
                bounds[col] = {'lower': None, 'upper': None}
                continue
            
            if self.detection_method == 'zscore':
                # Z-score method
                mean = series.mean()
                std = series.std()
                bounds[col] = {
                    'lower': mean - self.threshold * std,
                    'upper': mean + self.threshold * std,
                    'method': 'zscore'
                }
                
            elif self.detection_method == 'iqr':
                # IQR method
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                bounds[col] = {
                    'lower': q1 - self.iqr_multiplier * iqr,
                    'upper': q3 + self.iqr_multiplier * iqr,
                    'method': 'iqr'
                }
                
            elif self.detection_method == 'modified_zscore':
                # Modified Z-score (using MAD)
                median = series.median()
                mad = np.median(np.abs(series - median))
                if mad > 0:
                    modified_z_scores = 0.6745 * (series - median) / mad
                    bounds[col] = {
                        'lower': median - self.threshold * mad / 0.6745,
                        'upper': median + self.threshold * mad / 0.6745,
                        'method': 'modified_zscore'
                    }
                else:
                    bounds[col] = {'lower': None, 'upper': None}
                    
            elif self.detection_method == 'percentile':
                # Percentile method
                bounds[col] = {
                    'lower': series.quantile(self.lower_percentile / 100),
                    'upper': series.quantile(self.upper_percentile / 100),
                    'method': 'percentile'
                }
            
            logger.debug(f"Column {col} bounds: {bounds[col]}")
        
        return bounds
    
    def _calculate_outlier_stats(self, df: pd.DataFrame) -> Dict:
        """
        Calculate outlier statistics for each column.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with outlier statistics
        """
        stats = {}
        
        for col in self.numerical_columns:
            if col not in self.outlier_bounds:
                continue
            
            bounds = self.outlier_bounds[col]
            if bounds['lower'] is None or bounds['upper'] is None:
                stats[col] = {'outlier_count': 0, 'outlier_pct': 0}
                continue
            
            series = df[col]
            outlier_mask = (series < bounds['lower']) | (series > bounds['upper'])
            outlier_count = outlier_mask.sum()
            outlier_pct = (outlier_count / len(series)) * 100
            
            stats[col] = {
                'outlier_count': outlier_count,
                'outlier_pct': outlier_pct,
                'lower_bound': bounds['lower'],
                'upper_bound': bounds['upper']
            }
            
            if outlier_count > 0:
                logger.debug(f"Column {col}: {outlier_count} outliers ({outlier_pct:.2f}%)")
        
        return stats
    
    def _fit_ml_detector(self, df: pd.DataFrame):
        """
        Fit ML-based outlier detector.
        
        Args:
            df: Input DataFrame
        """
        # Prepare data (use all numerical columns)
        X = df[self.numerical_columns].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Scale for DBSCAN
        if self.detection_method == 'dbscan':
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.outlier_detector = DBSCAN(
                eps=self.kwargs.get('eps', 0.5),
                min_samples=self.kwargs.get('min_samples', 5)
            )
            self.outlier_detector.fit(X_scaled)
            self.scaler = scaler
            
        elif self.detection_method == 'isolation_forest':
            self.outlier_detector = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                **self.kwargs
            )
            self.outlier_detector.fit(X)
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Detect outliers using fitted methods.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with outlier masks for each method
        """
        outlier_masks = {}
        
        # Per-column outlier detection
        col_outliers = pd.DataFrame(index=df.index)
        
        for col in self.numerical_columns:
            if col in self.outlier_bounds:
                bounds = self.outlier_bounds[col]
                if bounds['lower'] is not None and bounds['upper'] is not None:
                    col_outliers[col] = (df[col] < bounds['lower']) | (df[col] > bounds['upper'])
        
        outlier_masks['per_column'] = col_outliers
        
        # ML-based outlier detection
        if self.outlier_detector is not None:
            X = df[self.numerical_columns].fillna(df[self.numerical_columns].median())
            
            if self.detection_method == 'dbscan':
                X_scaled = self.scaler.transform(X)
                labels = self.outlier_detector.fit_predict(X_scaled)
                outlier_masks['ml'] = pd.Series(labels == -1, index=df.index)
                
            elif self.detection_method == 'isolation_forest':
                predictions = self.outlier_detector.predict(X)
                outlier_masks['ml'] = pd.Series(predictions == -1, index=df.index)
        
        return outlier_masks
    
    def _add_outlier_flags(self, df: pd.DataFrame, outlier_masks: Dict) -> pd.DataFrame:
        """
        Add outlier indicator columns.
        
        Args:
            df: Input DataFrame
            outlier_masks: Dictionary of outlier masks
            
        Returns:
            DataFrame with outlier flags
        """
        result = df.copy()
        
        # Add per-column outlier flags
        if 'per_column' in outlier_masks:
            for col in outlier_masks['per_column'].columns:
                result[f"{col}_is_outlier"] = outlier_masks['per_column'][col].astype(int)
        
        # Add ML-based outlier flag
        if 'ml' in outlier_masks:
            result['is_ml_outlier'] = outlier_masks['ml'].astype(int)
        
        # Add combined outlier flag (any column)
        if 'per_column' in outlier_masks:
            any_outlier = outlier_masks['per_column'].any(axis=1)
            result['is_any_outlier'] = any_outlier.astype(int)
        
        return result
    
    def _handle_outliers(self, df: pd.DataFrame, outlier_masks: Dict) -> pd.DataFrame:
        """
        Handle outliers according to specified method.
        
        Args:
            df: Input DataFrame
            outlier_masks: Dictionary of outlier masks
            
        Returns:
            DataFrame with outliers handled
        """
        result = df.copy()
        
        if self.handling_method == 'remove':
            # Remove rows with outliers
            if 'per_column' in outlier_masks:
                outlier_rows = outlier_masks['per_column'].any(axis=1)
                
                # Preserve fraud cases if requested
                if self.preserve_fraud and self.fraud_values is not None:
                    fraud_mask = self.fraud_values == 1
                    rows_to_remove = outlier_rows & ~fraud_mask
                else:
                    rows_to_remove = outlier_rows
                
                result = result[~rows_to_remove]
                logger.info(f"Removed {rows_to_remove.sum()} outlier rows")
        
        elif self.handling_method in ['cap', 'winsorize']:
            # Cap or winsorize outliers
            for col in self.numerical_columns:
                if col in self.outlier_bounds:
                    bounds = self.outlier_bounds[col]
                    if bounds['lower'] is not None and bounds['upper'] is not None:
                        if self.handling_method == 'cap':
                            # Cap to bounds
                            result[col] = result[col].clip(bounds['lower'], bounds['upper'])
                        else:
                            # Winsorize (replace with bounds)
                            result.loc[result[col] < bounds['lower'], col] = bounds['lower']
                            result.loc[result[col] > bounds['upper'], col] = bounds['upper']
        
        elif self.handling_method == 'transform':
            # Apply log transformation to reduce outlier impact
            for col in self.numerical_columns:
                if col in self.outlier_stats and self.outlier_stats[col]['outlier_count'] > 0:
                    # Apply log transformation to skewed columns
                    if result[col].min() >= 0:
                        result[f"{col}_log"] = np.log1p(result[col])
        
        elif self.handling_method == 'flag':
            # Already added flags, keep original values
            pass
        
        return result
    
    def _generate_feature_names(self, X: pd.DataFrame):
        """
        Generate names of features after transformation.
        
        Args:
            X: Input DataFrame
        """
        self.feature_names_ = X.columns.tolist()
        
        if self.add_outlier_flag:
            for col in self.numerical_columns:
                self.feature_names_.append(f"{col}_is_outlier")
            self.feature_names_.append('is_any_outlier')
    
    def get_feature_names_out(self, input_features=None) -> List[str]:
        """
        Get output feature names.
        
        Returns:
            List of feature names
        """
        return self.feature_names_
    
    def get_outlier_report(self) -> Dict:
        """
        Get detailed report on outlier detection and handling.
        
        Returns:
            Dictionary with outlier report
        """
        return {
            'detection_method': self.detection_method,
            'handling_method': self.handling_method,
            'outlier_stats': self.outlier_stats,
            'outlier_bounds': self.outlier_bounds,
            'numerical_columns': self.numerical_columns
        }