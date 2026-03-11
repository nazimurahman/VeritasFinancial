"""
Missing Value Handler Module

This module provides comprehensive strategies for handling missing values
in banking fraud detection data. Different strategies are appropriate
for different types of data and missingness patterns.

Handling strategies:
1. Deletion - Remove rows/columns with too many missing values
2. Mean/Median/Mode imputation - Simple statistical imputation
3. Forward/Backward fill - For time series data
4. Interpolation - Linear, polynomial, spline
5. KNN imputation - Using similar samples
6. MICE - Multiple Imputation by Chained Equations
7. Flag and fill - Create missing indicator and impute
8. Domain-specific imputation - Based on business rules
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import logging
from typing import Dict, List, Optional, Union, Any, Callable

logger = logging.getLogger(__name__)


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """
    Comprehensive missing value handler for fraud detection.
    
    This transformer provides multiple imputation strategies and
    automatically selects appropriate methods based on data characteristics.
    
    Attributes:
        strategies (dict): Mapping of column names to imputation strategies
        missing_threshold (float): Maximum allowed missing percentage
        add_missing_indicator (bool): Whether to add missing value indicators
        column_types (dict): Types of columns (numeric, categorical, etc.)
    """
    
    def __init__(
        self,
        strategy: Union[str, Dict] = 'auto',
        missing_threshold: float = 0.5,
        add_missing_indicator: bool = True,
        fill_value: Optional[Any] = None,
        numeric_strategy: str = 'median',
        categorical_strategy: str = 'mode',
        time_series_columns: Optional[List[str]] = None,
        knn_neighbors: int = 5,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize the MissingValueHandler.
        
        Args:
            strategy: 'auto', 'delete', 'mean', 'median', 'mode', 'constant',
                     'ffill', 'bfill', 'interpolate', 'knn', 'mice',
                     or dict mapping columns to strategies
            missing_threshold: Maximum allowed missing percentage per column
            add_missing_indicator: Whether to add missing value indicator columns
            fill_value: Value to use for constant imputation
            numeric_strategy: Default strategy for numeric columns
            categorical_strategy: Default strategy for categorical columns
            time_series_columns: Columns that should use time-series imputation
            knn_neighbors: Number of neighbors for KNN imputation
            random_state: Random state for reproducibility
            **kwargs: Additional arguments
        """
        self.strategy = strategy
        self.missing_threshold = missing_threshold
        self.add_missing_indicator = add_missing_indicator
        self.fill_value = fill_value
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.time_series_columns = time_series_columns or []
        self.knn_neighbors = knn_neighbors
        self.random_state = random_state
        self.kwargs = kwargs
        
        # Storage for fitted data
        self.imputers = {}
        self.missing_stats = {}
        self.columns_to_drop = []
        self.columns_to_impute = []
        self.column_types = {}
        self.selected_strategies = {}
        self.feature_names_ = []
        
        logger.info(f"MissingValueHandler initialized with strategy: {strategy}")
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the missing value handler to the data.
        
        Args:
            X: Input DataFrame
            y: Ignored
            
        Returns:
            self
        """
        logger.info(f"Fitting MissingValueHandler on {len(X)} samples")
        
        # Calculate missing value statistics
        self.missing_stats = self._calculate_missing_stats(X)
        
        # Determine column types
        self.column_types = self._identify_column_types(X)
        
        # Identify columns to drop
        self.columns_to_drop = self._identify_columns_to_drop(X)
        
        # Identify columns to impute
        self.columns_to_impute = [
            col for col in X.columns 
            if col not in self.columns_to_drop
        ]
        
        # Determine imputation strategies for each column
        self.selected_strategies = self._determine_strategies(X)
        
        # Fit imputers for each column group
        self._fit_imputers(X)
        
        # Generate feature names
        self._generate_feature_names(X)
        
        logger.info(f"Will drop {len(self.columns_to_drop)} columns, impute {len(self.columns_to_impute)} columns")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by handling missing values.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        logger.info(f"Transforming {len(X)} samples")
        
        result = X.copy()
        
        # Drop columns with too many missing values
        result = result.drop(columns=self.columns_to_drop, errors='ignore')
        
        # Add missing indicators if requested
        if self.add_missing_indicator:
            result = self._add_missing_indicators(result)
        
        # Impute missing values
        result = self._impute_missing_values(result)
        
        logger.info(f"Missing value handling complete")
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
    
    def _calculate_missing_stats(self, df: pd.DataFrame) -> Dict:
        """
        Calculate missing value statistics for each column.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with missing statistics
        """
        stats = {}
        
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            
            stats[col] = {
                'count': missing_count,
                'percentage': missing_pct,
                'dtype': df[col].dtype
            }
            
            logger.debug(f"Column {col}: {missing_pct:.2f}% missing")
        
        return stats
    
    def _identify_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Identify types of columns (numeric, categorical, datetime).
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary mapping columns to types
        """
        types = {}
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() < 10:
                    types[col] = 'numeric_low_cardinality'
                else:
                    types[col] = 'numeric'
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                types[col] = 'datetime'
            elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object':
                if df[col].nunique() < 20:
                    types[col] = 'categorical_low_cardinality'
                else:
                    types[col] = 'categorical_high_cardinality'
            else:
                types[col] = 'other'
        
        return types
    
    def _identify_columns_to_drop(self, df: pd.DataFrame) -> List[str]:
        """
        Identify columns that should be dropped due to too many missing values.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of column names to drop
        """
        to_drop = []
        
        for col, stats in self.missing_stats.items():
            if stats['percentage'] > self.missing_threshold * 100:
                logger.warning(f"Dropping column {col} with {stats['percentage']:.2f}% missing values")
                to_drop.append(col)
        
        return to_drop
    
    def _determine_strategies(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Determine imputation strategy for each column.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary mapping columns to strategies
        """
        strategies = {}
        
        if isinstance(self.strategy, dict):
            # Use provided mapping
            return self.strategy
        
        for col in self.columns_to_impute:
            if col in self.time_series_columns:
                # Time series columns
                strategies[col] = 'ffill'  # Forward fill as default
            elif self.strategy != 'auto':
                # Use specified strategy
                strategies[col] = self.strategy
            else:
                # Auto-select based on column type
                col_type = self.column_types.get(col, 'other')
                missing_pct = self.missing_stats[col]['percentage']
                
                if col_type.startswith('numeric'):
                    if missing_pct < 5:
                        strategies[col] = self.numeric_strategy
                    elif missing_pct < 20:
                        strategies[col] = 'knn' if len(df) < 10000 else self.numeric_strategy
                    else:
                        strategies[col] = 'mice' if len(df) < 5000 else self.numeric_strategy
                
                elif col_type.startswith('categorical'):
                    strategies[col] = self.categorical_strategy
                
                elif col_type == 'datetime':
                    strategies[col] = 'ffill'
                
                else:
                    strategies[col] = 'mode'
        
        return strategies
    
    def _fit_imputers(self, df: pd.DataFrame):
        """
        Fit imputers for each column group.
        
        Args:
            df: Input DataFrame
        """
        # Group columns by strategy
        strategy_groups = {}
        for col, strategy in self.selected_strategies.items():
            if col not in self.columns_to_drop:
                if strategy not in strategy_groups:
                    strategy_groups[strategy] = []
                strategy_groups[strategy].append(col)
        
        # Fit imputer for each strategy group
        for strategy, columns in strategy_groups.items():
            if strategy in ['mean', 'median', 'mode', 'constant']:
                # Simple imputer
                if strategy == 'mode':
                    sklearn_strategy = 'most_frequent'
                else:
                    sklearn_strategy = strategy
                
                imputer = SimpleImputer(
                    strategy=sklearn_strategy,
                    fill_value=self.fill_value,
                    **self.kwargs
                )
                
                # Fit on all columns in this group together
                if columns:
                    imputer.fit(df[columns])
                    self.imputers[strategy] = {
                        'imputer': imputer,
                        'columns': columns
                    }
            
            elif strategy == 'knn':
                # KNN imputer
                imputer = KNNImputer(
                    n_neighbors=self.knn_neighbors,
                    **self.kwargs
                )
                
                if columns:
                    # KNN imputer works on all numeric columns together
                    numeric_cols = [c for c in columns if self.column_types.get(c, '').startswith('numeric')]
                    if numeric_cols:
                        imputer.fit(df[numeric_cols])
                        self.imputers['knn'] = {
                            'imputer': imputer,
                            'columns': numeric_cols
                        }
            
            elif strategy == 'mice':
                # MICE imputer
                imputer = IterativeImputer(
                    random_state=self.random_state,
                    **self.kwargs
                )
                
                if columns:
                    numeric_cols = [c for c in columns if self.column_types.get(c, '').startswith('numeric')]
                    if numeric_cols:
                        imputer.fit(df[numeric_cols])
                        self.imputers['mice'] = {
                            'imputer': imputer,
                            'columns': numeric_cols
                        }
            
            elif strategy in ['ffill', 'bfill', 'interpolate']:
                # Time series methods don't need fitting
                self.imputers[strategy] = {
                    'imputer': None,
                    'columns': columns
                }
    
    def _add_missing_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add indicator columns for missing values.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing indicators
        """
        result = df.copy()
        
        for col in self.columns_to_impute:
            if col in df.columns:
                missing_mask = df[col].isna()
                if missing_mask.any():
                    result[f"{col}_missing"] = missing_mask.astype(int)
                    logger.debug(f"Added missing indicator for {col}")
        
        return result
    
    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using fitted imputers.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with imputed values
        """
        result = df.copy()
        
        # Apply each imputer
        for strategy, imputer_info in self.imputers.items():
            columns = [c for c in imputer_info['columns'] if c in result.columns]
            
            if not columns:
                continue
            
            if strategy in ['mean', 'median', 'mode', 'constant']:
                # Simple imputation
                imputer = imputer_info['imputer']
                imputed_values = imputer.transform(result[columns])
                result[columns] = imputed_values
                
            elif strategy == 'knn':
                # KNN imputation (numeric only)
                imputer = imputer_info['imputer']
                numeric_cols = [c for c in columns if self.column_types.get(c, '').startswith('numeric')]
                if numeric_cols:
                    imputed_values = imputer.transform(result[numeric_cols])
                    result[numeric_cols] = imputed_values
            
            elif strategy == 'mice':
                # MICE imputation
                imputer = imputer_info['imputer']
                numeric_cols = [c for c in columns if self.column_types.get(c, '').startswith('numeric')]
                if numeric_cols:
                    imputed_values = imputer.transform(result[numeric_cols])
                    result[numeric_cols] = imputed_values
            
            elif strategy == 'ffill':
                # Forward fill
                for col in columns:
                    if col in result.columns:
                        result[col] = result[col].fillna(method='ffill')
            
            elif strategy == 'bfill':
                # Backward fill
                for col in columns:
                    if col in result.columns:
                        result[col] = result[col].fillna(method='bfill')
            
            elif strategy == 'interpolate':
                # Interpolation
                for col in columns:
                    if col in result.columns:
                        result[col] = result[col].interpolate(method='linear', limit_direction='both')
        
        # Fill any remaining missing values with a default
        for col in result.columns:
            if result[col].isna().any():
                if self.column_types.get(col, '').startswith('numeric'):
                    result[col] = result[col].fillna(0)
                elif self.column_types.get(col, '').startswith('categorical'):
                    result[col] = result[col].fillna('UNKNOWN')
                else:
                    result[col] = result[col].fillna('MISSING')
        
        return result
    
    def _generate_feature_names(self, X: pd.DataFrame):
        """
        Generate names of features after transformation.
        
        Args:
            X: Input DataFrame
        """
        self.feature_names_ = []
        
        # Add columns that remain
        for col in X.columns:
            if col not in self.columns_to_drop:
                self.feature_names_.append(col)
        
        # Add missing indicators
        if self.add_missing_indicator:
            for col in self.columns_to_impute:
                if self.missing_stats[col]['count'] > 0:
                    self.feature_names_.append(f"{col}_missing")
    
    def get_feature_names_out(self, input_features=None) -> List[str]:
        """
        Get output feature names.
        
        Returns:
            List of feature names
        """
        return self.feature_names_
    
    def get_missing_report(self) -> Dict:
        """
        Get detailed report on missing value handling.
        
        Returns:
            Dictionary with missing value report
        """
        return {
            'missing_stats': self.missing_stats,
            'columns_dropped': self.columns_to_drop,
            'columns_imputed': self.columns_to_impute,
            'imputation_strategies': self.selected_strategies,
            'column_types': self.column_types
        }