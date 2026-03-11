"""
Categorical Encoder Module

This module provides various encoding techniques for categorical variables
in banking fraud detection. Different encoding methods capture different
aspects of categorical data for machine learning models.

Encoding techniques implemented:
1. One-Hot Encoding - For nominal categories with few levels
2. Label Encoding - For ordinal categories
3. Target Encoding - For high-cardinality features
4. Frequency Encoding - Based on category counts
5. Weight of Evidence (WOE) - For logistic regression
6. Embedding Encoding - For neural networks
7. Hash Encoding - For memory-efficient encoding
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import category_encoders as ce
import logging
from typing import Dict, List, Optional, Union, Any
import warnings

logger = logging.getLogger(__name__)


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Comprehensive categorical encoder for fraud detection.
    
    This transformer provides multiple encoding strategies and automatically
    selects the appropriate method based on feature characteristics.
    
    Attributes:
        encoding_methods (dict): Mapping of column names to encoding methods
        fitted_encoders (dict): Fitted encoders for each column
        cardinality_thresholds (dict): Thresholds for choosing encoding methods
        target_column (str): Target column for supervised encodings
    """
    
    def __init__(
        self,
        encoding_method: str = 'auto',
        target_column: Optional[str] = None,
        cardinality_thresholds: Optional[Dict] = None,
        handle_unknown: str = 'value',
        handle_missing: str = 'value',
        **kwargs
    ):
        """
        Initialize the CategoricalEncoder.
        
        Args:
            encoding_method: 'auto', 'onehot', 'label', 'target', 'frequency', 
                           'woe', 'embedding', 'hash', or dict mapping columns to methods
            target_column: Name of target column (required for target/woe encoding)
            cardinality_thresholds: Dict with thresholds for auto-selection
            handle_unknown: Strategy for handling unknown categories
            handle_missing: Strategy for handling missing values
            **kwargs: Additional arguments for specific encoders
        """
        self.encoding_method = encoding_method
        self.target_column = target_column
        self.cardinality_thresholds = cardinality_thresholds or {
            'onehot_max': 10,  # Use one-hot for cardinality <= 10
            'target_min': 5,    # Use target encoding for cardinality >= 5
            'target_max': 50,    # Use target encoding for cardinality <= 50
            'embedding_min': 20  # Use embedding for cardinality >= 20
        }
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self.kwargs = kwargs
        
        # Storage for fitted encoders
        self.fitted_encoders = {}
        self.column_cardinality = {}
        self.selected_methods = {}
        self.feature_names_ = []
        
        logger.info(f"CategoricalEncoder initialized with method: {encoding_method}")
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the encoder to the data.
        
        Args:
            X: Input DataFrame with categorical columns
            y: Target values (required for target/woe encoding)
            
        Returns:
            self
        """
        logger.info(f"Fitting CategoricalEncoder on {len(X)} samples")
        
        # Identify categorical columns
        self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not self.categorical_columns:
            logger.warning("No categorical columns found")
            return self
        
        # Calculate cardinality for each column
        for col in self.categorical_columns:
            self.column_cardinality[col] = X[col].nunique()
            logger.debug(f"Column {col} has cardinality {self.column_cardinality[col]}")
        
        # Determine encoding method for each column
        if isinstance(self.encoding_method, dict):
            # Use provided mapping
            self.selected_methods = self.encoding_method
        elif self.encoding_method == 'auto':
            # Auto-select based on cardinality
            self.selected_methods = self._auto_select_methods()
        else:
            # Use same method for all columns
            self.selected_methods = {col: self.encoding_method for col in self.categorical_columns}
        
        logger.info(f"Selected encoding methods: {self.selected_methods}")
        
        # Fit encoders for each column
        for col, method in self.selected_methods.items():
            if method not in ['drop']:
                self.fitted_encoders[col] = self._fit_encoder(
                    col, X[col], method, y
                )
        
        # Generate feature names
        self._generate_feature_names(X)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical columns using fitted encoders.
        
        Args:
            X: Input DataFrame with categorical columns
            
        Returns:
            DataFrame with encoded features
        """
        logger.info(f"Transforming {len(X)} samples")
        
        # Start with non-categorical columns
        non_cat_cols = [col for col in X.columns if col not in self.categorical_columns]
        result = X[non_cat_cols].copy() if non_cat_cols else pd.DataFrame(index=X.index)
        
        # Transform each categorical column
        for col, method in self.selected_methods.items():
            if method == 'drop':
                logger.debug(f"Dropping column {col}")
                continue
            
            if col not in X.columns:
                logger.warning(f"Column {col} not found in transform data")
                continue
            
            encoder = self.fitted_encoders.get(col)
            if encoder is None:
                logger.warning(f"No encoder found for column {col}")
                continue
            
            # Transform the column
            transformed = self._transform_column(encoder, X[col], method)
            
            # Add to result
            if isinstance(transformed, pd.DataFrame):
                for tcol in transformed.columns:
                    result[f"{col}_{tcol}"] = transformed[tcol]
            else:
                result[f"{col}_encoded"] = transformed
        
        logger.info(f"Transformed to {result.shape[1]} features")
        return result
    
    def _auto_select_methods(self) -> Dict[str, str]:
        """
        Automatically select encoding methods based on cardinality.
        
        Returns:
            Dictionary mapping columns to encoding methods
        """
        methods = {}
        
        for col, cardinality in self.column_cardinality.items():
            if cardinality == 2:
                # Binary feature - use label encoding
                methods[col] = 'label'
            elif cardinality <= self.cardinality_thresholds['onehot_max']:
                # Low cardinality - use one-hot
                methods[col] = 'onehot'
            elif cardinality <= self.cardinality_thresholds['target_max']:
                # Medium cardinality with target - use target encoding
                if self.target_column is not None:
                    methods[col] = 'target'
                else:
                    methods[col] = 'frequency'
            elif cardinality > self.cardinality_thresholds['embedding_min']:
                # High cardinality - use frequency or hash
                methods[col] = 'frequency'  # or 'hash' for memory efficiency
            else:
                # Default
                methods[col] = 'frequency'
        
        return methods
    
    def _fit_encoder(
        self, 
        col: str, 
        series: pd.Series, 
        method: str, 
        y: Optional[pd.Series] = None
    ) -> Any:
        """
        Fit an encoder for a single column.
        
        Args:
            col: Column name
            series: Column data
            method: Encoding method
            y: Target values
            
        Returns:
            Fitted encoder
        """
        logger.debug(f"Fitting {method} encoder for column {col}")
        
        if method == 'onehot':
            encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown='ignore',
                **self.kwargs
            )
            encoder.fit(series.values.reshape(-1, 1))
            
        elif method == 'label':
            encoder = LabelEncoder()
            encoder.fit(series.astype(str))
            
        elif method == 'target':
            if y is None:
                raise ValueError("Target encoding requires y values")
            encoder = ce.TargetEncoder(
                cols=[col],
                handle_unknown=self.handle_unknown,
                handle_missing=self.handle_missing,
                **self.kwargs
            )
            encoder.fit(series, y)
            
        elif method == 'woe':
            if y is None:
                raise ValueError("WOE encoding requires y values")
            encoder = ce.WOEEncoder(
                cols=[col],
                handle_unknown=self.handle_unknown,
                handle_missing=self.handle_missing,
                **self.kwargs
            )
            encoder.fit(series, y)
            
        elif method == 'frequency':
            # Frequency encoding (count of each category)
            encoder = series.value_counts(normalize=True).to_dict()
            
        elif method == 'hash':
            encoder = ce.HashingEncoder(
                cols=[col],
                n_components=self.kwargs.get('n_components', 8),
                **self.kwargs
            )
            encoder.fit(series)
            
        elif method == 'embedding':
            # For neural networks - return mapping for embedding layer
            # This just returns the mapping, actual embedding happens in model
            encoder = {
                'vocab_size': series.nunique() + 1,  # +1 for unknown
                'mapping': {val: i+1 for i, val in enumerate(series.unique())}
            }
            
        else:
            raise ValueError(f"Unknown encoding method: {method}")
        
        return encoder
    
    def _transform_column(
        self, 
        encoder: Any, 
        series: pd.Series, 
        method: str
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Transform a single column using fitted encoder.
        
        Args:
            encoder: Fitted encoder
            series: Column data
            method: Encoding method
            
        Returns:
            Transformed data
        """
        if method == 'onehot':
            transformed = encoder.transform(series.values.reshape(-1, 1))
            return pd.DataFrame(
                transformed,
                columns=encoder.get_feature_names_out([series.name]),
                index=series.index
            )
            
        elif method == 'label':
            # Handle unknown values
            transformed = series.astype(str).apply(
                lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
            )
            return transformed
            
        elif method in ['target', 'woe', 'hash']:
            transformed = encoder.transform(series)
            return transformed
            
        elif method == 'frequency':
            # Frequency encoding
            return series.map(encoder).fillna(0)
            
        elif method == 'embedding':
            # Return integer mapping for embedding layer
            mapping = encoder['mapping']
            return series.map(mapping).fillna(0).astype(int)
        
        return series
    
    def _generate_feature_names(self, X: pd.DataFrame):
        """
        Generate feature names after transformation.
        
        Args:
            X: Original DataFrame
        """
        self.feature_names_ = []
        
        # Add non-categorical columns
        non_cat_cols = [col for col in X.columns if col not in self.categorical_columns]
        self.feature_names_.extend(non_cat_cols)
        
        # Add encoded feature names
        for col, method in self.selected_methods.items():
            if method == 'drop':
                continue
            
            if method == 'onehot':
                cardinality = self.column_cardinality.get(col, 0)
                for i in range(min(cardinality, 100)):  # Limit for one-hot
                    self.feature_names_.append(f"{col}_cat_{i}")
            else:
                self.feature_names_.append(f"{col}_encoded")
    
    def get_feature_names_out(self, input_features=None) -> List[str]:
        """
        Get output feature names.
        
        Returns:
            List of feature names
        """
        return self.feature_names_
    
    def get_encoding_info(self) -> Dict:
        """
        Get information about encoding methods used.
        
        Returns:
            Dictionary with encoding information
        """
        return {
            'categorical_columns': self.categorical_columns,
            'cardinality': self.column_cardinality,
            'selected_methods': self.selected_methods,
            'feature_names': self.feature_names_
        }