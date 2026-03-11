"""
Class Imbalance Handler Module

This module provides comprehensive strategies for handling class imbalance
in fraud detection. Fraud datasets are typically highly imbalanced with
very few fraudulent transactions compared to legitimate ones.

Handling strategies:
1. Undersampling - Random undersampling of majority class
2. Oversampling - Random oversampling of minority class
3. SMOTE - Synthetic Minority Over-sampling Technique
4. ADASYN - Adaptive Synthetic Sampling
5. Borderline-SMOTE - Focus on borderline examples
6. SMOTE-ENN - SMOTE with Edited Nearest Neighbors
7. SMOTE-Tomek - SMOTE with Tomek links
8. Ensemble methods - Balanced Random Forest, etc.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import (
    SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler
)
from imblearn.under_sampling import (
    RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours
)
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

logger = logging.getLogger(__name__)


class ImbalanceHandler(BaseEstimator, TransformerMixin):
    """
    Comprehensive class imbalance handler for fraud detection.
    
    This transformer applies various resampling techniques to handle
    class imbalance in training data.
    
    Attributes:
        method (str): Resampling method
        sampling_strategy (float or dict): Target ratio or dictionary
        random_state (int): Random state for reproducibility
        preserve_original (bool): Whether to preserve original data
        target_column (str): Name of target column
    """
    
    def __init__(
        self,
        method: str = 'smote',
        sampling_strategy: Union[float, str, Dict] = 'auto',
        random_state: int = 42,
        preserve_original: bool = False,
        target_column: Optional[str] = None,
        knn_neighbors: int = 5,
        n_jobs: int = -1,
        **kwargs
    ):
        """
        Initialize the ImbalanceHandler.
        
        Args:
            method: 'smote', 'adasyn', 'borderline_smote', 'random_oversampler',
                   'random_undersampler', 'nearmiss', 'tomek', 'enn',
                   'smote_enn', 'smote_tomek'
            sampling_strategy: Target ratio or dictionary
            random_state: Random state for reproducibility
            preserve_original: Whether to preserve original data
            target_column: Name of target column
            knn_neighbors: Number of neighbors for SMOTE methods
            n_jobs: Number of parallel jobs
            **kwargs: Additional arguments
        """
        self.method = method
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.preserve_original = preserve_original
        self.target_column = target_column
        self.knn_neighbors = knn_neighbors
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        
        # Storage for fitted data
        self.resampler = None
        self.original_class_distribution = None
        self.resampled_class_distribution = None
        self.feature_names_ = []
        
        logger.info(f"ImbalanceHandler initialized with method: {method}")
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the imbalance handler to the data.
        
        Args:
            X: Input DataFrame (features)
            y: Target values
            
        Returns:
            self
        """
        logger.info(f"Fitting ImbalanceHandler on {len(X)} samples")
        
        # Get target values
        if y is not None:
            self.y = y
        elif self.target_column and self.target_column in X.columns:
            self.y = X[self.target_column]
            self.X = X.drop(columns=[self.target_column])
        else:
            raise ValueError("Target values must be provided")
        
        # Calculate original class distribution
        self.original_class_distribution = self._get_class_distribution(self.y)
        logger.info(f"Original class distribution: {self.original_class_distribution}")
        
        # Initialize resampler
        self.resampler = self._create_resampler()
        
        # Store feature names
        self.feature_names_ = X.columns.tolist()
        
        return self
    
    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Transform data by applying resampling.
        
        Args:
            X: Input DataFrame
            y: Target values
            
        Returns:
            Tuple of (resampled_features, resampled_target)
        """
        logger.info(f"Transforming {len(X)} samples")
        
        # Get target values
        if y is not None:
            y_data = y
        elif self.target_column and self.target_column in X.columns:
            y_data = X[self.target_column]
            X_data = X.drop(columns=[self.target_column])
        else:
            X_data = X
            y_data = self.y if hasattr(self, 'y') else None
        
        if y_data is None:
            raise ValueError("Target values must be provided")
        
        # Apply resampling
        if self.preserve_original:
            # Combine original and resampled data
            X_resampled, y_resampled = self.resampler.fit_resample(X_data, y_data)
            
            # Add original data
            X_resampled = pd.concat([X_data, X_resampled], axis=0, ignore_index=True)
            y_resampled = pd.concat([y_data, y_resampled], axis=0, ignore_index=True)
        else:
            X_resampled, y_resampled = self.resampler.fit_resample(X_data, y_data)
        
        # Convert to DataFrame/Series if needed
        if not isinstance(X_resampled, pd.DataFrame):
            X_resampled = pd.DataFrame(X_resampled, columns=X_data.columns)
        if not isinstance(y_resampled, pd.Series):
            y_resampled = pd.Series(y_resampled, name=y_data.name)
        
        # Calculate resampled class distribution
        self.resampled_class_distribution = self._get_class_distribution(y_resampled)
        logger.info(f"Resampled class distribution: {self.resampled_class_distribution}")
        
        logger.info(f"Resampling complete. New shape: {X_resampled.shape}")
        return X_resampled, y_resampled
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit to data, then transform it.
        
        Args:
            X: Input DataFrame
            y: Target values
            
        Returns:
            Tuple of (resampled_features, resampled_target)
        """
        return self.fit(X, y).transform(X, y)
    
    def _get_class_distribution(self, y: pd.Series) -> Dict:
        """
        Get class distribution statistics.
        
        Args:
            y: Target series
            
        Returns:
            Dictionary with class distribution
        """
        counts = y.value_counts()
        percentages = y.value_counts(normalize=True) * 100
        
        distribution = {}
        for cls in counts.index:
            distribution[cls] = {
                'count': counts[cls],
                'percentage': percentages[cls]
            }
        
        return distribution
    
    def _create_resampler(self):
        """
        Create resampler based on selected method.
        
        Returns:
            Resampler object
        """
        # Oversampling methods
        if self.method == 'smote':
            return SMOTE(
                sampling_strategy=self.sampling_strategy,
                k_neighbors=self.knn_neighbors,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                **self.kwargs
            )
        
        elif self.method == 'adasyn':
            return ADASYN(
                sampling_strategy=self.sampling_strategy,
                n_neighbors=self.knn_neighbors,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                **self.kwargs
            )
        
        elif self.method == 'borderline_smote':
            return BorderlineSMOTE(
                sampling_strategy=self.sampling_strategy,
                k_neighbors=self.knn_neighbors,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                **self.kwargs
            )
        
        elif self.method == 'random_oversampler':
            return RandomOverSampler(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                **self.kwargs
            )
        
        # Undersampling methods
        elif self.method == 'random_undersampler':
            return RandomUnderSampler(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                **self.kwargs
            )
        
        elif self.method == 'nearmiss':
            return NearMiss(
                sampling_strategy=self.sampling_strategy,
                n_neighbors=self.knn_neighbors,
                **self.kwargs
            )
        
        elif self.method == 'tomek':
            return TomekLinks(
                sampling_strategy=self.sampling_strategy,
                n_jobs=self.n_jobs,
                **self.kwargs
            )
        
        elif self.method == 'enn':
            return EditedNearestNeighbours(
                sampling_strategy=self.sampling_strategy,
                n_neighbors=self.knn_neighbors,
                n_jobs=self.n_jobs,
                **self.kwargs
            )
        
        # Combined methods
        elif self.method == 'smote_enn':
            return SMOTEENN(
                smote=SMOTE(
                    sampling_strategy=self.sampling_strategy,
                    k_neighbors=self.knn_neighbors,
                    random_state=self.random_state
                ),
                enn=EditedNearestNeighbours(
                    n_neighbors=self.knn_neighbors,
                    n_jobs=self.n_jobs
                ),
                random_state=self.random_state
            )
        
        elif self.method == 'smote_tomek':
            return SMOTETomek(
                smote=SMOTE(
                    sampling_strategy=self.sampling_strategy,
                    k_neighbors=self.knn_neighbors,
                    random_state=self.random_state
                ),
                tomek=TomekLinks(),
                random_state=self.random_state
            )
        
        else:
            raise ValueError(f"Unknown resampling method: {self.method}")
    
    def get_resampling_report(self) -> Dict:
        """
        Get detailed report on resampling.
        
        Returns:
            Dictionary with resampling report
        """
        return {
            'method': self.method,
            'sampling_strategy': self.sampling_strategy,
            'original_distribution': self.original_class_distribution,
            'resampled_distribution': self.resampled_class_distribution,
            'original_size': sum(d['count'] for d in self.original_class_distribution.values()),
            'resampled_size': sum(d['count'] for d in self.resampled_class_distribution.values()) if self.resampled_class_distribution else None
        }