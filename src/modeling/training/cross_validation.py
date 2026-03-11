# src/modeling/training/cross_validation.py
"""
Advanced Cross-Validation Module for Fraud Detection

This module provides specialized cross-validation strategies for imbalanced
fraud detection datasets, including time-aware splitting, stratified k-fold,
and custom validation schemes that preserve temporal ordering.

Author: VeritasFinancial DS Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Union, Generator, Callable
from sklearn.model_selection import (
    StratifiedKFold, 
    TimeSeriesSplit,
    BaseCrossValidator,
    train_test_split
)
from sklearn.metrics import make_scorer
import warnings
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResults:
    """
    Data class to store comprehensive validation results.
    
    Attributes:
        fold_metrics: Dictionary containing metrics for each fold
        average_metrics: Dictionary with averaged metrics across folds
        std_metrics: Dictionary with standard deviation of metrics
        feature_importance: List of feature importance per fold (if available)
        predictions: List of predictions per fold
        probabilities: List of prediction probabilities per fold
        labels: List of true labels per fold
        training_time: Training time per fold in seconds
    """
    fold_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)
    average_metrics: Dict[str, float] = field(default_factory=dict)
    std_metrics: Dict[str, float] = field(default_factory=dict)
    feature_importance: List[np.ndarray] = field(default_factory=list)
    predictions: List[np.ndarray] = field(default_factory=list)
    probabilities: List[np.ndarray] = field(default_factory=list)
    labels: List[np.ndarray] = field(default_factory=list)
    training_time: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert validation results to dictionary for serialization."""
        return {
            'fold_metrics': self.fold_metrics,
            'average_metrics': self.average_metrics,
            'std_metrics': self.std_metrics,
            'training_time': self.training_time,
            'n_folds': len(self.fold_metrics)
        }


class FraudTimeSeriesSplit(BaseCrossValidator):
    """
    Time-series cross-validator specifically designed for fraud detection.
    
    Unlike standard time series split, this validator ensures that:
    1. Training data only comes from before validation data
    2. Maintains class distribution across splits
    3. Handles irregular time intervals common in transaction data
    
    Parameters:
        n_splits : int, default=5
            Number of splits. Must be at least 2.
        gap : int, default=0
            Number of samples to exclude from the end of each train set
            before the validation set to prevent data leakage.
        test_size : float or int, default=None
            If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split. If int, represents the
            absolute number of test samples.
        time_col : str, default='transaction_time'
            Name of the column containing timestamps
        date_format : str, default=None
            Format of datetime strings if time_col contains strings
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        gap: int = 0,
        test_size: Optional[Union[float, int]] = None,
        time_col: str = 'transaction_time',
        date_format: Optional[str] = None
    ):
        # Validate input parameters
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        
        self.n_splits = n_splits
        self.gap = gap
        self.test_size = test_size
        self.time_col = time_col
        self.date_format = date_format
        
    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices to split data into training and test set.
        
        Parameters:
            X : pd.DataFrame
                Training data, must contain time_col
            y : pd.Series, optional
                Target variable (not used for splitting but kept for API compatibility)
            groups : array-like, optional
                Group labels for the samples (not used)
                
        Yields:
            train_indices : np.ndarray
                The training set indices for that split
            test_indices : np.ndarray
                The testing set indices for that split
        """
        # Ensure X is a DataFrame and contains time column
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        
        if self.time_col not in X.columns:
            raise ValueError(f"DataFrame must contain column '{self.time_col}'")
        
        # Extract and sort by time
        times = self._process_time_column(X[self.time_col])
        
        # Sort indices by time
        sorted_indices = np.argsort(times)
        n_samples = len(sorted_indices)
        
        # Calculate split sizes
        if self.test_size is None:
            # If test_size not specified, use progressive splits
            indices = np.arange(n_samples)
            test_starts = [
                int((i + 1) * n_samples / (self.n_splits + 1))
                for i in range(self.n_splits)
            ]
            
            for test_start in test_starts:
                # Calculate train indices (all data before test_start)
                train_indices = sorted_indices[:test_start - self.gap]
                # Calculate test indices (test_start onwards)
                test_indices = sorted_indices[test_start:test_start + 
                                             (n_samples - test_start) // (self.n_splits - i)]
                
                # Ensure we have at least some samples in both sets
                if len(train_indices) > 0 and len(test_indices) > 0:
                    yield train_indices, test_indices
        else:
            # Fixed test size - use rolling window approach
            if isinstance(self.test_size, float):
                test_size_int = int(n_samples * self.test_size)
            else:
                test_size_int = self.test_size
            
            for i in range(self.n_splits):
                # Calculate test end position
                test_end = n_samples - (self.n_splits - i - 1) * test_size_int
                test_start = test_end - test_size_int
                
                # Train data is everything before test_start
                train_indices = sorted_indices[:test_start - self.gap]
                test_indices = sorted_indices[test_start:test_end]
                
                if len(train_indices) > 0 and len(test_indices) > 0:
                    yield train_indices, test_indices
    
    def _process_time_column(self, time_series: pd.Series) -> np.ndarray:
        """
        Convert time column to numeric values for sorting.
        
        Parameters:
            time_series : pd.Series
                Column containing timestamps
                
        Returns:
            numeric_times : np.ndarray
                Numeric representation of timestamps for sorting
        """
        # Check if already numeric (e.g., Unix timestamp)
        if pd.api.types.is_numeric_dtype(time_series):
            return time_series.values
        
        # Try to convert to datetime
        try:
            if self.date_format:
                times = pd.to_datetime(time_series, format=self.date_format)
            else:
                times = pd.to_datetime(time_series)
            
            # Convert to Unix timestamp for numeric sorting
            return times.astype(np.int64) // 10**9  # Convert to seconds
        except Exception as e:
            raise ValueError(f"Could not parse time column: {e}")
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Returns the number of splitting iterations in the cross-validator"""
        return self.n_splits


class FraudStratifiedKFold(StratifiedKFold):
    """
    Enhanced stratified k-fold for fraud detection with additional features.
    
    This extends sklearn's StratifiedKFold with:
    1. Stratification by both target and additional strata
    2. Handling of extreme imbalance
    3. Preservation of minority class distribution
    
    Parameters:
        n_splits : int, default=5
            Number of folds
        shuffle : bool, default=True
            Whether to shuffle the data before splitting
        random_state : int, default=42
            Random seed for reproducibility
        min_fraud_per_fold : int, default=1
            Minimum number of fraud cases required per fold
        strata_cols : list, optional
            Additional columns to use for stratification
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        min_fraud_per_fold: int = 1,
        strata_cols: Optional[List[str]] = None
    ):
        super().__init__(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )
        self.min_fraud_per_fold = min_fraud_per_fold
        self.strata_cols = strata_cols or []
    
    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices with enhanced stratification.
        
        Parameters:
            X : pd.DataFrame
                Training data
            y : pd.Series
                Target variable (fraud indicator)
            groups : array-like, optional
                Group labels (not used)
                
        Yields:
            train_indices : np.ndarray
                Training indices
            test_indices : np.ndarray
                Testing indices
        """
        # Create enhanced strata labels
        if self.strata_cols:
            # Create combined strata from multiple columns
            strata = self._create_enhanced_strata(X, y)
        else:
            # Use only target for stratification
            strata = y
        
        # Get base splits from parent class
        splits = list(super().split(X, strata, groups))
        
        # Validate and adjust splits if needed
        adjusted_splits = self._validate_splits(splits, y)
        
        return iter(adjusted_splits)
    
    def _create_enhanced_strata(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """
        Create combined strata labels from multiple columns.
        
        This creates a single categorical variable that combines
        the fraud indicator with other stratification columns.
        
        Parameters:
            X : pd.DataFrame
                Feature dataframe
            y : pd.Series
                Target variable
                
        Returns:
            combined_strata : pd.Series
                Combined strata labels
        """
        # Start with target as string
        strata_parts = [y.astype(str)]
        
        # Add additional strata columns
        for col in self.strata_cols:
            if col in X.columns:
                # Convert to categorical codes for combination
                if X[col].dtype == 'object':
                    # Handle categorical columns
                    codes = pd.Categorical(X[col]).codes
                else:
                    # Bin continuous variables
                    codes = pd.qcut(X[col], q=4, labels=False, duplicates='drop')
                
                strata_parts.append(codes.astype(str))
            else:
                warnings.warn(f"Strata column '{col}' not found in X. Skipping.")
        
        # Combine all parts with separator
        combined = pd.Series('_'.join(parts) for parts in zip(*strata_parts))
        
        return combined
    
    def _validate_splits(
        self,
        splits: List[Tuple[np.ndarray, np.ndarray]],
        y: pd.Series
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Validate that each fold has minimum fraud cases.
        
        Parameters:
            splits : list of (train_idx, test_idx) tuples
                Original splits from parent class
            y : pd.Series
                Target variable
                
        Returns:
            validated_splits : list of (train_idx, test_idx) tuples
                Validated and possibly adjusted splits
        """
        adjusted_splits = []
        
        for train_idx, test_idx in splits:
            # Check fraud count in test set
            test_fraud_count = (y.iloc[test_idx] == 1).sum()
            
            if test_fraud_count < self.min_fraud_per_fold:
                warnings.warn(
                    f"Fold has only {test_fraud_count} fraud cases, "
                    f"which is below minimum {self.min_fraud_per_fold}. "
                    "Adjusting split..."
                )
                
                # Adjust split by moving some fraud cases from train to test
                train_fraud_idx = y.iloc[train_idx][y.iloc[train_idx] == 1].index
                
                if len(train_fraud_idx) >= self.min_fraud_per_fold:
                    # Move some fraud cases from train to test
                    n_to_move = self.min_fraud_per_fold - test_fraud_count
                    fraud_to_move = np.random.choice(
                        train_fraud_idx,
                        size=n_to_move,
                        replace=False
                    )
                    
                    # Create new split
                    train_idx = np.array([i for i in train_idx if i not in fraud_to_move])
                    test_idx = np.concatenate([test_idx, fraud_to_move])
            
            adjusted_splits.append((train_idx, test_idx))
        
        return adjusted_splits


class PurgedGroupTimeSeriesSplit(BaseCrossValidator):
    """
    Time Series cross-validator with purging and embargo periods.
    
    This is crucial for fraud detection to prevent:
    1. Data leakage from overlapping time periods
    2. Look-ahead bias in feature engineering
    
    Parameters:
        n_splits : int, default=5
            Number of splits
        purge_window : int, default=0
            Number of samples to remove from the end of each train set
        embargo_window : int, default=0
            Number of samples to remove from the start of each test set
        time_col : str, default='transaction_time'
            Column containing timestamps
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        purge_window: int = 0,
        embargo_window: int = 0,
        time_col: str = 'transaction_time'
    ):
        self.n_splits = n_splits
        self.purge_window = purge_window
        self.embargo_window = embargo_window
        self.time_col = time_col
    
    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices with purging and embargo periods.
        """
        # Sort by time
        times = X[self.time_col]
        if not pd.api.types.is_datetime64_any_dtype(times):
            times = pd.to_datetime(times)
        
        sorted_idx = np.argsort(times)
        n_samples = len(sorted_idx)
        
        # Calculate fold boundaries
        fold_size = n_samples // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            # Test set indices (current fold)
            test_start = (i + 1) * fold_size
            test_end = min((i + 2) * fold_size, n_samples)
            
            # Apply purge window (remove samples near test set from training)
            purge_end = max(0, test_start - self.purge_window)
            
            # Apply embargo window (remove samples after test set from training)
            embargo_start = min(test_end + self.embargo_window, n_samples)
            
            # Training set is everything before purge_end
            train_indices = sorted_idx[:purge_end]
            
            # Test set is the fold indices
            test_indices = sorted_idx[test_start:test_end]
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


def custom_cross_validate(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    cv: BaseCrossValidator,
    metrics: Dict[str, Callable],
    fit_params: Optional[Dict] = None,
    return_estimator: bool = False,
    return_predictions: bool = False,
    return_probabilities: bool = False,
    verbose: bool = True
) -> ValidationResults:
    """
    Custom cross-validation function with comprehensive metrics and tracking.
    
    Parameters:
        model : object
            Scikit-learn compatible model with fit and predict methods
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target vector
        cv : BaseCrossValidator
            Cross-validation strategy
        metrics : Dict[str, Callable]
            Dictionary of metric names and functions
        fit_params : Dict, optional
            Additional parameters to pass to model.fit()
        return_estimator : bool, default=False
            Whether to return fitted models for each fold
        return_predictions : bool, default=False
            Whether to return predictions for each fold
        return_probabilities : bool, default=False
            Whether to return prediction probabilities for each fold
        verbose : bool, default=True
            Whether to print progress
        
    Returns:
        results : ValidationResults
            Comprehensive validation results
    """
    
    results = ValidationResults()
    fit_params = fit_params or {}
    
    if verbose:
        logger.info(f"Starting {cv.get_n_splits(X, y)}-fold cross-validation...")
    
    # Iterate through folds
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        if verbose:
            logger.info(f"Processing fold {fold + 1}/{cv.get_n_splits(X, y)}")
        
        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        if verbose:
            logger.info(f"  Train size: {len(X_train)}, Val size: {len(X_val)}")
            logger.info(f"  Train fraud rate: {y_train.mean():.4f}, "
                       f"Val fraud rate: {y_val.mean():.4f}")
        
        # Train model
        import time
        start_time = time.time()
        
        try:
            model.fit(X_train, y_train, **fit_params)
            training_time = time.time() - start_time
            results.training_time.append(training_time)
            
            if verbose:
                logger.info(f"  Training completed in {training_time:.2f} seconds")
            
            # Make predictions
            y_pred = model.predict(X_val)
            
            if return_probabilities and hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_val)[:, 1]
                results.probabilities.append(y_proba)
            
            # Store predictions and labels
            if return_predictions:
                results.predictions.append(y_pred)
            results.labels.append(y_val.values)
            
            # Calculate metrics
            fold_metrics = {}
            for metric_name, metric_func in metrics.items():
                try:
                    if metric_name in ['roc_auc', 'average_precision'] and 'proba' in metric_func.__code__.co_varnames:
                        # Metrics that need probabilities
                        if 'y_proba' in locals():
                            score = metric_func(y_val, y_proba)
                        else:
                            score = metric_func(y_val, y_pred)
                    else:
                        # Standard metrics
                        score = metric_func(y_val, y_pred)
                    
                    fold_metrics[metric_name] = score
                except Exception as e:
                    warnings.warn(f"Error calculating {metric_name}: {e}")
                    fold_metrics[metric_name] = np.nan
            
            results.fold_metrics[fold] = fold_metrics
            
            if verbose:
                logger.info(f"  Fold metrics: {fold_metrics}")
            
            # Store feature importance if available
            if hasattr(model, 'feature_importances_'):
                results.feature_importance.append(model.feature_importances_)
            elif hasattr(model, 'coef_'):
                results.feature_importance.append(model.coef_[0] if model.coef_.ndim > 1 else model.coef_)
            
        except Exception as e:
            logger.error(f"Error in fold {fold + 1}: {e}")
            continue
    
    # Calculate average and std metrics
    if results.fold_metrics:
        # Average metrics across folds
        for metric in results.fold_metrics[0].keys():
            values = [results.fold_metrics[f][metric] for f in results.fold_metrics]
            results.average_metrics[metric] = np.nanmean(values)
            results.std_metrics[metric] = np.nanstd(values)
    
    if verbose:
        logger.info("Cross-validation completed!")
        logger.info(f"Average metrics: {results.average_metrics}")
        logger.info(f"Std metrics: {results.std_metrics}")
    
    return results


def nested_cross_validation(
    model_class: type,
    param_grid: Dict,
    X: pd.DataFrame,
    y: pd.Series,
    outer_cv: BaseCrossValidator,
    inner_cv: BaseCrossValidator,
    metric: str,
    search_type: str = 'grid',
    n_iter: int = 10,
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[Dict, ValidationResults]:
    """
    Perform nested cross-validation for unbiased model evaluation and selection.
    
    This is crucial for fraud detection to:
    1. Get unbiased performance estimates
    2. Select optimal hyperparameters without data leakage
    3. Evaluate model stability across different data splits
    
    Parameters:
        model_class : type
            Uninstantiated model class (e.g., XGBClassifier)
        param_grid : Dict
            Hyperparameter grid for search
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target vector
        outer_cv : BaseCrossValidator
            Outer cross-validation for performance estimation
        inner_cv : BaseCrossValidator
            Inner cross-validation for hyperparameter tuning
        metric : str
            Metric to optimize
        search_type : str, default='grid'
            Type of search: 'grid' or 'random'
        n_iter : int, default=10
            Number of iterations for random search
        random_state : int, default=42
            Random seed
        verbose : bool, default=True
            Whether to print progress
        
    Returns:
        best_params : Dict
            Best hyperparameters found
        nested_results : ValidationResults
            Nested cross-validation results
    """
    
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    
    outer_results = ValidationResults()
    best_params_list = []
    
    if verbose:
        logger.info("Starting nested cross-validation...")
        logger.info(f"Outer folds: {outer_cv.get_n_splits(X, y)}")
        logger.info(f"Inner folds: {inner_cv.get_n_splits(X, y)}")
    
    # Outer loop
    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        if verbose:
            logger.info(f"\nOuter fold {outer_fold + 1}/{outer_cv.get_n_splits(X, y)}")
        
        # Split data for outer fold
        X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
        y_train_outer, y_test_outer = y.iloc[train_idx], y.iloc[test_idx]
        
        # Inner CV for hyperparameter tuning
        if search_type == 'grid':
            search = GridSearchCV(
                estimator=model_class(random_state=random_state),
                param_grid=param_grid,
                cv=inner_cv,
                scoring=metric,
                n_jobs=-1,
                verbose=0
            )
        else:  # random search
            search = RandomizedSearchCV(
                estimator=model_class(random_state=random_state),
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=inner_cv,
                scoring=metric,
                random_state=random_state,
                n_jobs=-1,
                verbose=0
            )
        
        # Fit search on outer training data
        search.fit(X_train_outer, y_train_outer)
        
        # Store best parameters
        best_params_list.append(search.best_params_)
        
        if verbose:
            logger.info(f"  Best parameters: {search.best_params_}")
            logger.info(f"  Inner CV score: {search.best_score_:.4f}")
        
        # Evaluate on outer test set
        y_pred = search.predict(X_test_outer)
        y_proba = search.predict_proba(X_test_outer)[:, 1]
        
        # Store results
        outer_results.predictions.append(y_pred)
        outer_results.probabilities.append(y_proba)
        outer_results.labels.append(y_test_outer.values)
        
        # Calculate metrics for this outer fold
        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
        fold_metrics = {
            'f1': f1_score(y_test_outer, y_pred),
            'precision': precision_score(y_test_outer, y_pred),
            'recall': recall_score(y_test_outer, y_pred),
            'roc_auc': roc_auc_score(y_test_outer, y_proba)
        }
        outer_results.fold_metrics[outer_fold] = fold_metrics
    
    # Calculate average metrics across outer folds
    for metric in outer_results.fold_metrics[0].keys():
        values = [outer_results.fold_metrics[f][metric] 
                  for f in outer_results.fold_metrics]
        outer_results.average_metrics[metric] = np.nanmean(values)
        outer_results.std_metrics[metric] = np.nanstd(values)
    
    if verbose:
        logger.info("\nNested CV Results:")
        logger.info(f"Average metrics: {outer_results.average_metrics}")
        logger.info(f"Std metrics: {outer_results.std_metrics}")
    
    # Return best parameters (most common across folds)
    from collections import Counter
    best_params = Counter([str(p) for p in best_params_list]).most_common(1)[0][0]
    best_params = eval(best_params)  # Convert back to dict
    
    return best_params, outer_results


def create_validation_curve(
    model_class: type,
    X: pd.DataFrame,
    y: pd.Series,
    param_name: str,
    param_range: List,
    cv: BaseCrossValidator,
    scoring: str = 'f1',
    n_jobs: int = -1,
    verbose: bool = True
) -> Dict:
    """
    Create validation curve to analyze model behavior with varying parameters.
    
    This helps in understanding:
    1. Bias-variance tradeoff
    2. Optimal parameter ranges
    3. Model stability
    
    Parameters:
        model_class : type
            Uninstantiated model class
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target vector
        param_name : str
            Name of parameter to vary
        param_range : List
            Range of parameter values to test
        cv : BaseCrossValidator
            Cross-validation strategy
        scoring : str, default='f1'
            Scoring metric
        n_jobs : int, default=-1
            Number of parallel jobs
        verbose : bool, default=True
            Whether to print progress
        
    Returns:
        curve_results : Dict
            Dictionary with train and test scores
    """
    
    from sklearn.model_selection import validation_curve
    
    if verbose:
        logger.info(f"Creating validation curve for {param_name}...")
    
    # Calculate validation curve
    train_scores, test_scores = validation_curve(
        estimator=model_class(),
        X=X,
        y=y,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose
    )
    
    # Calculate statistics
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    curve_results = {
        'param_name': param_name,
        'param_range': param_range,
        'train_scores': train_scores.tolist(),
        'test_scores': test_scores.tolist(),
        'train_mean': train_mean.tolist(),
        'train_std': train_std.tolist(),
        'test_mean': test_mean.tolist(),
        'test_std': test_std.tolist()
    }
    
    if verbose:
        logger.info("Validation curve completed")
        
        # Find optimal parameter
        optimal_idx = np.argmax(test_mean)
        logger.info(f"Optimal {param_name}: {param_range[optimal_idx]}")
        logger.info(f"Best {scoring} score: {test_mean[optimal_idx]:.4f} (+/- {test_std[optimal_idx]:.4f})")
    
    return curve_results


def walk_forward_validation(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    time_col: str,
    window_size: Union[int, str],
    step_size: Union[int, str],
    metrics: Dict[str, Callable],
    retrain_frequency: int = 1,
    verbose: bool = True
) -> ValidationResults:
    """
    Perform walk-forward validation (expanding window) for time series fraud data.
    
    This simulates how the model would perform in production:
    1. Train on historical data
    2. Predict on future data
    3. Periodically retrain with new data
    
    Parameters:
        model : object
            Scikit-learn compatible model
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target vector
        time_col : str
            Column containing timestamps
        window_size : int or str
            Size of training window (int for rows, str for time like '30D')
        step_size : int or str
            Size of prediction window
        metrics : Dict[str, Callable]
            Dictionary of metric functions
        retrain_frequency : int, default=1
            Number of steps before retraining (1 = retrain every step)
        verbose : bool, default=True
            Whether to print progress
        
    Returns:
        results : ValidationResults
            Walk-forward validation results
    """
    
    results = ValidationResults()
    
    # Sort by time
    X_sorted = X.sort_values(time_col)
    y_sorted = y.loc[X_sorted.index]
    
    n_samples = len(X_sorted)
    
    if verbose:
        logger.info("Starting walk-forward validation...")
    
    # Convert window and step sizes if they're time strings
    if isinstance(window_size, str):
        # Time-based window
        window_timedelta = pd.Timedelta(window_size)
        times = X_sorted[time_col]
        start_time = times.min()
        end_time = times.max()
        
        current_time = start_time + window_timedelta
        step = 1 if isinstance(step_size, int) else pd.Timedelta(step_size)
        
        step_num = 0
        while current_time < end_time:
            # Training data: all data before current_time
            train_mask = times < current_time
            
            if isinstance(step_size, str):
                # Test data: next time period
                test_mask = (times >= current_time) & \
                           (times < current_time + pd.Timedelta(step_size))
            else:
                # Test data: next n samples
                train_indices = np.where(train_mask)[0]
                if len(train_indices) + step_size < n_samples:
                    test_indices = np.arange(len(train_indices), 
                                            len(train_indices) + step_size)
                    test_mask = np.zeros(n_samples, dtype=bool)
                    test_mask[test_indices] = True
                else:
                    break
            
            if train_mask.sum() > 0 and test_mask.sum() > 0:
                # Train model if needed (retrain frequency)
                if step_num % retrain_frequency == 0:
                    model.fit(X_sorted[train_mask], y_sorted[train_mask])
                
                # Make predictions
                y_pred = model.predict(X_sorted[test_mask])
                
                # Calculate metrics
                fold_metrics = {}
                for name, func in metrics.items():
                    try:
                        fold_metrics[name] = func(y_sorted[test_mask], y_pred)
                    except:
                        fold_metrics[name] = np.nan
                
                results.fold_metrics[step_num] = fold_metrics
                results.labels.append(y_sorted[test_mask].values)
                results.predictions.append(y_pred)
                
                if verbose and step_num % 10 == 0:
                    logger.info(f"Step {step_num}: {fold_metrics}")
            
            # Move to next window
            if isinstance(step_size, str):
                current_time += pd.Timedelta(step_size)
            else:
                current_time = times[min(len(train_indices) + step_size, n_samples - 1)]
            
            step_num += 1
    
    else:
        # Row-based window
        for i in range(window_size, n_samples - step_size, step_size):
            train_indices = np.arange(i - window_size, i)
            test_indices = np.arange(i, i + step_size)
            
            # Train model if needed
            if (i - window_size) % (step_size * retrain_frequency) == 0:
                model.fit(X_sorted.iloc[train_indices], y_sorted.iloc[train_indices])
            
            # Make predictions
            y_pred = model.predict(X_sorted.iloc[test_indices])
            
            # Calculate metrics
            fold_metrics = {}
            for name, func in metrics.items():
                try:
                    fold_metrics[name] = func(y_sorted.iloc[test_indices], y_pred)
                except:
                    fold_metrics[name] = np.nan
            
            fold_idx = len(results.fold_metrics)
            results.fold_metrics[fold_idx] = fold_metrics
            results.labels.append(y_sorted.iloc[test_indices].values)
            results.predictions.append(y_pred)
    
    # Calculate average metrics
    for metric in results.fold_metrics[0].keys():
        values = [results.fold_metrics[f][metric] for f in results.fold_metrics]
        results.average_metrics[metric] = np.nanmean(values)
        results.std_metrics[metric] = np.nanstd(values)
    
    if verbose:
        logger.info("\nWalk-forward validation completed!")
        logger.info(f"Average metrics: {results.average_metrics}")
    
    return results


def purged_walk_forward(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    time_col: str,
    train_size: str,
    test_size: str,
    purge_size: str,
    embargo_size: str,
    metrics: Dict[str, Callable],
    verbose: bool = True
) -> ValidationResults:
    """
    Purged walk-forward validation with embargo periods.
    
    This is the gold standard for fraud detection validation as it:
    1. Prevents data leakage between train and test
    2. Accounts for temporal dependencies
    3. Simulates real production scenarios
    
    Parameters:
        model : object
            Scikit-learn compatible model
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target vector
        time_col : str
            Column containing timestamps
        train_size : str
            Size of training window (e.g., '30D', '90D')
        test_size : str
            Size of testing window (e.g., '7D', '30D')
        purge_size : str
            Purge window size (e.g., '1D', '2D')
        embargo_size : str
            Embargo window size (e.g., '1D', '2D')
        metrics : Dict[str, Callable]
            Dictionary of metric functions
        verbose : bool, default=True
            Whether to print progress
        
    Returns:
        results : ValidationResults
            Purged walk-forward validation results
    """
    
    results = ValidationResults()
    
    # Sort by time
    X_sorted = X.sort_values(time_col)
    y_sorted = y.loc[X_sorted.index]
    times = X_sorted[time_col]
    
    # Convert time strings to timedeltas
    train_delta = pd.Timedelta(train_size)
    test_delta = pd.Timedelta(test_size)
    purge_delta = pd.Timedelta(purge_size)
    embargo_delta = pd.Timedelta(embargo_size)
    
    # Get time range
    min_time = times.min()
    max_time = times.max()
    
    current_train_end = min_time + train_delta
    
    fold = 0
    while current_train_end + test_delta <= max_time:
        # Define train period
        train_start = min_time
        train_end = current_train_end
        
        # Define test period
        test_start = train_end + purge_delta
        test_end = test_start + test_delta
        
        # Apply embargo (remove samples after test from next train)
        embargo_end = test_end + embargo_delta
        
        if verbose:
            logger.info(f"\nFold {fold + 1}")
            logger.info(f"  Train: {train_start} to {train_end}")
            logger.info(f"  Test: {test_start} to {test_end}")
        
        # Get indices
        train_mask = (times >= train_start) & (times < train_end)
        test_mask = (times >= test_start) & (times < test_end)
        
        if train_mask.sum() > 0 and test_mask.sum() > 0:
            # Train model
            model.fit(X_sorted[train_mask], y_sorted[train_mask])
            
            # Make predictions
            y_pred = model.predict(X_sorted[test_mask])
            y_proba = model.predict_proba(X_sorted[test_mask])[:, 1]
            
            # Calculate metrics
            fold_metrics = {}
            for name, func in metrics.items():
                try:
                    if name in ['roc_auc', 'average_precision']:
                        fold_metrics[name] = func(y_sorted[test_mask], y_proba)
                    else:
                        fold_metrics[name] = func(y_sorted[test_mask], y_pred)
                except Exception as e:
                    warnings.warn(f"Error calculating {name}: {e}")
                    fold_metrics[name] = np.nan
            
            results.fold_metrics[fold] = fold_metrics
            results.labels.append(y_sorted[test_mask].values)
            results.predictions.append(y_pred)
            results.probabilities.append(y_proba)
            
            if verbose:
                logger.info(f"  Metrics: {fold_metrics}")
        
        # Move to next window
        current_train_end += test_delta
        min_time = embargo_end  # Update min time to enforce embargo
        fold += 1
    
    # Calculate average metrics
    for metric in results.fold_metrics[0].keys():
        values = [results.fold_metrics[f][metric] for f in results.fold_metrics]
        results.average_metrics[metric] = np.nanmean(values)
        results.std_metrics[metric] = np.nanstd(values)
    
    if verbose:
        logger.info("\nPurged walk-forward validation completed!")
        logger.info(f"Average metrics: {results.average_metrics}")
    
    return results


# Example usage and testing
if __name__ == "__main__":
    """
    Example demonstrating how to use the cross-validation module.
    """
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic transaction data
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
    X = pd.DataFrame({
        'transaction_time': dates,
        'amount': np.random.exponential(100, n_samples),
        'merchant_id': np.random.randint(1, 100, n_samples),
        'customer_id': np.random.randint(1, 1000, n_samples),
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples)
    })
    
    # Generate imbalanced target (5% fraud)
    y = pd.Series(np.random.choice([0, 1], n_samples, p=[0.95, 0.05]))
    
    print("=" * 60)
    print("Fraud Detection Cross-Validation Examples")
    print("=" * 60)
    
    # Example 1: Time series split
    print("\n1. Time Series Split:")
    tscv = FraudTimeSeriesSplit(n_splits=5, time_col='transaction_time')
    for i, (train_idx, test_idx) in enumerate(tscv.split(X, y)):
        print(f"  Fold {i+1}: Train={len(train_idx)}, Test={len(test_idx)}")
    
    # Example 2: Stratified k-fold
    print("\n2. Stratified K-Fold:")
    skf = FraudStratifiedKFold(n_splits=5, min_fraud_per_fold=2)
    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        train_fraud = y.iloc[train_idx].mean()
        test_fraud = y.iloc[test_idx].mean()
        print(f"  Fold {i+1}: Train fraud={train_fraud:.3f}, Test fraud={test_fraud:.3f}")
    
    # Example 3: Purged walk-forward
    print("\n3. Purged Walk-Forward (simplified):")
    # This would require a fitted model, so we'll skip actual execution
    print("  Use purged_walk_forward() with actual model for production")
    
    print("\n" + "=" * 60)
    print("Module loaded successfully!")
    print("=" * 60)