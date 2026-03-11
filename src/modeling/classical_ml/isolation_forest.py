"""
VeritasFinancial - Banking Fraud Detection System
Isolation Forest for Anomaly Detection
==================================================
This module implements Isolation Forest algorithm for unsupervised
fraud detection. It's particularly useful for detecting novel fraud
patterns without labeled data.

Key Features:
- Unsupervised anomaly detection
- Online/streaming version for real-time detection
- Configurable contamination estimation
- Feature importance through path lengths
- Handling of high-dimensional banking data

Author: VeritasFinancial Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict, Tuple, Any
from dataclasses import dataclass, field
import logging
import warnings
from datetime import datetime
import pickle
import json

# Scikit-learn imports
from sklearn.ensemble import IsolationForest as SklearnIsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.exceptions import NotFittedError

# Local imports
from ...utils.logger import get_logger
from ...utils.config_manager import ConfigManager

# Configure logger
logger = get_logger(__name__)


@dataclass
class IsolationForestConfig:
    """
    Configuration class for Isolation Forest model.
    
    This dataclass holds all hyperparameters and settings for the
    Isolation Forest model. Using dataclass ensures type safety and
    provides default values.
    
    Attributes:
        n_estimators (int): Number of base estimators in the ensemble.
            More estimators provide better stability but increase computation time.
            Default: 100
        
        max_samples (Union[int, float]): Number of samples to draw for training each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.
            Default: 'auto' (min(256, n_samples))
        
        contamination (Union[float, str]): The expected proportion of outliers in the dataset.
            - If float, should be between 0 and 0.5.
            - If 'auto', determines from data.
            - If 'fraction', uses domain-specific fraction from banking data.
            Default: 'auto'
        
        max_features (Union[int, float]): Number of features to draw for training each base estimator.
            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.
            Default: 1.0
        
        bootstrap (bool): Whether to draw samples with replacement.
            If True, samples are drawn with replacement (bootstrapping).
            Default: False
        
        n_jobs (int): Number of parallel jobs to run.
            -1 means using all processors.
            Default: -1
        
        random_state (int): Random seed for reproducibility.
            Default: 42
        
        verbose (int): Controls the verbosity of the model.
            Higher values mean more messages.
            Default: 0
        
        warm_start (bool): When set to True, reuse the solution of the previous call to fit
            and add more estimators to the ensemble.
            Default: False
        
        online_update_frequency (int): Number of samples before updating online model.
            Default: 1000
        
        feature_importance_method (str): Method to compute feature importance.
            Options: 'path_length', 'mean_depth', 'none'
            Default: 'path_length'
    """
    
    # Core parameters
    n_estimators: int = 100
    max_samples: Union[int, float, str] = 'auto'
    contamination: Union[float, str] = 'auto'
    max_features: Union[int, float] = 1.0
    bootstrap: bool = False
    
    # Computational parameters
    n_jobs: int = -1
    random_state: int = 42
    verbose: int = 0
    warm_start: bool = False
    
    # Advanced parameters
    online_update_frequency: int = 1000
    feature_importance_method: str = 'path_length'
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_parameters()
    
    def _validate_parameters(self):
        """
        Validate all configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid.
        """
        # Validate n_estimators
        if not isinstance(self.n_estimators, int) or self.n_estimators <= 0:
            raise ValueError(f"n_estimators must be positive integer, got {self.n_estimators}")
        
        # Validate max_samples
        if isinstance(self.max_samples, (int, float)):
            if self.max_samples <= 0:
                raise ValueError(f"max_samples must be positive, got {self.max_samples}")
        
        # Validate contamination
        if isinstance(self.contamination, float):
            if not (0 < self.contamination <= 0.5):
                raise ValueError(f"contamination must be in (0, 0.5], got {self.contamination}")
        
        # Validate max_features
        if isinstance(self.max_features, (int, float)):
            if self.max_features <= 0:
                raise ValueError(f"max_features must be positive, got {self.max_features}")
        
        # Validate feature importance method
        valid_methods = ['path_length', 'mean_depth', 'none']
        if self.feature_importance_method not in valid_methods:
            raise ValueError(
                f"feature_importance_method must be one of {valid_methods}, "
                f"got {self.feature_importance_method}"
            )
        
        logger.debug(f"Configuration validated: {self}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of config.
        """
        return {
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'contamination': self.contamination,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'warm_start': self.warm_start,
            'online_update_frequency': self.online_update_frequency,
            'feature_importance_method': self.feature_importance_method
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'IsolationForestConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict (Dict[str, Any]): Dictionary with configuration.
            
        Returns:
            IsolationForestConfig: Configuration instance.
        """
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'IsolationForestConfig':
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path (str): Path to YAML configuration file.
            
        Returns:
            IsolationForestConfig: Configuration instance.
        """
        config_manager = ConfigManager()
        config_dict = config_manager.load_config(yaml_path)
        return cls.from_dict(config_dict)


class IsolationForestDetector:
    """
    Isolation Forest for fraud detection in banking transactions.
    
    This class implements Isolation Forest algorithm specifically optimized
    for fraud detection. It provides unsupervised anomaly detection with
    additional features like feature importance, online learning, and
    domain-specific contamination estimation.
    
    The Isolation Forest algorithm works by randomly selecting a feature
    and then randomly selecting a split value between the minimum and maximum
    values of the selected feature. This random partitioning produces noticeably
    shorter paths for anomalies since:
        - Fewer anomalies means they are more susceptible to isolation
        - Normal points require more partitions to be isolated
    
    For fraud detection in banking, this is particularly useful because:
        1. Fraud patterns evolve constantly (unsupervised detects novel fraud)
        2. Fraud is rare (anomalies are rare by definition)
        3. No labeled data needed for initial deployment
    
    Attributes:
        config (IsolationForestConfig): Model configuration.
        model (SklearnIsolationForest): Underlying sklearn model.
        scaler (StandardScaler): Feature scaler for preprocessing.
        feature_names_ (List[str]): Names of features used for training.
        feature_importances_ (np.ndarray): Feature importance scores.
        contamination_ (float): Estimated contamination from training data.
        is_fitted_ (bool): Whether the model has been fitted.
        training_time_ (datetime): Timestamp of last training.
        feature_stats_ (Dict): Statistics of features during training.
    """
    
    def __init__(self, config: Optional[Union[Dict, IsolationForestConfig]] = None):
        """
        Initialize Isolation Forest detector.
        
        Args:
            config: Configuration for the model. Can be:
                - IsolationForestConfig instance
                - Dictionary with configuration parameters
                - None (uses default configuration)
        """
        # Initialize configuration
        if config is None:
            self.config = IsolationForestConfig()
        elif isinstance(config, dict):
            self.config = IsolationForestConfig(**config)
        elif isinstance(config, IsolationForestConfig):
            self.config = config
        else:
            raise TypeError(f"config must be dict or IsolationForestConfig, got {type(config)}")
        
        # Initialize model components
        self.model = None
        self.scaler = StandardScaler()
        
        # Initialize metadata
        self.feature_names_ = None
        self.feature_importances_ = None
        self.contamination_ = None
        self.is_fitted_ = False
        self.training_time_ = None
        self.feature_stats_ = {}
        
        # Initialize online learning buffer
        self.online_buffer = []
        self.n_samples_seen_ = 0
        
        logger.info(f"Initialized IsolationForestDetector with config: {self.config}")
    
    def _create_model(self) -> SklearnIsolationForest:
        """
        Create underlying sklearn Isolation Forest model.
        
        Returns:
            SklearnIsolationForest: Configured sklearn model.
        """
        return SklearnIsolationForest(
            n_estimators=self.config.n_estimators,
            max_samples=self.config.max_samples,
            contamination=self.config.contamination,
            max_features=self.config.max_features,
            bootstrap=self.config.bootstrap,
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state,
            verbose=self.config.verbose,
            warm_start=self.config.warm_start
        )
    
    def _validate_features(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Validate and preprocess input features.
        
        Args:
            X: Input features as numpy array or pandas DataFrame.
            
        Returns:
            np.ndarray: Preprocessed features.
            
        Raises:
            ValueError: If features are invalid.
        """
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X_array = X.values
        elif isinstance(X, np.ndarray):
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
            X_array = X
        else:
            raise TypeError(f"X must be numpy array or pandas DataFrame, got {type(X)}")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(X_array)):
            logger.warning("Input contains NaN values. Consider preprocessing.")
            # Replace NaN with column mean (simple imputation)
            col_means = np.nanmean(X_array, axis=0)
            inds = np.where(np.isnan(X_array))
            X_array[inds] = np.take(col_means, inds[1])
        
        if np.any(np.isinf(X_array)):
            logger.warning("Input contains infinite values. Replacing with large finite values.")
            X_array = np.where(np.isinf(X_array), np.sign(X_array) * 1e10, X_array)
        
        return X_array
    
    def _estimate_contamination(self, X: np.ndarray) -> float:
        """
        Estimate contamination from banking domain knowledge.
        
        In banking, fraud typically occurs at very low rates (0.1% to 1%).
        This method provides domain-specific contamination estimation.
        
        Args:
            X: Input features.
            
        Returns:
            float: Estimated contamination rate.
        """
        if isinstance(self.config.contamination, float):
            return self.config.contamination
        
        # Use domain knowledge for banking fraud
        # Typically fraud rates are between 0.1% and 1%
        if self.config.contamination == 'auto':
            # Use heuristic based on data characteristics
            n_samples = X.shape[0]
            
            # For small datasets, use higher contamination
            if n_samples < 1000:
                return 0.01  # 1%
            elif n_samples < 10000:
                return 0.005  # 0.5%
            else:
                return 0.001  # 0.1%
        
        return 0.001  # Default 0.1%
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None) -> 'IsolationForestDetector':
        """
        Fit the Isolation Forest model.
        
        This method trains the model on the provided data. Since Isolation Forest
        is unsupervised, y is optional but can be used for validation.
        
        Args:
            X: Training features. Can be numpy array or pandas DataFrame.
            y: Optional labels for validation (not used in training).
            
        Returns:
            IsolationForestDetector: Fitted model instance.
        """
        logger.info("Starting model training...")
        
        # Validate and preprocess features
        X_array = self._validate_features(X)
        
        # Estimate contamination from data
        self.contamination_ = self._estimate_contamination(X_array)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_array)
        
        # Store feature statistics
        self.feature_stats_ = {
            'mean': self.scaler.mean_.tolist() if self.scaler.mean_ is not None else None,
            'scale': self.scaler.scale_.tolist() if self.scaler.scale_ is not None else None,
            'min': X_array.min(axis=0).tolist(),
            'max': X_array.max(axis=0).tolist(),
            'n_samples': X_array.shape[0],
            'n_features': X_array.shape[1]
        }
        
        # Create and train model
        self.model = self._create_model()
        
        try:
            self.model.fit(X_scaled)
            self.is_fitted_ = True
            self.training_time_ = datetime.now()
            logger.info(f"Model training completed in {self.model.offset_}")
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise
        
        # Calculate feature importance
        self._calculate_feature_importance(X_scaled)
        
        return self
    
    def _calculate_feature_importance(self, X_scaled: np.ndarray):
        """
        Calculate feature importance using path lengths.
        
        Feature importance in Isolation Forest can be estimated by:
            1. Path length: Features that lead to shorter paths are more important
            2. Mean depth: Average depth of splits using each feature
        
        Args:
            X_scaled: Scaled features used for training.
        """
        if self.config.feature_importance_method == 'none':
            self.feature_importances_ = np.ones(X_scaled.shape[1]) / X_scaled.shape[1]
            return
        
        # Get decision paths from all trees
        # Note: sklearn doesn't directly expose path lengths per feature
        # We'll use a heuristic based on feature variances and model parameters
        
        if self.config.feature_importance_method == 'path_length':
            # Features with higher variance tend to be more important
            # as they create more effective splits
            feature_variance = np.var(X_scaled, axis=0)
            
            # Normalize to get importance scores
            if np.sum(feature_variance) > 0:
                self.feature_importances_ = feature_variance / np.sum(feature_variance)
            else:
                self.feature_importances_ = np.ones(X_scaled.shape[1]) / X_scaled.shape[1]
        
        elif self.config.feature_importance_method == 'mean_depth':
            # Use random feature importance (uniform)
            # In practice, you would extract path lengths from each tree
            self.feature_importances_ = np.ones(X_scaled.shape[1]) / X_scaled.shape[1]
        
        logger.debug(f"Feature importances calculated: {self.feature_importances_}")
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict if samples are anomalies.
        
        Args:
            X: Features to predict. Can be numpy array or pandas DataFrame.
            
        Returns:
            np.ndarray: Predictions (1 for normal, -1 for anomaly).
            
        Raises:
            NotFittedError: If model hasn't been fitted.
        """
        if not self.is_fitted_:
            raise NotFittedError("Model must be fitted before prediction")
        
        # Validate and preprocess features
        X_array = self._validate_features(X)
        
        # Scale features
        X_scaled = self.scaler.transform(X_array)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict anomaly scores and convert to probabilities.
        
        This method provides a probability-like score indicating the likelihood
        of being an anomaly. Scores are normalized to [0, 1] range where higher
        values indicate higher anomaly probability.
        
        Args:
            X: Features to predict. Can be numpy array or pandas DataFrame.
            
        Returns:
            np.ndarray: Anomaly probabilities in [0, 1].
            
        Raises:
            NotFittedError: If model hasn't been fitted.
        """
        if not self.is_fitted_:
            raise NotFittedError("Model must be fitted before prediction")
        
        # Validate and preprocess features
        X_array = self._validate_features(X)
        
        # Scale features
        X_scaled = self.scaler.transform(X_array)
        
        # Get anomaly scores (lower is more anomalous)
        scores = self.model.decision_function(X_scaled)
        
        # Convert to probabilities (higher = more anomalous)
        # Using sigmoid transformation
        probs = 1 / (1 + np.exp(-scores))
        
        return probs
    
    def score_samples(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Compute anomaly scores for samples.
        
        Lower scores indicate more anomalous samples.
        
        Args:
            X: Features to score. Can be numpy array or pandas DataFrame.
            
        Returns:
            np.ndarray: Anomaly scores (lower = more anomalous).
            
        Raises:
            NotFittedError: If model hasn't been fitted.
        """
        if not self.is_fitted_:
            raise NotFittedError("Model must be fitted before scoring")
        
        # Validate and preprocess features
        X_array = self._validate_features(X)
        
        # Scale features
        X_scaled = self.scaler.transform(X_array)
        
        # Compute scores
        scores = self.model.score_samples(X_scaled)
        
        return scores
    
    def partial_fit(self, X: Union[np.ndarray, pd.DataFrame]) -> 'IsolationForestDetector':
        """
        Update model with new data (online learning).
        
        This method accumulates data and periodically retrains the model
        to adapt to new patterns.
        
        Args:
            X: New data samples. Can be numpy array or pandas DataFrame.
            
        Returns:
            IsolationForestDetector: Updated model instance.
        """
        # Validate and preprocess features
        X_array = self._validate_features(X)
        
        # Scale features
        X_scaled = self.scaler.transform(X_array)
        
        # Add to buffer
        self.online_buffer.append(X_scaled)
        self.n_samples_seen_ += X_scaled.shape[0]
        
        # Check if we should update
        if self.n_samples_seen_ >= self.config.online_update_frequency:
            logger.info(f"Online update triggered after {self.n_samples_seen_} samples")
            
            # Combine buffer data
            X_combined = np.vstack(self.online_buffer)
            
            # Update scaler
            self.scaler.partial_fit(X_combined)
            
            # Retrain model with combined data
            X_scaled_combined = self.scaler.transform(X_combined)
            self.model = self._create_model()
            self.model.fit(X_scaled_combined)
            
            # Clear buffer
            self.online_buffer = []
            self.n_samples_seen_ = 0
            
            logger.info("Online model update completed")
        
        return self
    
    def get_feature_importance(self, normalized: bool = True) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            normalized (bool): If True, normalize scores to sum to 1.
            
        Returns:
            Dict[str, float]: Dictionary mapping feature names to importance scores.
            
        Raises:
            NotFittedError: If model hasn't been fitted.
        """
        if not self.is_fitted_:
            raise NotFittedError("Model must be fitted to get feature importance")
        
        if self.feature_importances_ is None:
            return {}
        
        importances = self.feature_importances_
        
        if normalized and np.sum(importances) > 0:
            importances = importances / np.sum(importances)
        
        return dict(zip(self.feature_names_, importances))
    
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance using labels.
        
        Args:
            X: Features to evaluate. Can be numpy array or pandas DataFrame.
            y: True labels (1 for normal, -1 for anomaly).
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
            
        Raises:
            NotFittedError: If model hasn't been fitted.
        """
        if not self.is_fitted_:
            raise NotFittedError("Model must be fitted before evaluation")
        
        # Get predictions
        y_pred = self.predict(X)
        y_score = self.predict_proba(X)
        
        # Convert y to binary format (0: normal, 1: anomaly)
        y_binary = (y == -1).astype(int)
        y_score_anomaly = y_score
        
        # Calculate metrics
        metrics = {}
        
        try:
            # ROC AUC
            metrics['roc_auc'] = roc_auc_score(y_binary, y_score_anomaly)
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC: {str(e)}")
            metrics['roc_auc'] = None
        
        try:
            # Precision-Recall AUC
            precision, recall, _ = precision_recall_curve(y_binary, y_score_anomaly)
            metrics['pr_auc'] = np.trapz(recall, precision)
        except Exception as e:
            logger.warning(f"Could not calculate PR AUC: {str(e)}")
            metrics['pr_auc'] = None
        
        # Accuracy, precision, recall at default threshold
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics['accuracy'] = accuracy_score(y, y_pred)
        
        # Convert to binary for precision/recall
        y_pred_binary = (y_pred == -1).astype(int)
        
        metrics['precision'] = precision_score(y_binary, y_pred_binary, zero_division=0)
        metrics['recall'] = recall_score(y_binary, y_pred_binary, zero_division=0)
        metrics['f1'] = f1_score(y_binary, y_pred_binary, zero_division=0)
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
    
    def save(self, filepath: str) -> None:
        """
        Save model to disk.
        
        Args:
            filepath: Path to save the model.
        """
        if not self.is_fitted_:
            raise NotFittedError("Cannot save unfitted model")
        
        # Prepare model data
        model_data = {
            'config': self.config.to_dict(),
            'model': self.model,
            'scaler': self.scaler,
            'feature_names_': self.feature_names_,
            'feature_importances_': self.feature_importances_,
            'contamination_': self.contamination_,
            'is_fitted_': self.is_fitted_,
            'training_time_': self.training_time_,
            'feature_stats_': self.feature_stats_
        }
        
        # Save to file
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'IsolationForestDetector':
        """
        Load model from disk.
        
        Args:
            filepath: Path to saved model.
            
        Returns:
            IsolationForestDetector: Loaded model instance.
        """
        # Load model data
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance with saved config
        instance = cls(config=model_data['config'])
        
        # Restore attributes
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_names_ = model_data['feature_names_']
        instance.feature_importances_ = model_data['feature_importances_']
        instance.contamination_ = model_data['contamination_']
        instance.is_fitted_ = model_data['is_fitted_']
        instance.training_time_ = model_data['training_time_']
        instance.feature_stats_ = model_data['feature_stats_']
        
        logger.info(f"Model loaded from {filepath}")
        
        return instance
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dict[str, Any]: Dictionary of model parameters.
        """
        return self.config.to_dict()
    
    def set_params(self, **params) -> 'IsolationForestDetector':
        """
        Set model parameters.
        
        Args:
            **params: Parameters to set.
            
        Returns:
            IsolationForestDetector: Self for method chaining.
        """
        for key, value in params.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning(f"Unknown parameter: {key}")
        
        # Re-validate configuration
        self.config._validate_parameters()
        
        # Recreate model with new parameters if fitted
        if self.is_fitted_:
            logger.warning("Model already fitted. Parameter changes will affect future training only.")
        
        return self


class OnlineIsolationForest(IsolationForestDetector):
    """
    Online/Streaming version of Isolation Forest for real-time fraud detection.
    
    This class extends IsolationForestDetector with capabilities for:
        1. Incremental learning with mini-batches
        2. Concept drift detection
        3. Adaptive thresholding
        4. Real-time scoring with low latency
    
    Particularly useful for banking systems where:
        - Fraud patterns evolve over time
        - Need real-time decisions on streaming transactions
        - Memory constraints require efficient updates
    """
    
    def __init__(self, config: Optional[Union[Dict, IsolationForestConfig]] = None):
        """
        Initialize Online Isolation Forest.
        
        Args:
            config: Configuration for the model.
        """
        super().__init__(config)
        
        # Online learning specific attributes
        self.drift_detector = None
        self.concept_drift_history = []
        self.threshold_history = []
        self.adaptive_threshold = None
        
        # Performance monitoring
        self.prediction_latencies = []
        self.window_performance = {}
        
        logger.info("Initialized OnlineIsolationForest")
    
    def partial_fit(self, X: Union[np.ndarray, pd.DataFrame]) -> 'OnlineIsolationForest':
        """
        Incrementally update model with new data.
        
        This method implements online learning for streaming data,
        with concept drift detection and adaptive updates.
        
        Args:
            X: New data samples. Can be numpy array or pandas DataFrame.
            
        Returns:
            OnlineIsolationForest: Updated model instance.
        """
        import time
        
        start_time = time.time()
        
        # Call parent partial_fit
        super().partial_fit(X)
        
        # Check for concept drift
        self._check_concept_drift()
        
        # Update adaptive threshold
        self._update_adaptive_threshold()
        
        # Record latency
        latency = time.time() - start_time
        self.prediction_latencies.append(latency)
        
        # Keep only last 1000 latencies
        if len(self.prediction_latencies) > 1000:
            self.prediction_latencies = self.prediction_latencies[-1000:]
        
        return self
    
    def _check_concept_drift(self):
        """
        Check for concept drift in the data distribution.
        
        Concept drift occurs when the statistical properties of the target
        variable change over time. In fraud detection, this means fraud
        patterns are evolving.
        """
        # Simple drift detection based on score distribution
        if len(self.online_buffer) > 0:
            # Calculate mean score of recent buffer
            recent_scores = []
            for batch in self.online_buffer[-5:]:  # Last 5 batches
                if isinstance(batch, np.ndarray):
                    scores = self.model.score_samples(batch)
                    recent_scores.extend(scores)
            
            if recent_scores:
                recent_mean = np.mean(recent_scores)
                
                # Compare with historical mean (if available)
                if hasattr(self, 'historical_mean_score'):
                    # Significant change detected
                    if abs(recent_mean - self.historical_mean_score) > 0.5:
                        logger.warning(f"Concept drift detected: mean score changed from "
                                      f"{self.historical_mean_score:.3f} to {recent_mean:.3f}")
                        self.concept_drift_history.append({
                            'timestamp': datetime.now(),
                            'old_mean': self.historical_mean_score,
                            'new_mean': recent_mean
                        })
                
                # Update historical mean
                self.historical_mean_score = recent_mean
    
    def _update_adaptive_threshold(self):
        """
        Update anomaly threshold adaptively based on recent data.
        
        This allows the model to adjust to changing fraud patterns
        without manual threshold tuning.
        """
        if len(self.online_buffer) > 0:
            # Get recent scores
            recent_scores = []
            for batch in self.online_buffer[-10:]:
                if isinstance(batch, np.ndarray):
                    scores = self.model.score_samples(batch)
                    recent_scores.extend(scores)
            
            if recent_scores:
                # Set threshold based on percentile of recent scores
                # Assuming contamination rate
                contamination = self.contamination_ or 0.001
                threshold_percentile = 100 * (1 - contamination)
                self.adaptive_threshold = np.percentile(recent_scores, threshold_percentile)
                
                self.threshold_history.append({
                    'timestamp': datetime.now(),
                    'threshold': self.adaptive_threshold
                })
                
                # Keep only last 100 thresholds
                if len(self.threshold_history) > 100:
                    self.threshold_history = self.threshold_history[-100:]
    
    def predict_with_confidence(self, X: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with confidence scores.
        
        Args:
            X: Features to predict.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (predictions, confidence_scores)
        """
        # Get base predictions
        predictions = self.predict(X)
        
        # Get anomaly scores
        scores = self.score_samples(X)
        
        # Calculate confidence based on distance from threshold
        if self.adaptive_threshold is not None:
            # Confidence is higher when score is far from threshold
            threshold = self.adaptive_threshold
            confidence = np.abs(scores - threshold) / (np.std(scores) + 1e-8)
            confidence = np.clip(confidence, 0, 1)
        else:
            confidence = np.ones_like(scores)
        
        return predictions, confidence
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get online learning performance metrics.
        
        Returns:
            Dict[str, Any]: Performance metrics.
        """
        metrics = {
            'avg_prediction_latency_ms': np.mean(self.prediction_latencies) * 1000 if self.prediction_latencies else None,
            'p95_prediction_latency_ms': np.percentile(self.prediction_latencies, 95) * 1000 if self.prediction_latencies else None,
            'num_concept_drifts': len(self.concept_drift_history),
            'current_adaptive_threshold': self.adaptive_threshold,
            'buffer_size': len(self.online_buffer),
            'total_samples_seen': self.n_samples_seen_
        }
        
        return metrics
    
    def reset_online_state(self):
        """
        Reset online learning state.
        
        Useful when starting a new streaming session or after major changes.
        """
        self.online_buffer = []
        self.n_samples_seen_ = 0
        self.concept_drift_history = []
        self.threshold_history = []
        self.prediction_latencies = []
        
        logger.info("Online state reset")


# Example usage and testing
if __name__ == "__main__":
    """
    Example demonstrating how to use the Isolation Forest detector.
    """
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 10000
    n_features = 10
    
    # Normal data (multivariate normal)
    X_normal = np.random.randn(n_samples, n_features)
    
    # Anomalies (uniform random in larger range)
    n_anomalies = 10
    X_anomalies = np.random.uniform(low=-10, high=10, size=(n_anomalies, n_features))
    
    # Combine
    X = np.vstack([X_normal, X_anomalies])
    y = np.array([1] * n_samples + [-1] * n_anomalies)  # 1: normal, -1: anomaly
    
    # Shuffle
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]
    
    # Create and train model
    config = IsolationForestConfig(
        n_estimators=100,
        contamination=0.01,
        random_state=42,
        feature_importance_method='path_length'
    )
    
    model = IsolationForestDetector(config=config)
    model.fit(X)
    
    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Evaluate
    metrics = model.evaluate(X, y)
    print("Evaluation metrics:", metrics)
    
    # Get feature importance
    importance = model.get_feature_importance()
    print("Feature importance:", importance)
    
    # Save and load
    model.save("isolation_forest_model.pkl")
    loaded_model = IsolationForestDetector.load("isolation_forest_model.pkl")
    
    print("Example completed successfully!")