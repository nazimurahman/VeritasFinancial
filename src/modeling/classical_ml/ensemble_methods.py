"""
Ensemble Methods for Fraud Detection
=====================================
Ensemble methods combine multiple models to achieve better performance.
For fraud detection, ensembles are crucial because:
- Different models capture different fraud patterns
- Reduces false positives through consensus
- More robust to concept drift
- Better generalization on imbalanced data
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                           f1_score, precision_score, recall_score)
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from .xgboost_model import FraudXGBoostModel
from .lightgbm_model import FraudLightGBMModel

logger = logging.getLogger(__name__)


class FraudEnsembleModel(BaseEstimator, ClassifierMixin):
    """
    Ensemble model combining multiple fraud detection algorithms.
    
    This ensemble combines the strengths of different models:
    - XGBoost: Robust with regularization, good for tabular data
    - LightGBM: Fast, handles categorical features well
    - Additional models can be added (Random Forest, etc.)
    
    Two ensemble strategies:
    1. Voting: Simple averaging of predictions
    2. Stacking: Meta-model learns to combine predictions optimally
    """
    
    def __init__(self, models: Optional[Dict] = None,
                 ensemble_type: str = 'voting',
                 voting: str = 'soft',
                 meta_model: Optional[Any] = None,
                 weights: Optional[List[float]] = None,
                 random_state: int = 42):
        """
        Initialize ensemble model.
        
        Args:
            models: Dictionary of models {name: model_instance}
            ensemble_type: 'voting' or 'stacking'
            voting: 'soft' (probabilities) or 'hard' (class labels)
            meta_model: Meta-model for stacking (if None, uses LogisticRegression)
            weights: Model weights for weighted voting
            random_state: Random seed
        """
        self.random_state = random_state
        
        # Default models if none provided
        if models is None:
            self.models = {
                'xgboost': FraudXGBoostModel(random_state=random_state),
                'lightgbm': FraudLightGBMModel(random_state=random_state)
            }
        else:
            self.models = models
        
        self.ensemble_type = ensemble_type
        self.voting = voting
        self.meta_model = meta_model
        self.weights = weights
        self.ensemble = None
        self.feature_names = None
        self.threshold = 0.5
        self.training_history = {}
        
        logger.info(f"Initialized FraudEnsembleModel with {len(self.models)} models")
        logger.info(f"Ensemble type: {ensemble_type}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None,
              categorical_features: Optional[List[str]] = None,
              use_early_stopping: bool = True) -> 'FraudEnsembleModel':
        """
        Train all base models and build ensemble.
        
        For stacking, uses k-fold cross-validation to generate
        out-of-fold predictions for training the meta-model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            categorical_features: Categorical feature names
            use_early_stopping: Whether to use early stopping
        
        Returns:
            Trained ensemble model
        """
        logger.info("Starting ensemble training...")
        
        self.feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else None
        
        # Train all base models
        for name, model in self.models.items():
            logger.info(f"Training base model: {name}")
            
            if hasattr(model, 'train'):
                # Our custom models
                model.train(
                    X_train, y_train,
                    X_val, y_val,
                    categorical_features=categorical_features,
                    use_early_stopping=use_early_stopping
                )
            else:
                # sklearn-compatible models
                model.fit(X_train, y_train)
        
        # Build ensemble based on type
        if self.ensemble_type == 'voting':
            self._build_voting_ensemble()
        elif self.ensemble_type == 'stacking':
            self._build_stacking_ensemble(X_train, y_train)
        else:
            raise ValueError(f"Unsupported ensemble type: {self.ensemble_type}")
        
        return self
    
    def _build_voting_ensemble(self):
        """
        Build voting ensemble using sklearn's VotingClassifier.
        """
        from sklearn.ensemble import VotingClassifier
        
        # Prepare estimators list
        estimators = []
        for name, model in self.models.items():
            if hasattr(model, 'model') and hasattr(model.model, 'predict_proba'):
                # Extract underlying model from our wrappers
                estimators.append((name, model.model))
            else:
                estimators.append((name, model))
        
        # Create voting classifier
        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting=self.voting,
            weights=self.weights,
            n_jobs=-1
        )
        
        logger.info("Voting ensemble built successfully")
    
    def _build_stacking_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Build stacking ensemble with meta-model.
        
        Uses k-fold cross-validation to generate out-of-fold predictions
        for training the meta-model.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        
        if self.meta_model is None:
            # Default meta-model: logistic regression with class weights
            self.meta_model = LogisticRegression(
                class_weight='balanced',
                random_state=self.random_state,
                max_iter=1000
            )
        
        n_samples = len(X_train)
        n_models = len(self.models)
        
        # Generate out-of-fold predictions for meta-model training
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        meta_features = np.zeros((n_samples, n_models))
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
            logger.info(f"Generating meta-features - Fold {fold + 1}/5")
            
            X_train_fold = X_train.iloc[train_idx]
            y_train_fold = y_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            
            # Train models on this fold
            fold_models = []
            for name, model in self.models.items():
                if hasattr(model, 'train'):
                    # Clone and train custom model
                    import copy
                    fold_model = copy.deepcopy(model)
                    fold_model.train(X_train_fold, y_train_fold, use_early_stopping=False)
                    fold_models.append(fold_model)
                else:
                    # Clone and train sklearn model
                    from sklearn.base import clone
                    fold_model = clone(model)
                    fold_model.fit(X_train_fold, y_train_fold)
                    fold_models.append(fold_model)
            
            # Generate predictions for validation fold
            for i, fold_model in enumerate(fold_models):
                if hasattr(fold_model, 'predict_proba'):
                    probs = fold_model.predict_proba(X_val_fold)[:, 1]
                else:
                    probs = fold_model.predict(X_val_fold)
                meta_features[val_idx, i] = probs
        
        # Train meta-model on out-of-fold predictions
        self.meta_model.fit(meta_features, y_train)
        
        # Now train all base models on full dataset
        logger.info("Training base models on full dataset...")
        for name, model in self.models.items():
            if hasattr(model, 'train'):
                model.train(X_train, y_train, use_early_stopping=False)
            else:
                model.fit(X_train, y_train)
        
        logger.info("Stacking ensemble built successfully")
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict fraud probabilities using ensemble.
        
        Returns:
            Array of shape (n_samples, 2) with probabilities
        """
        if self.ensemble_type == 'voting':
            # Use voting ensemble
            if self.voting == 'soft':
                probs = self.ensemble.predict_proba(X)
                return probs
            else:
                # Hard voting returns classes, convert to probabilities
                preds = self.ensemble.predict(X)
                probs = np.zeros((len(X), 2))
                probs[:, 0] = 1 - preds
                probs[:, 1] = preds
                return probs
                
        elif self.ensemble_type == 'stacking':
            # Generate base model predictions
            n_samples = len(X)
            n_models = len(self.models)
            meta_features = np.zeros((n_samples, n_models))
            
            for i, (name, model) in enumerate(self.models.items()):
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X)[:, 1]
                else:
                    probs = model.predict(X)
                meta_features[:, i] = probs
            
            # Meta-model prediction
            if hasattr(self.meta_model, 'predict_proba'):
                return self.meta_model.predict_proba(meta_features)
            else:
                preds = self.meta_model.predict(meta_features)
                probs = np.zeros((len(X), 2))
                probs[:, 0] = 1 - preds
                probs[:, 1] = preds
                return probs
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray],
                threshold: Optional[float] = None) -> np.ndarray:
        """Predict binary labels."""
        if threshold is None:
            threshold = self.threshold
        
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)
    
    def find_optimal_threshold(self, X_val: pd.DataFrame, y_val: pd.Series,
                                metric: str = 'f1') -> float:
        """Find optimal classification threshold."""
        probs = self.predict_proba(X_val)[:, 1]
        
        thresholds = np.arange(0.1, 0.9, 0.01)
        scores = []
        
        for threshold in thresholds:
            preds = (probs >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_val, preds, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_val, preds, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_val, preds, zero_division=0)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            scores.append(score)
        
        optimal_idx = np.argmax(scores)
        self.threshold = thresholds[optimal_idx]
        
        logger.info(f"Ensemble optimal threshold: {self.threshold:.3f}")
        return self.threshold
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series,
                 threshold: Optional[float] = None) -> Dict[str, float]:
        """Comprehensive ensemble evaluation."""
        if threshold is None:
            threshold = self.threshold
        
        probs = self.predict_proba(X_test)[:, 1]
        preds = (probs >= threshold).astype(int)
        
        metrics = {}
        
        # Basic metrics
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        metrics['accuracy'] = accuracy_score(y_test, preds)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_test, preds)
        metrics['precision'] = precision_score(y_test, preds, zero_division=0)
        metrics['recall'] = recall_score(y_test, preds, zero_division=0)
        metrics['f1_score'] = f1_score(y_test, preds, zero_division=0)
        
        # AUC metrics
        metrics['roc_auc'] = roc_auc_score(y_test, probs)
        metrics['pr_auc'] = average_precision_score(y_test, probs)
        
        # Individual model performance for comparison
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                model_probs = model.predict_proba(X_test)[:, 1]
            else:
                model_probs = model.predict(X_test)
            
            metrics[f'{name}_roc_auc'] = roc_auc_score(y_test, model_probs)
            metrics[f'{name}_pr_auc'] = average_precision_score(y_test, model_probs)
        
        logger.info("Ensemble evaluation complete")
        for metric, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"{metric:25s}: {value:.4f}")
        
        return metrics
    
    def get_feature_importance(self) -> Dict[str, pd.DataFrame]:
        """
        Get feature importance from all base models.
        
        Returns:
            Dictionary with model names as keys and importance DataFrames as values
        """
        importance_dict = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importance'):
                importance_dict[name] = model.feature_importance
            elif hasattr(model, 'feature_importances_'):
                # sklearn models
                if self.feature_names:
                    importance_df = pd.DataFrame({
                        'feature': self.feature_names,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                else:
                    importance_df = pd.DataFrame(
                        model.feature_importances_,
                        columns=['importance']
                    )
                importance_dict[name] = {'gain': importance_df}
        
        return importance_dict
    
    def save_model(self, filepath: str) -> None:
        """Save ensemble model to disk."""
        model_data = {
            'models': self.models,
            'ensemble_type': self.ensemble_type,
            'voting': self.voting,
            'meta_model': self.meta_model,
            'ensemble': self.ensemble,
            'weights': self.weights,
            'feature_names': self.feature_names,
            'threshold': self.threshold,
            'random_state': self.random_state,
            'saved_at': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Ensemble model saved to {filepath}")
    
    def load_model(self, filepath: str) -> 'FraudEnsembleModel':
        """Load ensemble model from disk."""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.ensemble_type = model_data['ensemble_type']
        self.voting = model_data['voting']
        self.meta_model = model_data['meta_model']
        self.ensemble = model_data['ensemble']
        self.weights = model_data['weights']
        self.feature_names = model_data['feature_names']
        self.threshold = model_data['threshold']
        self.random_state = model_data['random_state']
        
        logger.info(f"Ensemble model loaded from {filepath}")
        return self


class FraudStackingEnsemble(FraudEnsembleModel):
    """
    Specialized stacking ensemble for fraud detection.
    
    Uses multiple base models and a meta-model optimized for
    imbalanced classification.
    """
    
    def __init__(self, base_models: Optional[List] = None,
                 meta_model: Optional[Any] = None,
                 use_probabilities: bool = True,
                 random_state: int = 42):
        """
        Initialize stacking ensemble.
        
        Args:
            base_models: List of base models
            meta_model: Meta-model (if None, uses calibrated logistic regression)
            use_probabilities: Use probabilities (True) or class labels (False)
            random_state: Random seed
        """
        super().__init__(
            models={f'model_{i}': model for i, model in enumerate(base_models)} if base_models else None,
            ensemble_type='stacking',
            meta_model=meta_model,
            random_state=random_state
        )
        self.use_probabilities = use_probabilities


class FraudVotingEnsemble(FraudEnsembleModel):
    """
    Specialized voting ensemble for fraud detection.
    
    Combines predictions from multiple fraud detection models
    with weights optimized for imbalanced data.
    """
    
    def __init__(self, base_models: Optional[Dict] = None,
                 voting: str = 'soft',
                 weights: Optional[List[float]] = None,
                 optimize_weights: bool = True,
                 random_state: int = 42):
        """
        Initialize voting ensemble.
        
        Args:
            base_models: Dictionary of models
            voting: 'soft' or 'hard'
            weights: Model weights (if None, optimized automatically)
            optimize_weights: Whether to optimize weights on validation data
            random_state: Random seed
        """
        super().__init__(
            models=base_models,
            ensemble_type='voting',
            voting=voting,
            weights=weights,
            random_state=random_state
        )
        self.optimize_weights = optimize_weights
    
    def optimize_ensemble_weights(self, X_val: pd.DataFrame, y_val: pd.Series,
                                  metric: str = 'pr_auc') -> List[float]:
        """
        Optimize voting weights for best performance.
        
        Uses grid search to find optimal weights that maximize
        the chosen metric on validation data.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            metric: Metric to optimize ('pr_auc', 'roc_auc', 'f1')
        
        Returns:
            Optimal weights
        """
        n_models = len(self.models)
        
        # Generate predictions from all models
        model_probs = []
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_val)[:, 1]
            else:
                probs = model.predict(X_val)
            model_probs.append(probs)
        
        # Grid search over weights
        best_score = -np.inf
        best_weights = None
        
        # Simple grid search for 2-3 models
        if n_models == 2:
            for w1 in np.arange(0, 1.1, 0.1):
                w2 = 1 - w1
                weights = [w1, w2]
                
                # Weighted average
                ensemble_probs = np.zeros_like(model_probs[0])
                for i, probs in enumerate(model_probs):
                    ensemble_probs += weights[i] * probs
                
                # Calculate metric
                if metric == 'pr_auc':
                    score = average_precision_score(y_val, ensemble_probs)
                elif metric == 'roc_auc':
                    score = roc_auc_score(y_val, ensemble_probs)
                elif metric == 'f1':
                    preds = (ensemble_probs >= 0.5).astype(int)
                    score = f1_score(y_val, preds)
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
                
                if score > best_score:
                    best_score = score
                    best_weights = weights
        
        elif n_models == 3:
            # 3-model grid search
            for w1 in np.arange(0, 1.1, 0.2):
                for w2 in np.arange(0, 1.1 - w1, 0.2):
                    w3 = 1 - w1 - w2
                    if w3 < 0:
                        continue
                    weights = [w1, w2, w3]
                    
                    ensemble_probs = np.zeros_like(model_probs[0])
                    for i, probs in enumerate(model_probs):
                        ensemble_probs += weights[i] * probs
                    
                    if metric == 'pr_auc':
                        score = average_precision_score(y_val, ensemble_probs)
                    elif metric == 'roc_auc':
                        score = roc_auc_score(y_val, ensemble_probs)
                    elif metric == 'f1':
                        preds = (ensemble_probs >= 0.5).astype(int)
                        score = f1_score(y_val, preds)
                    else:
                        raise ValueError(f"Unsupported metric: {metric}")
                    
                    if score > best_score:
                        best_score = score
                        best_weights = weights
        else:
            # For more models, use simple heuristic
            # Give more weight to better performing models
            val_scores = []
            for i, probs in enumerate(model_probs):
                score = average_precision_score(y_val, probs)
                val_scores.append(score)
            
            # Normalize scores to weights
            best_weights = np.array(val_scores) / np.sum(val_scores)
        
        self.weights = best_weights
        logger.info(f"Optimized weights: {best_weights}")
        logger.info(f"Best {metric} score: {best_score:.4f}")
        
        return best_weights