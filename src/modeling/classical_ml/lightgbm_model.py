"""
LightGBM Model for Fraud Detection
==================================
LightGBM is particularly effective for fraud detection due to:
- Faster training on large datasets
- Native support for categorical features
- Leaf-wise growth for better accuracy
- Built-in handling of missing values
- Excellent performance on imbalanced data
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                           precision_recall_curve, f1_score, confusion_matrix)
import shap
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class FraudLightGBMModel:
    """
    LightGBM model optimized for banking fraud detection.
    
    Key advantages for fraud:
    - Handles categorical variables natively (merchant categories, device types)
    - Efficient with large-scale transaction data
    - Built-in handling of class imbalance
    - Faster training for rapid iteration
    """
    
    def __init__(self, params: Optional[Dict] = None, random_state: int = 42):
        """
        Initialize LightGBM model with fraud-optimized parameters.
        
        Args:
            params: Custom parameters (overrides defaults)
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        
        # Default parameters optimized for fraud detection
        self.default_params = {
            'objective': 'binary',  # Binary classification
            'boosting_type': 'gbdt',  # Gradient boosting decision tree
            'metric': 'aucpr',  # PR-AUC is better for imbalanced data
            'num_leaves': 31,  # Maximum tree leaves
            'max_depth': -1,  # No limit (controlled by num_leaves)
            'learning_rate': 0.05,  # Lower for better convergence
            'feature_fraction': 0.8,  # Feature sampling
            'bagging_fraction': 0.8,  # Row sampling
            'bagging_freq': 5,  # Frequency of bagging
            'lambda_l1': 1.0,  # L1 regularization
            'lambda_l2': 1.0,  # L2 regularization
            'min_gain_to_split': 0.1,  # Minimum gain for split
            'min_child_samples': 20,  # Minimum data in leaf
            'is_unbalance': True,  # Auto-handle class imbalance
            'verbose': -1,  # Silent mode
            'seed': random_state,
            'num_threads': -1,  # Use all CPU cores
            'device_type': 'cpu'  # Use CPU (GPU optional)
        }
        
        if params:
            self.default_params.update(params)
        
        self.params = self.default_params
        self.model = None
        self.feature_names = None
        self.categorical_features = None
        self.best_iteration = None
        self.feature_importance = None
        self.threshold = 0.5
        self.explainer = None
        self.training_history = {}
        
        logger.info("Initialized FraudLightGBMModel with fraud-optimized parameters")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None,
              categorical_features: Optional[List[str]] = None,
              use_early_stopping: bool = True,
              early_stopping_rounds: int = 50,
              verbose_eval: int = 100) -> 'FraudLightGBMModel':
        """
        Train LightGBM model with support for categorical features.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            categorical_features: List of categorical column names
            use_early_stopping: Whether to use early stopping
            early_stopping_rounds: Rounds to wait before early stopping
            verbose_eval: Verbosity level
        
        Returns:
            Trained model instance
        """
        logger.info("Starting LightGBM training...")
        
        self.feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else None
        self.categorical_features = categorical_features
        
        # Convert categorical features to categorical type for LightGBM
        if categorical_features and hasattr(X_train, 'columns'):
            for col in categorical_features:
                if col in X_train.columns:
                    X_train[col] = X_train[col].astype('category')
                    if X_val is not None:
                        X_val[col] = X_val[col].astype('category')
        
        # Prepare datasets
        train_data = lgb.Dataset(
            X_train, 
            label=y_train,
            feature_name=self.feature_names,
            categorical_feature=categorical_features,
            free_raw_data=False  # Keep data for later use
        )
        
        # Validation set for early stopping
        if X_val is not None and y_val is not None and use_early_stopping:
            val_data = lgb.Dataset(
                X_val, 
                label=y_val,
                feature_name=self.feature_names,
                categorical_feature=categorical_features,
                reference=train_data
            )
            
            # Train with early stopping
            self.model = lgb.train(
                params=self.params,
                train_set=train_data,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'eval'],
                num_boost_round=1000,
                callbacks=[
                    lgb.early_stopping(early_stopping_rounds),
                    lgb.log_evaluation(verbose_eval)
                ]
            )
            
            self.best_iteration = self.model.best_iteration
            logger.info(f"Best iteration: {self.best_iteration}")
            
        else:
            # Train without validation
            self.model = lgb.train(
                params=self.params,
                train_set=train_data,
                num_boost_round=100,
                callbacks=[lgb.log_evaluation(verbose_eval)]
            )
        
        # Calculate feature importance
        self._calculate_feature_importance()
        
        # Create SHAP explainer
        try:
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("SHAP explainer created successfully")
        except Exception as e:
            logger.warning(f"Could not create SHAP explainer: {e}")
        
        return self
    
    def _calculate_feature_importance(self):
        """
        Calculate multiple types of feature importance.
        LightGBM provides 'split' and 'gain' importance.
        """
        if self.model is None:
            return
        
        importance_types = ['split', 'gain']
        self.feature_importance = {}
        
        for imp_type in importance_types:
            importance_dict = self.model.feature_importance(importance_type=imp_type)
            
            if self.feature_names:
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importance_dict
                }).sort_values('importance', ascending=False)
            else:
                importance_df = pd.DataFrame(
                    importance_dict,
                    columns=['importance']
                )
            
            self.feature_importance[imp_type] = importance_df
            
            logger.info(f"Top 5 features by {imp_type}:")
            logger.info(f"\n{importance_df.head(5).to_string()}")
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict fraud probabilities.
        
        Args:
            X: Input features
        
        Returns:
            Array of shape (n_samples, 2) with probabilities
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        # Handle categorical features
        if self.categorical_features and isinstance(X, pd.DataFrame):
            X = X.copy()
            for col in self.categorical_features:
                if col in X.columns:
                    X[col] = X[col].astype('category')
        
        # Get raw predictions
        raw_probs = self.model.predict(X, num_iteration=self.best_iteration)
        
        # Return 2-column format
        return np.column_stack([1 - raw_probs, raw_probs])
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray],
                threshold: Optional[float] = None) -> np.ndarray:
        """Predict binary labels using specified threshold."""
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
        
        from sklearn.metrics import precision_score, recall_score
        
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
        
        logger.info(f"Optimal threshold: {self.threshold:.3f}")
        return self.threshold
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series,
                 threshold: Optional[float] = None) -> Dict[str, float]:
        """Comprehensive model evaluation."""
        if threshold is None:
            threshold = self.threshold
        
        probs = self.predict_proba(X_test)[:, 1]
        preds = (probs >= threshold).astype(int)
        
        metrics = {}
        
        # Basic metrics
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score
        metrics['accuracy'] = accuracy_score(y_test, preds)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_test, preds)
        metrics['precision'] = precision_score(y_test, preds, zero_division=0)
        metrics['recall'] = recall_score(y_test, preds, zero_division=0)
        metrics['f1_score'] = f1_score(y_test, preds, zero_division=0)
        
        # AUC metrics
        metrics['roc_auc'] = roc_auc_score(y_test, probs)
        metrics['pr_auc'] = average_precision_score(y_test, probs)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        metrics['true_positives'] = tp
        
        # Derived metrics
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'params': self.params,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'best_iteration': self.best_iteration,
            'threshold': self.threshold,
            'feature_importance': self.feature_importance,
            'random_state': self.random_state,
            'saved_at': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> 'FraudLightGBMModel':
        """Load trained model from disk."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.params = model_data['params']
        self.feature_names = model_data['feature_names']
        self.categorical_features = model_data['categorical_features']
        self.best_iteration = model_data['best_iteration']
        self.threshold = model_data['threshold']
        self.feature_importance = model_data['feature_importance']
        self.random_state = model_data['random_state']
        
        logger.info(f"Model loaded from {filepath}")
        return self
    
    def get_feature_importance_plot(self, importance_type: str = 'gain',
                                     top_n: int = 20) -> Dict:
        """Get feature importance data for visualization."""
        if self.feature_importance is None:
            self._calculate_feature_importance()
        
        importance_df = self.feature_importance.get(importance_type, pd.DataFrame())
        
        if importance_df.empty:
            return {'error': f'No {importance_type} importance available'}
        
        top_features = importance_df.head(top_n)
        
        return {
            'features': top_features['feature'].tolist(),
            'importance_scores': top_features['importance'].tolist(),
            'importance_type': importance_type
        }