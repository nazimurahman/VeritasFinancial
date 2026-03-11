"""
XGBoost Model for Fraud Detection
==================================
This module implements an optimized XGBoost classifier specifically designed
for fraud detection in banking transactions. It includes advanced features like
class imbalance handling, hyperparameter tuning, and model interpretability.

Key Features:
- Handles severe class imbalance (typical in fraud: 99.9% non-fraud, 0.1% fraud)
- Custom evaluation metrics focused on fraud recall and precision
- Built-in cross-validation and early stopping
- SHAP-based model explainability
- Production-ready serialization
"""

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                           precision_recall_curve, f1_score, confusion_matrix)
import shap
import joblib
import json
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up logging for production monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FraudXGBoostModel:
    """
    Advanced XGBoost model optimized for fraud detection in banking.
    
    This class provides a complete pipeline for training, evaluating, and
    deploying XGBoost models for fraud detection. It incorporates best practices
    from real-world banking fraud systems.
    
    Attributes:
        model: Trained XGBoost model instance
        params: Model hyperparameters
        feature_names: List of feature names used for training
        scaler: Optional feature scaler (for preprocessing)
        best_iteration: Best boosting round from early stopping
        feature_importance: Feature importance scores
        threshold: Optimal classification threshold
        explainer: SHAP explainer for model interpretability
    """
    
    def __init__(self, params: Optional[Dict] = None, random_state: int = 42):
        """
        Initialize the XGBoost model with custom parameters.
        
        Args:
            params: Dictionary of XGBoost parameters. If None, uses optimized defaults
            random_state: Random seed for reproducibility
        
        The default parameters are specifically tuned for fraud detection:
        - scale_pos_weight: Handles class imbalance (calculated from data)
        - max_depth: Limited to prevent overfitting on imbalanced data
        - learning_rate: Lower for better convergence on minority class
        - eval_metric: PR-AUC is better than ROC-AUC for imbalanced data
        """
        self.random_state = random_state
        
        # Default parameters optimized for fraud detection
        self.default_params = {
            'objective': 'binary:logistic',  # Binary classification
            'booster': 'gbtree',  # Tree-based booster
            'eval_metric': ['aucpr', 'logloss'],  # Focus on PR-AUC for imbalance
            'max_depth': 6,  # Moderate depth to prevent overfitting
            'learning_rate': 0.05,  # Lower learning rate for stability
            'subsample': 0.8,  # Row sampling to prevent overfitting
            'colsample_bytree': 0.8,  # Feature sampling
            'colsample_bylevel': 0.8,  # Feature sampling per level
            'min_child_weight': 5,  # Minimum sum of instance weight in child
            'reg_alpha': 1.0,  # L1 regularization
            'reg_lambda': 1.0,  # L2 regularization
            'gamma': 0.1,  # Minimum loss reduction for split
            'scale_pos_weight': 1,  # Will be auto-calculated from data
            'random_state': random_state,
            'n_jobs': -1,  # Use all CPU cores
            'verbosity': 0,  # Silent mode
            'tree_method': 'hist',  # Histogram-based for faster training
            'predictor': 'cpu_predictor'  # Use CPU for inference
        }
        
        # Update with user-provided parameters
        if params:
            self.default_params.update(params)
        
        self.params = self.default_params
        self.model = None
        self.feature_names = None
        self.best_iteration = None
        self.feature_importance = None
        self.threshold = 0.5  # Default threshold
        self.explainer = None
        self.training_history = {}
        
        logger.info("Initialized FraudXGBoostModel with optimized parameters")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None,
              use_early_stopping: bool = True,
              early_stopping_rounds: int = 50,
              verbose_eval: int = 100,
              calculate_scale_pos_weight: bool = True) -> 'FraudXGBoostModel':
        """
        Train the XGBoost model with advanced features.
        
        This method implements best practices for fraud detection:
        - Automatic calculation of scale_pos_weight for class imbalance
        - Early stopping to prevent overfitting
        - Validation set monitoring
        - Feature importance calculation
        
        Args:
            X_train: Training features DataFrame
            y_train: Training labels Series
            X_val: Validation features (optional, for early stopping)
            y_val: Validation labels (optional, for early stopping)
            use_early_stopping: Whether to use early stopping
            early_stopping_rounds: Rounds to wait before early stopping
            verbose_eval: Verbosity level for training output
            calculate_scale_pos_weight: Auto-calculate class weight
        
        Returns:
            self: Trained model instance
        
        Raises:
            ValueError: If input data is invalid
        """
        
        logger.info("Starting model training...")
        
        # Store feature names for later use
        self.feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else None
        
        # Convert to numpy arrays for XGBoost
        X_train_array = X_train.values if hasattr(X_train, 'values') else X_train
        y_train_array = y_train.values if hasattr(y_train, 'values') else y_train
        
        # Calculate class weights for imbalance handling
        if calculate_scale_pos_weight:
            # Typical fraud ratio: 0.1% to 1% of transactions
            neg_count = np.sum(y_train_array == 0)
            pos_count = np.sum(y_train_array == 1)
            
            if pos_count > 0:
                # scale_pos_weight = negative_samples / positive_samples
                # This gives more weight to the minority class
                self.params['scale_pos_weight'] = neg_count / pos_count
                logger.info(f"Calculated scale_pos_weight: {self.params['scale_pos_weight']:.2f}")
                logger.info(f"Class distribution - Non-fraud: {neg_count}, Fraud: {pos_count}")
            else:
                logger.warning("No positive samples in training data!")
                self.params['scale_pos_weight'] = 1
        
        # Prepare DMatrix for XGBoost (optimized data structure)
        dtrain = xgb.DMatrix(X_train_array, label=y_train_array, 
                            feature_names=self.feature_names)
        
        # If validation data is provided
        if X_val is not None and y_val is not None and use_early_stopping:
            X_val_array = X_val.values if hasattr(X_val, 'values') else X_val
            y_val_array = y_val.values if hasattr(y_val, 'values') else y_val
            
            dval = xgb.DMatrix(X_val_array, label=y_val_array,
                              feature_names=self.feature_names)
            
            # Watchlist for monitoring training progress
            watchlist = [(dtrain, 'train'), (dval, 'eval')]
            
            logger.info("Training with early stopping...")
            
            # Train with early stopping
            self.model = xgb.train(
                params=self.params,
                dtrain=dtrain,
                num_boost_round=1000,  # Maximum rounds
                evals=watchlist,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=verbose_eval,
                xgb_model=None
            )
            
            # Store best iteration for inference
            self.best_iteration = self.model.best_iteration
            logger.info(f"Best iteration: {self.best_iteration}")
            
        else:
            logger.info("Training without validation (fixed rounds)...")
            
            # Train without early stopping (fixed number of rounds)
            self.model = xgb.train(
                params=self.params,
                dtrain=dtrain,
                num_boost_round=100,
                verbose_eval=verbose_eval
            )
        
        # Calculate feature importance
        self._calculate_feature_importance()
        
        # Create SHAP explainer for model interpretability
        try:
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("SHAP explainer created successfully")
        except Exception as e:
            logger.warning(f"Could not create SHAP explainer: {e}")
            self.explainer = None
        
        logger.info("Model training completed successfully")
        return self
    
    def _calculate_feature_importance(self) -> None:
        """
        Calculate and store multiple types of feature importance.
        
        XGBoost provides several importance metrics:
        - weight: number of times a feature is used to split
        - gain: average gain when feature is used in splits
        - cover: average coverage of feature
        """
        if self.model is None:
            raise ValueError("Model must be trained before calculating importance")
        
        importance_types = ['weight', 'gain', 'cover']
        self.feature_importance = {}
        
        for imp_type in importance_types:
            try:
                importance_dict = self.model.get_score(importance_type=imp_type)
                
                # Convert to DataFrame for easier analysis
                if self.feature_names:
                    importance_df = pd.DataFrame({
                        'feature': self.feature_names,
                        'importance': [importance_dict.get(f, 0) for f in self.feature_names]
                    }).sort_values('importance', ascending=False)
                else:
                    importance_df = pd.DataFrame(
                        list(importance_dict.items()),
                        columns=['feature', 'importance']
                    ).sort_values('importance', ascending=False)
                
                self.feature_importance[imp_type] = importance_df
                
                logger.info(f"Top 5 features by {imp_type}:")
                logger.info(f"\n{importance_df.head(5).to_string()}")
                
            except Exception as e:
                logger.warning(f"Could not calculate {imp_type} importance: {e}")
                self.feature_importance[imp_type] = pd.DataFrame()
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict fraud probabilities for input samples.
        
        Args:
            X: Input features
        
        Returns:
            Array of fraud probabilities (shape: n_samples, 2)
            Column 0: Probability of non-fraud
            Column 1: Probability of fraud
        
        Raises:
            ValueError: If model is not trained
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert to DMatrix for optimal performance
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        dtest = xgb.DMatrix(X_array, feature_names=self.feature_names)
        
        # Get raw predictions
        raw_probs = self.model.predict(dtest, iteration_range=(0, self.best_iteration))
        
        # Convert to 2-column format (non-fraud prob, fraud prob)
        probs = np.column_stack([1 - raw_probs, raw_probs])
        
        return probs
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray], 
                threshold: Optional[float] = None) -> np.ndarray:
        """
        Predict binary fraud labels using specified threshold.
        
        Args:
            X: Input features
            threshold: Classification threshold (if None, uses self.threshold)
        
        Returns:
            Binary predictions (0: non-fraud, 1: fraud)
        """
        if threshold is None:
            threshold = self.threshold
        
        probs = self.predict_proba(X)[:, 1]  # Get fraud probabilities
        return (probs >= threshold).astype(int)
    
    def find_optimal_threshold(self, X_val: pd.DataFrame, y_val: pd.Series,
                                metric: str = 'f1') -> float:
        """
        Find optimal classification threshold for imbalanced data.
        
        In fraud detection, default threshold (0.5) is often suboptimal due to
        class imbalance. This method finds the best threshold based on business
        metrics like F1-score, precision, or recall.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            metric: Optimization metric ('f1', 'precision', 'recall', 'gmean')
        
        Returns:
            Optimal threshold value
        
        Raises:
            ValueError: If metric is not supported
        """
        # Get fraud probabilities
        probs = self.predict_proba(X_val)[:, 1]
        
        # Test thresholds from 0.1 to 0.9
        thresholds = np.arange(0.1, 0.9, 0.01)
        scores = []
        
        for threshold in thresholds:
            preds = (probs >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_val, preds, zero_division=0)
            elif metric == 'precision':
                from sklearn.metrics import precision_score
                score = precision_score(y_val, preds, zero_division=0)
            elif metric == 'recall':
                from sklearn.metrics import recall_score
                score = recall_score(y_val, preds, zero_division=0)
            elif metric == 'gmean':
                # Geometric mean of precision and recall
                precision = precision_score(y_val, preds, zero_division=0)
                recall = recall_score(y_val, preds, zero_division=0)
                score = np.sqrt(precision * recall) if precision * recall > 0 else 0
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            scores.append(score)
        
        # Find optimal threshold
        optimal_idx = np.argmax(scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_score = scores[optimal_idx]
        
        self.threshold = optimal_threshold
        
        logger.info(f"Optimal threshold found: {optimal_threshold:.3f}")
        logger.info(f"Optimal {metric} score: {optimal_score:.4f}")
        
        # Also show metrics at default threshold for comparison
        default_preds = (probs >= 0.5).astype(int)
        default_f1 = f1_score(y_val, default_preds, zero_division=0)
        logger.info(f"Default threshold (0.5) F1 score: {default_f1:.4f}")
        
        return optimal_threshold
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series,
                 threshold: Optional[float] = None) -> Dict[str, float]:
        """
        Comprehensive model evaluation with multiple metrics.
        
        For fraud detection, we need metrics that handle class imbalance:
        - PR-AUC (Precision-Recall AUC) is better than ROC-AUC
        - F1-score balances precision and recall
        - Precision@K for business impact
        
        Args:
            X_test: Test features
            y_test: Test labels
            threshold: Classification threshold (if None, uses self.threshold)
        
        Returns:
            Dictionary of evaluation metrics
        """
        if threshold is None:
            threshold = self.threshold
        
        # Get predictions
        probs = self.predict_proba(X_test)[:, 1]
        preds = (probs >= threshold).astype(int)
        
        # Calculate metrics
        metrics = {}
        
        # Basic metrics
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        metrics['accuracy'] = accuracy_score(y_test, preds)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_test, preds)
        
        # Precision/Recall based metrics
        metrics['precision'] = precision_score(y_test, preds, zero_division=0)
        metrics['recall'] = recall_score(y_test, preds, zero_division=0)
        metrics['f1_score'] = f1_score(y_test, preds, zero_division=0)
        metrics['specificity'] = self._calculate_specificity(y_test, preds)
        
        # AUC metrics (threshold independent)
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
        
        # Business metrics
        if tp + fp > 0:
            metrics['precision_at_recall'] = tp / (tp + fp)  # Same as precision
        else:
            metrics['precision_at_recall'] = 0
        
        # Log results
        logger.info("=" * 50)
        logger.info("Model Evaluation Results")
        logger.info("=" * 50)
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"{metric:25s}: {value:.4f}")
        
        return metrics
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate specificity (true negative rate).
        
        Specificity measures the proportion of actual negatives correctly identified.
        Important for fraud detection to minimize false positives.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            Specificity score
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    
    def explain_prediction(self, X_sample: pd.DataFrame, 
                           feature_names: Optional[List[str]] = None) -> Dict:
        """
        Explain individual predictions using SHAP values.
        
        For regulatory compliance in banking, we need to explain why
        a transaction was flagged as fraudulent.
        
        Args:
            X_sample: Single sample or small batch to explain
            feature_names: Names of features (uses stored names if None)
        
        Returns:
            Dictionary containing explanation data
        
        Raises:
            ValueError: If explainer is not available
        """
        if self.explainer is None:
            raise ValueError("SHAP explainer not available. Train model with explainer.")
        
        if feature_names is None:
            feature_names = self.feature_names
        
        # Convert to numpy array if DataFrame
        if isinstance(X_sample, pd.DataFrame):
            X_array = X_sample.values
            if feature_names is None:
                feature_names = X_sample.columns.tolist()
        else:
            X_array = X_sample
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_array)
        
        # Get base value (expected value)
        expected_value = self.explainer.expected_value
        
        explanations = []
        
        # For each sample in the batch
        for i in range(len(X_array) if len(X_array.shape) > 1 else 1):
            if len(X_array.shape) == 1:
                # Single sample
                shap_vals = shap_values
                features = X_array
            else:
                # Multiple samples
                shap_vals = shap_values[i]
                features = X_array[i]
            
            # Sort features by absolute SHAP value
            feature_impacts = []
            for j, (feat_name, shap_val, feat_val) in enumerate(
                zip(feature_names, shap_vals, features)
            ):
                impact = {
                    'feature': feat_name,
                    'shap_value': float(shap_val),
                    'feature_value': float(feat_val),
                    'impact_direction': 'positive' if shap_val > 0 else 'negative',
                    'magnitude': abs(float(shap_val))
                }
                feature_impacts.append(impact)
            
            # Sort by magnitude
            feature_impacts.sort(key=lambda x: x['magnitude'], reverse=True)
            
            # Calculate fraud probability
            fraud_prob = 1 / (1 + np.exp(-(expected_value + np.sum(shap_vals))))
            
            explanation = {
                'fraud_probability': float(fraud_prob),
                'base_value': float(expected_value),
                'top_contributing_factors': feature_impacts[:10],  # Top 10 factors
                'all_factors': feature_impacts,
                'model_threshold': self.threshold
            }
            
            explanations.append(explanation)
        
        # Return single explanation if only one sample
        if len(explanations) == 1:
            return explanations[0]
        
        return {'explanations': explanations}
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, 
                       n_folds: int = 5, 
                       params: Optional[Dict] = None) -> Dict[str, List[float]]:
        """
        Perform stratified k-fold cross-validation.
        
        Important for fraud detection to ensure robust evaluation across
        different time periods and customer segments.
        
        Args:
            X: Features
            y: Labels
            n_folds: Number of folds
            params: Model parameters (uses self.params if None)
        
        Returns:
            Dictionary of cross-validation scores
        """
        if params is None:
            params = self.params
        
        # Use stratified k-fold to maintain class distribution
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        cv_scores = {
            'roc_auc': [],
            'pr_auc': [],
            'f1': [],
            'precision': [],
            'recall': []
        }
        
        fold = 1
        for train_idx, val_idx in skf.split(X, y):
            logger.info(f"Training fold {fold}/{n_folds}")
            
            # Split data
            X_train_fold = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
            y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            X_val_fold = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
            y_val_fold = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]
            
            # Train model
            fold_model = FraudXGBoostModel(params=params.copy(), random_state=self.random_state)
            
            # Calculate scale_pos_weight for this fold
            neg_count = np.sum(y_train_fold == 0)
            pos_count = np.sum(y_train_fold == 1)
            if pos_count > 0:
                fold_model.params['scale_pos_weight'] = neg_count / pos_count
            
            fold_model.train(
                X_train_fold, y_train_fold,
                X_val_fold, y_val_fold,
                use_early_stopping=True,
                verbose_eval=False
            )
            
            # Find optimal threshold
            opt_threshold = fold_model.find_optimal_threshold(X_val_fold, y_val_fold, metric='f1')
            
            # Evaluate
            metrics = fold_model.evaluate(X_val_fold, y_val_fold, threshold=opt_threshold)
            
            # Store scores
            cv_scores['roc_auc'].append(metrics['roc_auc'])
            cv_scores['pr_auc'].append(metrics['pr_auc'])
            cv_scores['f1'].append(metrics['f1_score'])
            cv_scores['precision'].append(metrics['precision'])
            cv_scores['recall'].append(metrics['recall'])
            
            fold += 1
        
        # Calculate summary statistics
        cv_summary = {}
        for metric, scores in cv_scores.items():
            cv_summary[f'{metric}_mean'] = np.mean(scores)
            cv_summary[f'{metric}_std'] = np.std(scores)
            cv_summary[f'{metric}_min'] = np.min(scores)
            cv_summary[f'{metric}_max'] = np.max(scores)
        
        logger.info("=" * 50)
        logger.info("Cross-Validation Results")
        logger.info("=" * 50)
        for metric, value in cv_summary.items():
            logger.info(f"{metric:20s}: {value:.4f}")
        
        return cv_summary
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series,
                              param_grid: Optional[Dict] = None,
                              n_folds: int = 3,
                              scoring: str = 'average_precision') -> Dict:
        """
        Perform hyperparameter tuning using grid search.
        
        Args:
            X: Features
            y: Labels
            param_grid: Grid of parameters to search
            n_folds: Number of cross-validation folds
            scoring: Scoring metric for optimization
        
        Returns:
            Best parameters and scores
        """
        if param_grid is None:
            # Default parameter grid for fraud detection
            param_grid = {
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'min_child_weight': [1, 5, 10],
                'reg_alpha': [0.1, 1.0, 10.0],
                'reg_lambda': [0.1, 1.0, 10.0]
            }
        
        # Create XGBoost classifier
        xgb_clf = xgb.XGBClassifier(
            objective='binary:logistic',
            scale_pos_weight=self.params.get('scale_pos_weight', 1),
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='aucpr'
        )
        
        # Create stratified k-fold
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        # Grid search
        grid_search = GridSearchCV(
            estimator=xgb_clf,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        
        logger.info(f"Starting hyperparameter tuning with {len(param_grid)} parameters")
        grid_search.fit(X, y)
        
        # Update model with best parameters
        self.params.update(grid_search.best_params_)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best {scoring} score: {grid_search.best_score_:.4f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def save_model(self, filepath: str, include_explainer: bool = True) -> None:
        """
        Save trained model to disk with all metadata.
        
        Args:
            filepath: Path to save the model
            include_explainer: Whether to save SHAP explainer
        """
        if self.model is None:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'params': self.params,
            'feature_names': self.feature_names,
            'best_iteration': self.best_iteration,
            'threshold': self.threshold,
            'feature_importance': self.feature_importance,
            'random_state': self.random_state,
            'saved_at': datetime.now().isoformat()
        }
        
        if include_explainer and self.explainer is not None:
            model_data['explainer'] = self.explainer
        
        # Save using joblib (better for large numpy arrays)
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> 'FraudXGBoostModel':
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to saved model
        
        Returns:
            Loaded model instance
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.params = model_data['params']
        self.feature_names = model_data['feature_names']
        self.best_iteration = model_data['best_iteration']
        self.threshold = model_data['threshold']
        self.feature_importance = model_data['feature_importance']
        self.random_state = model_data['random_state']
        
        if 'explainer' in model_data:
            self.explainer = model_data['explainer']
        
        logger.info(f"Model loaded from {filepath}")
        return self
    
    def get_feature_importance_plot(self, importance_type: str = 'gain',
                                     top_n: int = 20) -> Dict:
        """
        Generate feature importance data for visualization.
        
        Args:
            importance_type: Type of importance ('weight', 'gain', 'cover')
            top_n: Number of top features to return
        
        Returns:
            Dictionary with feature importance data for plotting
        """
        if self.feature_importance is None or importance_type not in self.feature_importance:
            self._calculate_feature_importance()
        
        importance_df = self.feature_importance.get(importance_type, pd.DataFrame())
        
        if importance_df.empty:
            return {'error': f'No {importance_type} importance available'}
        
        # Get top N features
        top_features = importance_df.head(top_n)
        
        return {
            'features': top_features['feature'].tolist(),
            'importance_scores': top_features['importance'].tolist(),
            'importance_type': importance_type,
            'total_features': len(importance_df)
        }


# Utility function for quick model training
def train_fraud_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series,
                        **kwargs) -> FraudXGBoostModel:
    """
    Convenience function to quickly train a fraud detection XGBoost model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        **kwargs: Additional arguments for FraudXGBoostModel
    
    Returns:
        Trained model
    """
    model = FraudXGBoostModel(**kwargs)
    model.train(X_train, y_train, X_val, y_val, use_early_stopping=True)
    model.find_optimal_threshold(X_val, y_val, metric='f1')
    return model


# Example usage and testing
if __name__ == "__main__":
    # This section demonstrates how to use the model
    print("FraudXGBoostModel - Example Usage")
    print("=" * 50)
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 10000
    n_features = 20
    
    # Create imbalanced dataset (99% non-fraud, 1% fraud)
    X_demo = np.random.randn(n_samples, n_features)
    y_demo = np.zeros(n_samples)
    
    # Make some samples fraudulent
    fraud_indices = np.random.choice(n_samples, size=int(n_samples * 0.01), replace=False)
    y_demo[fraud_indices] = 1
    
    # Create DataFrames
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_demo_df = pd.DataFrame(X_demo, columns=feature_names)
    y_demo_series = pd.Series(y_demo, name='is_fraud')
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_demo_df, y_demo_series, test_size=0.3, random_state=42, stratify=y_demo_series
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Fraud ratio in training: {y_train.mean():.4f}")
    
    # Train model
    model = FraudXGBoostModel()
    model.train(X_train, y_train, X_val, y_val)
    
    # Find optimal threshold
    threshold = model.find_optimal_threshold(X_val, y_val)
    
    # Evaluate on test set
    metrics = model.evaluate(X_test, y_test)
    
    # Explain a prediction
    sample = X_test.iloc[:1]
    explanation = model.explain_prediction(sample)
    
    print("\nSample Explanation:")
    print(json.dumps(explanation['top_contributing_factors'][:3], indent=2))