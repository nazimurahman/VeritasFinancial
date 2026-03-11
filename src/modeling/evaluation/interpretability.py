# src/modeling/evaluation/interpretability.py
"""
Model Interpretability Module for Fraud Detection

This module provides comprehensive tools for explaining fraud predictions,
including:
1. Feature importance analysis
2. SHAP-based explanations
3. LIME explanations
4. Counterfactual explanations
5. Partial dependence plots
6. Decision tree surrogate models

Author: VeritasFinancial DS Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import warnings
import logging
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExplanationResult:
    """
    Data class for storing explanation results.
    
    Attributes:
        feature_name: Name of feature
        importance: Importance score
        shap_value: SHAP value (if applicable)
        contribution: Contribution to prediction
        direction: Positive or negative contribution
        description: Human-readable explanation
    """
    feature_name: str
    importance: float = 0.0
    shap_value: float = 0.0
    contribution: float = 0.0
    direction: str = ""
    description: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'feature_name': self.feature_name,
            'importance': self.importance,
            'shap_value': self.shap_value,
            'contribution': self.contribution,
            'direction': self.direction,
            'description': self.description
        }


class ModelInterpreter:
    """
    Comprehensive model interpreter for fraud detection.
    
    This class provides multiple methods for explaining model predictions,
    making it suitable for regulatory compliance and business understanding.
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        model_type: str = 'auto'
    ):
        """
        Initialize model interpreter.
        
        Parameters:
            model: Trained model object
            feature_names: List of feature names
            model_type: Type of model ('tree', 'linear', 'nn', 'auto')
        """
        self.model = model
        self.feature_names = feature_names
        self.model_type = self._detect_model_type() if model_type == 'auto' else model_type
        
        # Check for SHAP availability
        try:
            import shap
            self.shap_available = True
            self.shap = shap
        except ImportError:
            self.shap_available = False
            warnings.warn("SHAP not installed. Install with: pip install shap")
        
        # Check for LIME availability
        try:
            from lime import lime_tabular
            self.lime_available = True
            self.lime = lime_tabular
        except ImportError:
            self.lime_available = False
            warnings.warn("LIME not installed. Install with: pip install lime")
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        self.global_importance = None
    
    def _detect_model_type(self) -> str:
        """
        Automatically detect model type.
        
        Returns:
            Model type string
        """
        model_class = self.model.__class__.__name__.lower()
        
        if any(name in model_class for name in ['xgb', 'lgb', 'rf', 'tree', 'forest', 'gbm']):
            return 'tree'
        elif any(name in model_class for name in ['linear', 'logistic', 'ridge', 'lasso']):
            return 'linear'
        elif any(name in model_class for name in ['nn', 'mlp', 'dense', 'network']):
            return 'nn'
        else:
            return 'unknown'
    
    # ==================== Global Feature Importance ====================
    
    def get_feature_importance(
        self,
        method: str = 'auto',
        X: Optional[pd.DataFrame] = None,
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Get global feature importance.
        
        Parameters:
            method: 'auto', 'builtin', 'permutation', 'shap'
            X: Feature matrix (required for permutation importance)
            normalize: Whether to normalize importance scores
            
        Returns:
            DataFrame with feature importance
        """
        importance_dict = {}
        
        if method == 'auto':
            # Try built-in first, then permutation
            if hasattr(self.model, 'feature_importances_'):
                method = 'builtin'
            elif hasattr(self.model, 'coef_'):
                method = 'builtin'
            else:
                method = 'permutation'
        
        if method == 'builtin':
            # Built-in importance
            if hasattr(self.model, 'feature_importances_'):
                # Tree-based models
                importance = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                # Linear models
                if self.model.coef_.ndim == 1:
                    importance = np.abs(self.model.coef_)
                else:
                    importance = np.abs(self.model.coef_[0])
            else:
                raise ValueError("Model has no built-in feature importance")
            
            for name, imp in zip(self.feature_names, importance):
                importance_dict[name] = imp
        
        elif method == 'permutation':
            # Permutation importance
            if X is None:
                raise ValueError("X required for permutation importance")
            
            from sklearn.inspection import permutation_importance
            
            result = permutation_importance(
                self.model, X, np.zeros(len(X)),  # Dummy y, not used
                n_repeats=10,
                random_state=42,
                scoring=None
            )
            
            for name, imp, std in zip(
                self.feature_names,
                result.importances_mean,
                result.importances_std
            ):
                importance_dict[name] = {
                    'importance_mean': imp,
                    'importance_std': std
                }
        
        elif method == 'shap':
            # SHAP-based importance
            if not self.shap_available:
                raise ImportError("SHAP not available")
            
            if X is None:
                raise ValueError("X required for SHAP importance")
            
            shap_values = self.get_shap_values(X)
            importance = np.abs(shap_values).mean(axis=0)
            
            for name, imp in zip(self.feature_names, importance):
                importance_dict[name] = imp
        
        # Convert to DataFrame
        if method == 'permutation':
            df = pd.DataFrame([
                {
                    'feature': name,
                    'importance': values['importance_mean'],
                    'std': values['importance_std']
                }
                for name, values in importance_dict.items()
            ])
        else:
            df = pd.DataFrame([
                {'feature': name, 'importance': imp}
                for name, imp in importance_dict.items()
            ])
        
        # Sort by importance
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        # Normalize if requested
        if normalize and method != 'permutation':
            df['importance'] = df['importance'] / df['importance'].sum()
        
        self.global_importance = df
        return df
    
    def plot_feature_importance(
        self,
        top_n: int = 20,
        figsize: Tuple[int, int] = (10, 8),
        title: str = "Feature Importance"
    ) -> plt.Figure:
        """
        Plot feature importance.
        
        Parameters:
            top_n: Number of top features to show
            figsize: Figure size
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        if self.global_importance is None:
            raise ValueError("Run get_feature_importance first")
        
        df = self.global_importance.head(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if 'std' in df.columns:
            # Permutation importance with error bars
            ax.barh(
                range(len(df)),
                df['importance'].values[::-1],
                xerr=df['std'].values[::-1],
                capsize=3
            )
        else:
            # Regular importance
            ax.barh(range(len(df)), df['importance'].values[::-1])
        
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df['feature'].values[::-1])
        ax.set_xlabel('Importance')
        ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    # ==================== SHAP Explanations ====================
    
    def get_shap_values(
        self,
        X: pd.DataFrame,
        background_size: int = 100,
        **kwargs
    ) -> np.ndarray:
        """
        Calculate SHAP values for explanations.
        
        Parameters:
            X: Feature matrix
            background_size: Size of background dataset for KernelSHAP
            **kwargs: Additional arguments for SHAP explainer
            
        Returns:
            SHAP values array
        """
        if not self.shap_available:
            raise ImportError("SHAP not available")
        
        # Create explainer if not already created
        if self.shap_explainer is None:
            if self.model_type == 'tree':
                self.shap_explainer = self.shap.TreeExplainer(self.model)
            elif self.model_type == 'linear':
                self.shap_explainer = self.shap.LinearExplainer(self.model, X[:background_size])
            else:
                # Use KernelExplainer for other models
                background = X.sample(min(background_size, len(X)))
                self.shap_explainer = self.shap.KernelExplainer(
                    self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                    background
                )
        
        # Calculate SHAP values
        shap_values = self.shap_explainer.shap_values(X, **kwargs)
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            # Take values for positive class (fraud)
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        return shap_values
    
    def explain_with_shap(
        self,
        X: pd.DataFrame,
        sample_idx: int = 0,
        feature_names: Optional[List[str]] = None,
        plot_type: str = 'waterfall'
    ) -> Dict:
        """
        Explain a single prediction using SHAP.
        
        Parameters:
            X: Feature matrix
            sample_idx: Index of sample to explain
            feature_names: Custom feature names (optional)
            plot_type: 'waterfall', 'force', or 'bar'
            
        Returns:
            Dictionary with explanation
        """
        if not self.shap_available:
            raise ImportError("SHAP not available")
        
        if feature_names is None:
            feature_names = self.feature_names
        
        # Get SHAP values for this sample
        shap_values = self.get_shap_values(X)
        
        if shap_values.ndim == 2:
            sample_shap = shap_values[sample_idx:sample_idx+1]
        else:
            sample_shap = shap_values[sample_idx]
        
        # Get prediction
        if hasattr(self.model, 'predict_proba'):
            pred_proba = self.model.predict_proba(X.iloc[sample_idx:sample_idx+1])[0, 1]
        else:
            pred_proba = self.model.predict(X.iloc[sample_idx:sample_idx+1])[0]
        
        # Get base value
        if hasattr(self.shap_explainer, 'expected_value'):
            if isinstance(self.shap_explainer.expected_value, list):
                base_value = self.shap_explainer.expected_value[1]
            else:
                base_value = self.shap_explainer.expected_value
        else:
            base_value = 0
        
        # Create explanation
        explanations = []
        for i, (feature, shap_val) in enumerate(zip(feature_names, sample_shap.flatten())):
            direction = "positive" if shap_val > 0 else "negative"
            explanations.append(ExplanationResult(
                feature_name=feature,
                shap_value=shap_val,
                contribution=abs(shap_val),
                direction=direction,
                description=f"{feature} contributed {direction}ly to fraud probability"
            ))
        
        # Sort by absolute SHAP value
        explanations.sort(key=lambda x: abs(x.shap_value), reverse=True)
        
        # Create plot if requested
        if plot_type == 'waterfall':
            self._plot_shap_waterfall(sample_shap, base_value, feature_names, pred_proba)
        elif plot_type == 'force':
            self._plot_shap_force(sample_shap, base_value, feature_names)
        
        return {
            'sample_idx': sample_idx,
            'prediction_proba': pred_proba,
            'base_value': base_value,
            'explanations': [exp.to_dict() for exp in explanations],
            'shap_values': sample_shap.flatten().tolist()
        }
    
    def _plot_shap_waterfall(
        self,
        shap_values: np.ndarray,
        base_value: float,
        feature_names: List[str],
        prediction: float,
        max_display: int = 10
    ):
        """
        Create SHAP waterfall plot.
        """
        self.shap.plots.waterfall(
            self.shap.Explanation(
                values=shap_values[0],
                base_values=base_value,
                data=shap_values[0],  # Simplified
                feature_names=feature_names
            ),
            max_display=max_display,
            show=False
        )
        plt.title(f"SHAP Waterfall Plot (Prediction: {prediction:.4f})")
        plt.tight_layout()
        plt.show()
    
    def _plot_shap_force(
        self,
        shap_values: np.ndarray,
        base_value: float,
        feature_names: List[str]
    ):
        """
        Create SHAP force plot.
        """
        self.shap.force_plot(
            base_value,
            shap_values,
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        plt.tight_layout()
        plt.show()
    
    def get_shap_summary_plot(
        self,
        X: pd.DataFrame,
        max_display: int = 20,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Create SHAP summary plot.
        
        Parameters:
            X: Feature matrix
            max_display: Maximum number of features to show
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        if not self.shap_available:
            raise ImportError("SHAP not available")
        
        shap_values = self.get_shap_values(X)
        
        fig, ax = plt.subplots(figsize=figsize)
        self.shap.summary_plot(
            shap_values,
            X,
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        return fig
    
    # ==================== LIME Explanations ====================
    
    def init_lime_explainer(
        self,
        X_train: pd.DataFrame,
        mode: str = 'classification',
        **kwargs
    ):
        """
        Initialize LIME explainer.
        
        Parameters:
            X_train: Training feature matrix
            mode: 'classification' or 'regression'
            **kwargs: Additional arguments for LIME
        """
        if not self.lime_available:
            raise ImportError("LIME not available")
        
        self.lime_explainer = self.lime.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=self.feature_names,
            class_names=['legitimate', 'fraud'],
            mode=mode,
            **kwargs
        )
    
    def explain_with_lime(
        self,
        X: pd.DataFrame,
        sample_idx: int = 0,
        num_features: int = 10,
        num_samples: int = 5000
    ) -> Dict:
        """
        Explain a single prediction using LIME.
        
        Parameters:
            X: Feature matrix
            sample_idx: Index of sample to explain
            num_features: Number of features to include
            num_samples: Number of perturbation samples
            
        Returns:
            Dictionary with explanation
        """
        if not self.lime_available:
            raise ImportError("LIME not available")
        
        if self.lime_explainer is None:
            raise ValueError("Run init_lime_explainer first")
        
        # Get sample
        sample = X.iloc[sample_idx:sample_idx+1].values[0]
        
        # Define prediction function
        def predict_fn(x):
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(x)
            else:
                preds = self.model.predict(x)
                # Convert to probability-like format
                return np.column_stack([1-preds, preds])
        
        # Get explanation
        explanation = self.lime_explainer.explain_instance(
            sample,
            predict_fn,
            num_features=num_features,
            num_samples=num_samples
        )
        
        # Extract explanations
        explanations = []
        for feature, weight in explanation.as_list():
            explanations.append(ExplanationResult(
                feature_name=feature,
                importance=abs(weight),
                contribution=weight,
                direction="positive" if weight > 0 else "negative",
                description=f"{feature} contributed {weight:+.4f} to fraud probability"
            ))
        
        # Get prediction
        if hasattr(self.model, 'predict_proba'):
            pred_proba = self.model.predict_proba(sample.reshape(1, -1))[0, 1]
        else:
            pred_proba = self.model.predict(sample.reshape(1, -1))[0]
        
        # Create plot
        fig = explanation.as_pyplot_figure()
        plt.title(f"LIME Explanation (Prediction: {pred_proba:.4f})")
        plt.tight_layout()
        plt.show()
        
        return {
            'sample_idx': sample_idx,
            'prediction_proba': pred_proba,
            'explanations': [exp.to_dict() for exp in explanations],
            'lime_explanation': explanation
        }
    
    # ==================== Counterfactual Explanations ====================
    
    def generate_counterfactual(
        self,
        X: pd.DataFrame,
        sample_idx: int = 0,
        target_class: int = 0,
        features_to_vary: Optional[List[str]] = None,
        max_iter: int = 1000,
        step_size: float = 0.01
    ) -> Dict:
        """
        Generate counterfactual explanation.
        
        A counterfactual shows the smallest change needed to flip the prediction.
        
        Parameters:
            X: Feature matrix
            sample_idx: Index of sample
            target_class: Desired class (0 for legitimate, 1 for fraud)
            features_to_vary: List of features that can be changed
            max_iter: Maximum optimization iterations
            step_size: Step size for gradient descent
            
        Returns:
            Dictionary with counterfactual
        """
        original = X.iloc[sample_idx:sample_idx+1].values[0]
        
        if features_to_vary is None:
            feature_indices = list(range(len(self.feature_names)))
        else:
            feature_indices = [
                i for i, name in enumerate(self.feature_names)
                if name in features_to_vary
            ]
        
        # Define prediction function
        def predict_proba(x):
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(x.reshape(1, -1))[0, target_class]
            else:
                return self.model.predict(x.reshape(1, -1))[0]
        
        # Simple optimization to find counterfactual
        current = original.copy()
        best = original.copy()
        best_score = predict_proba(original)
        target_score = 0.5  # Decision boundary
        
        for iteration in range(max_iter):
            # Try small random changes
            delta = np.random.randn(len(original)) * step_size
            delta[~np.isin(np.arange(len(original)), feature_indices)] = 0
            
            candidate = current + delta
            
            # Clip to reasonable ranges (simplified)
            candidate = np.clip(candidate, -10, 10)
            
            score = predict_proba(candidate)
            
            # Check if we're closer to target
            if target_class == 0:
                # Want lower fraud probability
                if score < best_score:
                    best = candidate
                    best_score = score
                    current = candidate
                    
                    if score < target_score:
                        break
            else:
                # Want higher fraud probability
                if score > best_score:
                    best = candidate
                    best_score = score
                    current = candidate
                    
                    if score > target_score:
                        break
        
        # Calculate changes
        changes = []
        for i, (orig_val, new_val) in enumerate(zip(original, best)):
            if abs(new_val - orig_val) > 1e-6:
                changes.append({
                    'feature': self.feature_names[i],
                    'original_value': orig_val,
                    'new_value': new_val,
                    'change': new_val - orig_val,
                    'change_pct': (new_val - orig_val) / (abs(orig_val) + 1e-8)
                })
        
        return {
            'original_class': 1 if predict_proba(original) > 0.5 else 0,
            'original_probability': predict_proba(original),
            'counterfactual_class': 1 if best_score > 0.5 else 0,
            'counterfactual_probability': best_score,
            'target_class': target_class,
            'changes': changes,
            'n_iterations': iteration + 1
        }
    
    # ==================== Partial Dependence Plots ====================
    
    def partial_dependence(
        self,
        X: pd.DataFrame,
        feature: str,
        grid_resolution: int = 50,
        ice: bool = False
    ) -> Dict:
        """
        Calculate partial dependence for a feature.
        
        Parameters:
            X: Feature matrix
            feature: Feature name
            grid_resolution: Number of grid points
            ice: Whether to also calculate ICE curves
            
        Returns:
            Dictionary with partial dependence results
        """
        from sklearn.inspection import partial_dependence
        
        feature_idx = self.feature_names.index(feature)
        
        # Calculate partial dependence
        pdp_result = partial_dependence(
            self.model,
            X,
            features=[feature_idx],
            grid_resolution=grid_resolution,
            kind='average' if not ice else 'both'
        )
        
        result = {
            'feature': feature,
            'values': pdp_result['values'][0].tolist(),
            'average': pdp_result['average'][0].tolist()
        }
        
        if ice and 'individual' in pdp_result:
            result['individual'] = pdp_result['individual'][0].tolist()
        
        return result
    
    def plot_partial_dependence(
        self,
        X: pd.DataFrame,
        features: List[str],
        figsize: Tuple[int, int] = (15, 10)
    ) -> plt.Figure:
        """
        Plot partial dependence for multiple features.
        
        Parameters:
            X: Feature matrix
            features: List of feature names
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        from sklearn.inspection import PartialDependenceDisplay
        
        feature_indices = [self.feature_names.index(f) for f in features]
        
        fig, ax = plt.subplots(
            nrows=(len(features) + 1) // 2,
            ncols=2,
            figsize=figsize
        )
        ax = ax.flatten() if hasattr(ax, 'flatten') else [ax]
        
        display = PartialDependenceDisplay.from_estimator(
            self.model,
            X,
            features=feature_indices,
            feature_names=self.feature_names,
            ax=ax[:len(features)],
            grid_resolution=50,
            ice=False
        )
        
        plt.tight_layout()
        return fig
    
    # ==================== Decision Tree Surrogate ====================
    
    def create_surrogate_tree(
        self,
        X: pd.DataFrame,
        y_pred: Optional[np.ndarray] = None,
        max_depth: int = 3,
        min_samples_split: int = 100
    ) -> Any:
        """
        Create a decision tree surrogate to explain model behavior.
        
        Parameters:
            X: Feature matrix
            y_pred: Model predictions (if None, use model to predict)
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split
            
        Returns:
            Fitted decision tree
        """
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.tree import export_text, plot_tree
        
        # Get model predictions
        if y_pred is None:
            if hasattr(self.model, 'predict_proba'):
                y_pred = self.model.predict_proba(X)[:, 1]
            else:
                y_pred = self.model.predict(X)
        
        # Determine if classification or regression
        if len(np.unique(y_pred)) < 10 or np.all((y_pred >= 0) & (y_pred <= 1)):
            # Classification (binary or probability)
            if np.all((y_pred >= 0) & (y_pred <= 1)):
                # Probabilities - treat as regression
                tree = DecisionTreeRegressor(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42
                )
            else:
                # Binary classes
                tree = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42
                )
        else:
            # Regression
            tree = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            )
        
        # Fit tree
        tree.fit(X, y_pred)
        
        # Print tree rules
        tree_rules = export_text(tree, feature_names=self.feature_names)
        logger.info("Surrogate Decision Tree Rules:\n" + tree_rules)
        
        # Plot tree
        plt.figure(figsize=(20, 10))
        plot_tree(
            tree,
            feature_names=self.feature_names,
            filled=True,
            rounded=True,
            fontsize=10
        )
        plt.title("Surrogate Decision Tree")
        plt.show()
        
        return tree
    
    # ==================== Comprehensive Explanation ====================
    
    def explain_prediction(
        self,
        X: pd.DataFrame,
        sample_idx: int = 0,
        methods: List[str] = ['shap', 'lime'],
        **kwargs
    ) -> Dict:
        """
        Provide comprehensive explanation using multiple methods.
        
        Parameters:
            X: Feature matrix
            sample_idx: Index of sample
            methods: List of explanation methods to use
            **kwargs: Additional arguments for specific methods
            
        Returns:
            Dictionary with explanations from all methods
        """
        result = {
            'sample_idx': sample_idx,
            'feature_values': X.iloc[sample_idx].to_dict()
        }
        
        # Get prediction
        if hasattr(self.model, 'predict_proba'):
            result['prediction_proba'] = self.model.predict_proba(
                X.iloc[sample_idx:sample_idx+1]
            )[0, 1]
            result['prediction_class'] = 1 if result['prediction_proba'] > 0.5 else 0
        else:
            result['prediction_class'] = self.model.predict(
                X.iloc[sample_idx:sample_idx+1]
            )[0]
        
        # SHAP explanation
        if 'shap' in methods and self.shap_available:
            try:
                result['shap'] = self.explain_with_shap(
                    X, sample_idx, plot_type='waterfall' if 'plot' in kwargs else None
                )
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")
                result['shap'] = {'error': str(e)}
        
        # LIME explanation
        if 'lime' in methods and self.lime_available:
            try:
                if self.lime_explainer is None and 'X_train' in kwargs:
                    self.init_lime_explainer(kwargs['X_train'])
                
                result['lime'] = self.explain_with_lime(
                    X, sample_idx,
                    num_features=kwargs.get('num_features', 10)
                )
            except Exception as e:
                logger.warning(f"LIME explanation failed: {e}")
                result['lime'] = {'error': str(e)}
        
        # Counterfactual explanation
        if 'counterfactual' in methods:
            try:
                result['counterfactual'] = self.generate_counterfactual(
                    X, sample_idx,
                    target_class=kwargs.get('target_class', 0)
                )
            except Exception as e:
                logger.warning(f"Counterfactual explanation failed: {e}")
                result['counterfactual'] = {'error': str(e)}
        
        return result
    
    def generate_explanation_report(
        self,
        X: pd.DataFrame,
        sample_indices: Optional[List[int]] = None,
        n_samples: int = 5,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate explanation report for multiple samples.
        
        Parameters:
            X: Feature matrix
            sample_indices: Specific indices to explain
            n_samples: Number of random samples (if sample_indices not provided)
            **kwargs: Additional arguments for explain_prediction
            
        Returns:
            DataFrame with explanations
        """
        if sample_indices is None:
            # Randomly select samples, balancing classes if possible
            if hasattr(self.model, 'predict_proba'):
                y_pred = self.model.predict_proba(X)[:, 1] > 0.5
            else:
                y_pred = self.model.predict(X)
            
            fraud_indices = np.where(y_pred == 1)[0]
            legit_indices = np.where(y_pred == 0)[0]
            
            n_fraud = min(n_samples // 2, len(fraud_indices))
            n_legit = min(n_samples - n_fraud, len(legit_indices))
            
            selected_fraud = np.random.choice(fraud_indices, n_fraud, replace=False)
            selected_legit = np.random.choice(legit_indices, n_legit, replace=False)
            
            sample_indices = np.concatenate([selected_fraud, selected_legit])
        
        # Generate explanations
        explanations = []
        for idx in sample_indices:
            exp = self.explain_prediction(X, idx, **kwargs)
            explanations.append({
                'sample_idx': idx,
                'prediction_class': exp['prediction_class'],
                'prediction_proba': exp.get('prediction_proba', None),
                'top_shap_feature': exp.get('shap', {}).get('explanations', [{}])[0].get('feature_name', 'N/A'),
                'top_shap_value': exp.get('shap', {}).get('explanations', [{}])[0].get('shap_value', 0),
                'counterfactual_changes': len(exp.get('counterfactual', {}).get('changes', []))
            })
        
        return pd.DataFrame(explanations)


# Example usage
if __name__ == "__main__":
    """
    Example demonstrating model interpretability.
    """
    
    print("=" * 60)
    print("Model Interpretability Examples for Fraud Detection")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Create synthetic data
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=feature_names
    )
    
    # Create synthetic target (fraud)
    y = (X['feature_0'] + X['feature_1'] * 2 + np.random.randn(n_samples) * 0.5 > 0).astype(int)
    
    # Train a simple model
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X, y)
    
    # Create interpreter
    interpreter = ModelInterpreter(model, feature_names)
    
    print("\n1. Feature Importance:")
    importance_df = interpreter.get_feature_importance(method='builtin')
    print(importance_df.head())
    
    # Plot feature importance
    interpreter.plot_feature_importance(top_n=10)
    plt.show()
    
    print("\n2. SHAP Explanation for Sample 0:")
    # This would show SHAP plots if SHAP is installed
    if interpreter.shap_available:
        try:
            explanation = interpreter.explain_with_shap(X, sample_idx=0)
            print(f"  Prediction probability: {explanation['prediction_proba']:.4f}")
            print("  Top 3 features:")
            for exp in explanation['explanations'][:3]:
                print(f"    {exp['feature_name']}: {exp['shap_value']:.4f} ({exp['direction']})")
        except Exception as e:
            print(f"  SHAP explanation skipped: {e}")
    
    print("\n3. Partial Dependence for feature_0:")
    pdp = interpreter.partial_dependence(X, 'feature_0')
    print(f"  Values: {pdp['values'][:5]}...")
    print(f"  Average predictions: {pdp['average'][:5]}...")
    
    print("\n" + "=" * 60)
    print("Interpretability module loaded successfully!")
    print("=" * 60)