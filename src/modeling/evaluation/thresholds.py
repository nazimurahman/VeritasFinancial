# src/modeling/evaluation/thresholds.py
"""
Advanced Threshold Optimization Module for Fraud Detection

This module provides sophisticated threshold optimization strategies
specifically designed for imbalanced fraud detection, including:
1. Business-aware threshold optimization
2. Multi-objective optimization
3. Dynamic threshold adjustment
4. Cost-sensitive threshold selection

Author: VeritasFinancial DS Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score
)
from scipy.optimize import minimize_scalar, differential_evolution
from dataclasses import dataclass, field
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ThresholdResult:
    """
    Data class for threshold optimization results.
    
    Attributes:
        threshold: Optimal threshold value
        metrics: Dictionary of metrics at this threshold
        objective_value: Value of objective function
        method: Method used for optimization
        metadata: Additional metadata
    """
    threshold: float
    metrics: Dict[str, float] = field(default_factory=dict)
    objective_value: float = 0.0
    method: str = ""
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'threshold': self.threshold,
            'metrics': self.metrics,
            'objective_value': self.objective_value,
            'method': self.method,
            'metadata': self.metadata
        }


class ThresholdOptimizer:
    """
    Advanced threshold optimizer for fraud detection.
    
    This class provides multiple methods for finding optimal
    classification thresholds based on various objectives.
    """
    
    def __init__(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        pos_label: int = 1
    ):
        """
        Initialize threshold optimizer.
        
        Parameters:
            y_true: True labels
            y_score: Prediction scores/probabilities
            pos_label: Label of positive class (fraud)
        """
        self.y_true = np.asarray(y_true)
        self.y_score = np.asarray(y_score)
        self.pos_label = pos_label
        
        # Validate inputs
        self._validate_inputs()
        
        # Pre-calculate precision-recall curve
        self.precisions, self.recalls, self.pr_thresholds = precision_recall_curve(
            self.y_true, self.y_score, pos_label=pos_label
        )
        
        # Pre-calculate ROC curve
        self.fpr, self.tpr, self.roc_thresholds = roc_curve(
            self.y_true, self.y_score, pos_label=pos_label
        )
        
        # Calculate overall statistics
        self.n_samples = len(y_true)
        self.n_pos = (y_true == pos_label).sum()
        self.n_neg = (y_true != pos_label).sum()
        self.base_rate = self.n_pos / self.n_samples
    
    def _validate_inputs(self):
        """Validate input arrays."""
        if len(self.y_true) != len(self.y_score):
            raise ValueError(
                f"Length mismatch: y_true ({len(self.y_true)}) != "
                f"y_score ({len(self.y_score)})"
            )
        
        if not np.all((self.y_score >= 0) & (self.y_score <= 1)):
            warnings.warn("y_score contains values outside [0, 1]. This may affect results.")
    
    def _get_metrics_at_threshold(self, threshold: float) -> Dict[str, float]:
        """
        Calculate all metrics at a given threshold.
        
        Parameters:
            threshold: Classification threshold
            
        Returns:
            Dictionary of metrics
        """
        y_pred = (self.y_score >= threshold).astype(int)
        
        # Calculate confusion matrix components
        tp = np.sum((self.y_true == 1) & (y_pred == 1))
        fp = np.sum((self.y_true == 0) & (y_pred == 1))
        fn = np.sum((self.y_true == 1) & (y_pred == 0))
        tn = np.sum((self.y_true == 0) & (y_pred == 0))
        
        # Calculate metrics
        metrics = {
            'threshold': threshold,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'tpr': tp / (tp + fn) if (tp + fn) > 0 else 0,  # Recall
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'accuracy': (tp + tn) / self.n_samples,
            'detection_rate': tp / self.n_samples,
            'false_alarm_rate': fp / self.n_samples,
            'miss_rate': fn / self.n_samples
        }
        
        return metrics
    
    # ==================== Single Objective Optimization ====================
    
    def optimize_f1(self) -> ThresholdResult:
        """
        Find threshold that maximizes F1 score.
        
        Returns:
            ThresholdResult with optimal threshold
        """
        # Calculate F1 for all thresholds
        f1_scores = []
        valid_thresholds = []
        
        for i, threshold in enumerate(self.pr_thresholds):
            metrics = self._get_metrics_at_threshold(threshold)
            f1_scores.append(metrics['f1'])
            valid_thresholds.append(threshold)
        
        f1_scores = np.array(f1_scores)
        
        # Find best threshold
        best_idx = np.argmax(f1_scores)
        best_threshold = valid_thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        
        # Get all metrics at best threshold
        metrics = self._get_metrics_at_threshold(best_threshold)
        
        return ThresholdResult(
            threshold=best_threshold,
            metrics=metrics,
            objective_value=best_f1,
            method='max_f1',
            metadata={'f1_scores': f1_scores.tolist()}
        )
    
    def optimize_precision_at_recall(
        self,
        target_recall: float,
        tolerance: float = 0.01
    ) -> ThresholdResult:
        """
        Find threshold that maximizes precision given minimum recall.
        
        Parameters:
            target_recall: Minimum recall required
            tolerance: Allowed deviation from target recall
            
        Returns:
            ThresholdResult with optimal threshold
        """
        # Find thresholds that achieve target recall
        valid_indices = []
        valid_precisions = []
        valid_thresholds = []
        
        for i, threshold in enumerate(self.pr_thresholds[:-1]):  # Exclude last point
            recall = self.recalls[i]
            
            if abs(recall - target_recall) <= tolerance or recall >= target_recall:
                valid_indices.append(i)
                valid_precisions.append(self.precisions[i])
                valid_thresholds.append(threshold)
        
        if not valid_indices:
            # If no threshold meets criteria, use threshold with highest recall
            best_idx = np.argmax(self.recalls[:-1])
            best_threshold = self.pr_thresholds[best_idx]
            best_precision = self.precisions[best_idx]
        else:
            # Choose threshold with highest precision among valid ones
            best_local_idx = np.argmax(valid_precisions)
            best_threshold = valid_thresholds[best_local_idx]
            best_precision = valid_precisions[best_local_idx]
        
        # Get all metrics at best threshold
        metrics = self._get_metrics_at_threshold(best_threshold)
        
        return ThresholdResult(
            threshold=best_threshold,
            metrics=metrics,
            objective_value=best_precision,
            method='max_precision_at_recall',
            metadata={'target_recall': target_recall}
        )
    
    def optimize_recall_at_precision(
        self,
        target_precision: float,
        tolerance: float = 0.01
    ) -> ThresholdResult:
        """
        Find threshold that maximizes recall given minimum precision.
        
        Parameters:
            target_precision: Minimum precision required
            tolerance: Allowed deviation from target precision
            
        Returns:
            ThresholdResult with optimal threshold
        """
        # Find thresholds that achieve target precision
        valid_indices = []
        valid_recalls = []
        valid_thresholds = []
        
        for i, threshold in enumerate(self.pr_thresholds[:-1]):  # Exclude last point
            precision = self.precisions[i]
            
            if abs(precision - target_precision) <= tolerance or precision >= target_precision:
                valid_indices.append(i)
                valid_recalls.append(self.recalls[i])
                valid_thresholds.append(threshold)
        
        if not valid_indices:
            # If no threshold meets criteria, use threshold with highest precision
            best_idx = np.argmax(self.precisions[:-1])
            best_threshold = self.pr_thresholds[best_idx]
            best_recall = self.recalls[best_idx]
        else:
            # Choose threshold with highest recall among valid ones
            best_local_idx = np.argmax(valid_recalls)
            best_threshold = valid_thresholds[best_local_idx]
            best_recall = valid_recalls[best_local_idx]
        
        # Get all metrics at best threshold
        metrics = self._get_metrics_at_threshold(best_threshold)
        
        return ThresholdResult(
            threshold=best_threshold,
            metrics=metrics,
            objective_value=best_recall,
            method='max_recall_at_precision',
            metadata={'target_precision': target_precision}
        )
    
    def optimize_youden_index(self) -> ThresholdResult:
        """
        Find threshold that maximizes Youden's Index.
        
        Youden's Index = Sensitivity + Specificity - 1
        
        Returns:
            ThresholdResult with optimal threshold
        """
        youden_scores = self.tpr + (1 - self.fpr) - 1
        
        best_idx = np.argmax(youden_scores)
        best_threshold = self.roc_thresholds[best_idx]
        best_youden = youden_scores[best_idx]
        
        # Get all metrics at best threshold
        metrics = self._get_metrics_at_threshold(best_threshold)
        
        return ThresholdResult(
            threshold=best_threshold,
            metrics=metrics,
            objective_value=best_youden,
            method='youden_index'
        )
    
    def optimize_cost(
        self,
        cost_matrix: Optional[np.ndarray] = None,
        fraud_amounts: Optional[np.ndarray] = None,
        investigation_cost: float = 10.0
    ) -> ThresholdResult:
        """
        Find threshold that minimizes total cost.
        
        Parameters:
            cost_matrix: 2x2 cost matrix [TN, FP; FN, TP]
            fraud_amounts: Actual fraud amounts per transaction
            investigation_cost: Cost per investigation
            
        Returns:
            ThresholdResult with optimal threshold
        """
        if cost_matrix is None and fraud_amounts is None:
            # Default cost matrix
            cost_matrix = np.array([
                [0, 10],    # TN: 0, FP: 10
                [100, 0]    # FN: 100, TP: 0 (saved fraud)
            ])
        
        costs = []
        thresholds = np.linspace(0, 1, 1000)
        
        for threshold in thresholds:
            y_pred = (self.y_score >= threshold).astype(int)
            
            if fraud_amounts is not None:
                # Calculate cost using actual amounts
                tp_amount = fraud_amounts[(self.y_true == 1) & (y_pred == 1)].sum()
                fn_amount = fraud_amounts[(self.y_true == 1) & (y_pred == 0)].sum()
                fp_count = ((self.y_true == 0) & (y_pred == 1)).sum()
                
                # Profit = detected fraud - missed fraud - investigation cost
                profit = tp_amount - fn_amount - (fp_count * investigation_cost)
                cost = -profit  # Minimize negative profit = maximize profit
            else:
                # Use cost matrix
                tn = ((self.y_true == 0) & (y_pred == 0)).sum()
                fp = ((self.y_true == 0) & (y_pred == 1)).sum()
                fn = ((self.y_true == 1) & (y_pred == 0)).sum()
                tp = ((self.y_true == 1) & (y_pred == 1)).sum()
                
                total_cost = (
                    tn * cost_matrix[0, 0] +
                    fp * cost_matrix[0, 1] +
                    fn * cost_matrix[1, 0] +
                    tp * cost_matrix[1, 1]
                )
                cost = total_cost
            
            costs.append(cost)
        
        costs = np.array(costs)
        best_idx = np.argmin(costs)
        best_threshold = thresholds[best_idx]
        best_cost = costs[best_idx]
        
        # Get all metrics at best threshold
        metrics = self._get_metrics_at_threshold(best_threshold)
        
        return ThresholdResult(
            threshold=best_threshold,
            metrics=metrics,
            objective_value=best_cost,
            method='min_cost',
            metadata={'cost_matrix': cost_matrix.tolist() if cost_matrix is not None else None}
        )
    
    def optimize_fbeta(
        self,
        beta: float = 2.0
    ) -> ThresholdResult:
        """
        Find threshold that maximizes F-beta score.
        
        Parameters:
            beta: Weight of recall in harmonic mean
                  beta > 1 favors recall, beta < 1 favors precision
        
        Returns:
            ThresholdResult with optimal threshold
        """
        fbeta_scores = []
        valid_thresholds = []
        
        for threshold in np.linspace(0, 1, 1000):
            metrics = self._get_metrics_at_threshold(threshold)
            
            precision = metrics['precision']
            recall = metrics['recall']
            
            if precision + recall > 0:
                fbeta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
            else:
                fbeta = 0
            
            fbeta_scores.append(fbeta)
            valid_thresholds.append(threshold)
        
        fbeta_scores = np.array(fbeta_scores)
        best_idx = np.argmax(fbeta_scores)
        best_threshold = valid_thresholds[best_idx]
        best_fbeta = fbeta_scores[best_idx]
        
        # Get all metrics at best threshold
        metrics = self._get_metrics_at_threshold(best_threshold)
        
        return ThresholdResult(
            threshold=best_threshold,
            metrics=metrics,
            objective_value=best_fbeta,
            method=f'f{beta}_score',
            metadata={'beta': beta}
        )
    
    # ==================== Multi-Objective Optimization ====================
    
    def optimize_multi_objective(
        self,
        objectives: List[Dict],
        weights: Optional[List[float]] = None,
        method: str = 'weighted_sum'
    ) -> ThresholdResult:
        """
        Multi-objective threshold optimization.
        
        Parameters:
            objectives: List of objective specifications, each with:
                       - 'metric': Metric to optimize
                       - 'weight': Weight (if weights not provided separately)
                       - 'direction': 'max' or 'min'
            weights: Optional list of weights for each objective
            method: 'weighted_sum', 'pareto', or 'goal_programming'
            
        Returns:
            ThresholdResult with optimal threshold
        """
        if weights is None:
            weights = [obj.get('weight', 1.0) for obj in objectives]
        
        weights = np.array(weights) / np.sum(weights)  # Normalize
        
        thresholds = np.linspace(0, 1, 1000)
        combined_scores = []
        
        for threshold in thresholds:
            metrics = self._get_metrics_at_threshold(threshold)
            
            if method == 'weighted_sum':
                score = 0
                for obj, weight in zip(objectives, weights):
                    metric_value = metrics[obj['metric']]
                    
                    # Normalize if needed
                    if 'normalize' in obj and obj['normalize']:
                        # Normalize to [0,1] range
                        if obj['direction'] == 'max':
                            metric_value = metric_value / obj.get('max_value', 1.0)
                        else:
                            metric_value = 1 - metric_value / obj.get('max_value', 1.0)
                    
                    # Apply direction
                    if obj['direction'] == 'max':
                        score += weight * metric_value
                    else:
                        score += weight * (1 - metric_value)
                
                combined_scores.append(score)
            
            elif method == 'goal_programming':
                # Goal programming - minimize deviation from targets
                deviation = 0
                for obj, weight in zip(objectives, weights):
                    target = obj.get('target', 0.5)
                    metric_value = metrics[obj['metric']]
                    
                    if obj['direction'] == 'max':
                        deviation += weight * max(0, target - metric_value) ** 2
                    else:
                        deviation += weight * max(0, metric_value - target) ** 2
                
                combined_scores.append(-deviation)  # Negative because we maximize
        
        combined_scores = np.array(combined_scores)
        best_idx = np.argmax(combined_scores)
        best_threshold = thresholds[best_idx]
        best_score = combined_scores[best_idx]
        
        # Get all metrics at best threshold
        metrics = self._get_metrics_at_threshold(best_threshold)
        
        return ThresholdResult(
            threshold=best_threshold,
            metrics=metrics,
            objective_value=best_score,
            method=f'multi_{method}',
            metadata={'objectives': objectives, 'weights': weights.tolist()}
        )
    
    def find_pareto_frontier(
        self,
        objectives: List[str],
        n_points: int = 100
    ) -> List[Dict]:
        """
        Find Pareto-optimal thresholds for multiple objectives.
        
        Parameters:
            objectives: List of objective metric names
            n_points: Number of threshold points to evaluate
            
        Returns:
            List of Pareto-optimal points with thresholds and metrics
        """
        thresholds = np.linspace(0, 1, n_points)
        points = []
        
        for threshold in thresholds:
            metrics = self._get_metrics_at_threshold(threshold)
            point = {'threshold': threshold}
            point.update({obj: metrics[obj] for obj in objectives})
            points.append(point)
        
        # Find Pareto frontier
        pareto_points = []
        
        for i, point in enumerate(points):
            is_pareto = True
            for j, other in enumerate(points):
                if i == j:
                    continue
                
                # Check if other dominates point
                dominates = True
                for obj in objectives:
                    if other[obj] <= point[obj]:
                        dominates = False
                        break
                
                if dominates:
                    is_pareto = False
                    break
            
            if is_pareto:
                pareto_points.append(point)
        
        return pareto_points
    
    # ==================== Dynamic Threshold Optimization ====================
    
    def optimize_dynamic_threshold(
        self,
        context_features: pd.DataFrame,
        context_mapping: Dict[str, Callable],
        base_threshold: float = 0.5,
        adjustment_range: Tuple[float, float] = (0.3, 0.8)
    ) -> pd.DataFrame:
        """
        Optimize thresholds dynamically based on context.
        
        This is useful for adjusting thresholds based on:
        - Time of day
        - Transaction amount
        - Customer risk profile
        - Merchant type
        
        Parameters:
            context_features: DataFrame with context features
            context_mapping: Dictionary mapping feature names to adjustment functions
            base_threshold: Base threshold value
            adjustment_range: Min and max allowed threshold
            
        Returns:
            DataFrame with dynamic thresholds for each sample
        """
        dynamic_thresholds = np.full(len(self.y_true), base_threshold)
        
        for feature_name, adjustment_func in context_mapping.items():
            if feature_name in context_features.columns:
                # Calculate adjustment for each sample
                adjustments = adjustment_func(context_features[feature_name].values)
                
                # Apply adjustments
                dynamic_thresholds += adjustments
        
        # Clip to allowed range
        dynamic_thresholds = np.clip(
            dynamic_thresholds,
            adjustment_range[0],
            adjustment_range[1]
        )
        
        # Calculate predictions using dynamic thresholds
        y_pred_dynamic = (self.y_score >= dynamic_thresholds).astype(int)
        
        # Calculate overall metrics
        metrics_dynamic = {}
        for metric in ['precision', 'recall', 'f1']:
            if metric == 'precision':
                metrics_dynamic[metric] = precision_score(self.y_true, y_pred_dynamic)
            elif metric == 'recall':
                metrics_dynamic[metric] = recall_score(self.y_true, y_pred_dynamic)
            else:
                metrics_dynamic[metric] = f1_score(self.y_true, y_pred_dynamic)
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'y_true': self.y_true,
            'y_score': self.y_score,
            'dynamic_threshold': dynamic_thresholds,
            'y_pred_dynamic': y_pred_dynamic
        })
        
        # Add context features
        for col in context_features.columns:
            result_df[col] = context_features[col].values
        
        return result_df
    
    def optimize_time_varying_threshold(
        self,
        timestamps: np.ndarray,
        time_windows: List[Tuple],
        base_threshold: float = 0.5
    ) -> Dict:
        """
        Optimize thresholds that vary over time.
        
        Parameters:
            timestamps: Array of timestamps for each sample
            time_windows: List of (start, end) time windows
            base_threshold: Base threshold value
            
        Returns:
            Dictionary with optimal thresholds for each time window
        """
        from datetime import datetime
        
        window_thresholds = {}
        window_metrics = {}
        
        for window_start, window_end in time_windows:
            # Find samples in this time window
            if isinstance(window_start, str):
                # Convert string to datetime if needed
                window_start = pd.to_datetime(window_start)
                window_end = pd.to_datetime(window_end)
            
            if isinstance(timestamps[0], (str, np.str_)):
                timestamps_dt = pd.to_datetime(timestamps)
            else:
                timestamps_dt = timestamps
            
            window_mask = (timestamps_dt >= window_start) & (timestamps_dt < window_end)
            
            if window_mask.sum() == 0:
                continue
            
            # Create optimizer for this window
            window_optimizer = ThresholdOptimizer(
                self.y_true[window_mask],
                self.y_score[window_mask]
            )
            
            # Find optimal threshold
            result = window_optimizer.optimize_f1()
            
            window_thresholds[f"{window_start}_{window_end}"] = result.threshold
            window_metrics[f"{window_start}_{window_end}"] = result.metrics
        
        return {
            'thresholds': window_thresholds,
            'metrics': window_metrics,
            'base_threshold': base_threshold
        }
    
    # ==================== Visualization and Analysis ====================
    
    def get_threshold_curve(self, n_points: int = 100) -> pd.DataFrame:
        """
        Generate curve data for threshold analysis.
        
        Parameters:
            n_points: Number of threshold points
            
        Returns:
            DataFrame with metrics at different thresholds
        """
        thresholds = np.linspace(0, 1, n_points)
        curve_data = []
        
        for threshold in thresholds:
            metrics = self._get_metrics_at_threshold(threshold)
            curve_data.append(metrics)
        
        return pd.DataFrame(curve_data)
    
    def get_stability_analysis(
        self,
        n_bootstrap: int = 100,
        confidence_level: float = 0.95
    ) -> Dict:
        """
        Analyze threshold stability using bootstrap.
        
        Parameters:
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with stability metrics
        """
        np.random.seed(42)
        
        bootstrap_thresholds = []
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(
                self.n_samples,
                self.n_samples,
                replace=True
            )
            
            # Create optimizer for bootstrap sample
            boot_optimizer = ThresholdOptimizer(
                self.y_true[indices],
                self.y_score[indices]
            )
            
            # Find optimal threshold
            result = boot_optimizer.optimize_f1()
            bootstrap_thresholds.append(result.threshold)
        
        bootstrap_thresholds = np.array(bootstrap_thresholds)
        
        # Calculate statistics
        mean_threshold = np.mean(bootstrap_thresholds)
        std_threshold = np.std(bootstrap_thresholds)
        
        # Confidence interval
        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_thresholds, 100 * alpha / 2)
        upper = np.percentile(bootstrap_thresholds, 100 * (1 - alpha / 2))
        
        # Coefficient of variation
        cv = std_threshold / mean_threshold if mean_threshold > 0 else 0
        
        return {
            'mean_threshold': mean_threshold,
            'std_threshold': std_threshold,
            'cv': cv,
            'confidence_interval': (lower, upper),
            'bootstrap_thresholds': bootstrap_thresholds.tolist()
        }
    
    def recommend_threshold(
        self,
        business_constraints: Dict
    ) -> List[ThresholdResult]:
        """
        Recommend thresholds based on business constraints.
        
        Parameters:
            business_constraints: Dictionary with:
                - 'min_precision': Minimum acceptable precision
                - 'min_recall': Minimum acceptable recall
                - 'max_false_positives': Maximum allowed false positives
                - 'min_detection_rate': Minimum detection rate
                - 'cost_matrix': Cost matrix for optimization
                
        Returns:
            List of recommended thresholds with their metrics
        """
        recommendations = []
        
        # Method 1: Max F1 subject to constraints
        thresholds = np.linspace(0, 1, 1000)
        valid_f1_scores = []
        valid_thresholds = []
        
        for threshold in thresholds:
            metrics = self._get_metrics_at_threshold(threshold)
            
            # Check constraints
            valid = True
            
            if 'min_precision' in business_constraints:
                if metrics['precision'] < business_constraints['min_precision']:
                    valid = False
            
            if 'min_recall' in business_constraints:
                if metrics['recall'] < business_constraints['min_recall']:
                    valid = False
            
            if 'max_false_positives' in business_constraints:
                if metrics['fp'] > business_constraints['max_false_positives']:
                    valid = False
            
            if 'min_detection_rate' in business_constraints:
                if metrics['detection_rate'] < business_constraints['min_detection_rate']:
                    valid = False
            
            if valid:
                valid_f1_scores.append(metrics['f1'])
                valid_thresholds.append(threshold)
        
        if valid_thresholds:
            best_idx = np.argmax(valid_f1_scores)
            best_threshold = valid_thresholds[best_idx]
            best_metrics = self._get_metrics_at_threshold(best_threshold)
            
            recommendations.append(ThresholdResult(
                threshold=best_threshold,
                metrics=best_metrics,
                objective_value=valid_f1_scores[best_idx],
                method='constrained_f1',
                metadata={'constraints': business_constraints}
            ))
        
        # Method 2: Cost optimization
        if 'cost_matrix' in business_constraints:
            cost_result = self.optimize_cost(
                cost_matrix=business_constraints['cost_matrix']
            )
            recommendations.append(cost_result)
        
        # Method 3: Balanced approach
        if 'precision_weight' in business_constraints and 'recall_weight' in business_constraints:
            beta = business_constraints['recall_weight'] / business_constraints['precision_weight']
            fbeta_result = self.optimize_fbeta(beta=beta)
            recommendations.append(fbeta_result)
        
        return recommendations


# Example usage
if __name__ == "__main__":
    """
    Example demonstrating threshold optimization.
    """
    
    print("=" * 60)
    print("Threshold Optimization Examples for Fraud Detection")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # True labels (5% fraud)
    y_true = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    
    # Probabilities (imperfect model)
    y_score = y_true + 0.3 * np.random.randn(n_samples)
    y_score = 1 / (1 + np.exp(-y_score))  # Sigmoid to [0,1]
    
    # Create optimizer
    optimizer = ThresholdOptimizer(y_true, y_score)
    
    print("\n1. Basic Statistics:")
    print(f"  Total samples: {n_samples}")
    print(f"  Fraud samples: {optimizer.n_pos} ({optimizer.base_rate:.2%})")
    
    # Optimize F1
    print("\n2. F1 Optimization:")
    result_f1 = optimizer.optimize_f1()
    print(f"  Optimal threshold: {result_f1.threshold:.4f}")
    print(f"  F1 Score: {result_f1.metrics['f1']:.4f}")
    print(f"  Precision: {result_f1.metrics['precision']:.4f}")
    print(f"  Recall: {result_f1.metrics['recall']:.4f}")
    
    # Optimize F2 (recall-focused)
    print("\n3. F2 Optimization (recall-focused):")
    result_f2 = optimizer.optimize_fbeta(beta=2.0)
    print(f"  Optimal threshold: {result_f2.threshold:.4f}")
    print(f"  F2 Score: {result_f2.objective_value:.4f}")
    print(f"  Precision: {result_f2.metrics['precision']:.4f}")
    print(f"  Recall: {result_f2.metrics['recall']:.4f}")
    
    # Optimize cost
    print("\n4. Cost Optimization:")
    cost_matrix = np.array([
        [0, 10],    # TN: 0, FP: $10 investigation
        [100, 0]    # FN: $100 fraud loss, TP: 0
    ])
    result_cost = optimizer.optimize_cost(cost_matrix=cost_matrix)
    print(f"  Optimal threshold: {result_cost.threshold:.4f}")
    print(f"  Total cost: ${result_cost.objective_value:.2f}")
    print(f"  False positives: {result_cost.metrics['fp']}")
    print(f"  False negatives: {result_cost.metrics['fn']}")
    
    # Optimize with recall constraint
    print("\n5. Max Precision at 80% Recall:")
    result_prec = optimizer.optimize_precision_at_recall(target_recall=0.8)
    print(f"  Optimal threshold: {result_prec.threshold:.4f}")
    print(f"  Achieved recall: {result_prec.metrics['recall']:.4f}")
    print(f"  Precision: {result_prec.metrics['precision']:.4f}")
    
    # Stability analysis
    print("\n6. Threshold Stability Analysis:")
    stability = optimizer.get_stability_analysis(n_bootstrap=100)
    print(f"  Mean threshold: {stability['mean_threshold']:.4f}")
    print(f"  Std threshold: {stability['std_threshold']:.4f}")
    print(f"  CV: {stability['cv']:.4f}")
    print(f"  95% CI: [{stability['confidence_interval'][0]:.4f}, "
          f"{stability['confidence_interval'][1]:.4f}]")
    
    # Get threshold curve
    print("\n7. Threshold Curve (first 5 points):")
    curve = optimizer.get_threshold_curve(n_points=10)
    print(curve[['threshold', 'precision', 'recall', 'f1']].head())
    
    # Business recommendations
    print("\n8. Business-Aware Recommendations:")
    business_constraints = {
        'min_precision': 0.7,
        'min_recall': 0.6,
        'cost_matrix': cost_matrix,
        'precision_weight': 0.4,
        'recall_weight': 0.6
    }
    
    recommendations = optimizer.recommend_threshold(business_constraints)
    for i, rec in enumerate(recommendations):
        print(f"  Recommendation {i+1}:")
        print(f"    Method: {rec.method}")
        print(f"    Threshold: {rec.threshold:.4f}")
        print(f"    Precision: {rec.metrics['precision']:.4f}")
        print(f"    Recall: {rec.metrics['recall']:.4f}")
        print(f"    F1: {rec.metrics['f1']:.4f}")
    
    print("\n" + "=" * 60)
    print("Threshold optimization module loaded successfully!")
    print("=" * 60)