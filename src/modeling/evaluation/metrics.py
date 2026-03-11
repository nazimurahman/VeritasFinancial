# src/modeling/evaluation/metrics.py
"""
Comprehensive Metrics Module for Fraud Detection Evaluation

This module provides a wide range of metrics specifically designed
for evaluating fraud detection models, including:
1. Standard classification metrics with imbalanced data handling
2. Cost-sensitive metrics
3. Business impact metrics
4. Statistical significance tests
5. Multi-threshold metrics

Author: VeritasFinancial DS Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    auc,
    log_loss,
    brier_score_loss,
    matthews_corrcoef,
    cohen_kappa_score
)
from scipy import stats
from scipy.special import expit
import warnings
from dataclasses import dataclass, field
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """
    Data class for storing metric results with metadata.
    
    Attributes:
        name: Metric name
        value: Metric value
        confidence_interval: Tuple of (lower, upper) confidence bounds
        std_error: Standard error of the metric
        n_samples: Number of samples used
        description: Description of the metric
    """
    name: str
    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    std_error: Optional[float] = None
    n_samples: Optional[int] = None
    description: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'confidence_interval': self.confidence_interval,
            'std_error': self.std_error,
            'n_samples': self.n_samples,
            'description': self.description
        }


class FraudMetrics:
    """
    Comprehensive metrics calculator for fraud detection.
    
    This class provides methods to calculate various metrics
    and combine them into a complete evaluation report.
    """
    
    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
        pos_label: int = 1
    ):
        """
        Initialize metrics calculator.
        
        Parameters:
            y_true: True labels (0 for legitimate, 1 for fraud)
            y_pred: Predicted labels
            y_score: Prediction scores/probabilities (optional)
            sample_weight: Sample weights (optional)
            pos_label: Label of positive class (fraud)
        """
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.y_score = np.asarray(y_score) if y_score is not None else None
        self.sample_weight = sample_weight
        self.pos_label = pos_label
        
        # Validate inputs
        self._validate_inputs()
        
        # Calculate confusion matrix
        self.tn, self.fp, self.fn, self.tp = confusion_matrix(
            self.y_true, 
            self.y_pred,
            sample_weight=self.sample_weight
        ).ravel()
        
        # Total samples
        self.n_samples = len(y_true)
        self.n_pos = (y_true == pos_label).sum()
        self.n_neg = (y_true != pos_label).sum()
    
    def _validate_inputs(self):
        """Validate input arrays."""
        if len(self.y_true) != len(self.y_pred):
            raise ValueError(
                f"Length mismatch: y_true ({len(self.y_true)}) != "
                f"y_pred ({len(self.y_pred)})"
            )
        
        if self.y_score is not None and len(self.y_score) != len(self.y_true):
            raise ValueError(
                f"Length mismatch: y_true ({len(self.y_true)}) != "
                f"y_score ({len(self.y_score)})"
            )
    
    # ==================== Basic Metrics ====================
    
    def accuracy(self) -> float:
        """
        Calculate accuracy.
        
        Note: For imbalanced fraud detection, accuracy can be misleading
        as predicting all as legitimate gives high accuracy.
        """
        return (self.tp + self.tn) / self.n_samples
    
    def precision(self) -> float:
        """
        Calculate precision (Positive Predictive Value).
        
        Precision = TP / (TP + FP)
        Measures how many predicted frauds are actually fraud.
        """
        if self.tp + self.fp == 0:
            return 0.0
        return self.tp / (self.tp + self.fp)
    
    def recall(self) -> float:
        """
        Calculate recall (Sensitivity, True Positive Rate).
        
        Recall = TP / (TP + FN)
        Measures how many actual frauds are detected.
        """
        if self.tp + self.fn == 0:
            return 0.0
        return self.tp / (self.tp + self.fn)
    
    def specificity(self) -> float:
        """
        Calculate specificity (True Negative Rate).
        
        Specificity = TN / (TN + FP)
        Measures how many legitimate transactions are correctly classified.
        """
        if self.tn + self.fp == 0:
            return 0.0
        return self.tn / (self.tn + self.fp)
    
    def f1_score(self, beta: float = 1.0) -> float:
        """
        Calculate F-beta score.
        
        F1 = 2 * (precision * recall) / (precision + recall)
        Harmonic mean of precision and recall.
        
        Parameters:
            beta: Weight of precision in harmonic mean
        """
        if beta == 1.0:
            return f1_score(
                self.y_true, 
                self.y_pred,
                pos_label=self.pos_label,
                sample_weight=self.sample_weight
            )
        else:
            return fbeta_score(
                self.y_true,
                self.y_pred,
                beta=beta,
                pos_label=self.pos_label,
                sample_weight=self.sample_weight
            )
    
    def matthews_correlation(self) -> float:
        """
        Calculate Matthews Correlation Coefficient (MCC).
        
        MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
        
        A balanced measure even for imbalanced classes.
        Range: [-1, 1], where 1 is perfect prediction.
        """
        return matthews_corrcoef(
            self.y_true, 
            self.y_pred,
            sample_weight=self.sample_weight
        )
    
    def cohen_kappa(self) -> float:
        """
        Calculate Cohen's Kappa.
        
        Measures agreement between predictions and actual values,
        accounting for agreement by chance.
        """
        return cohen_kappa_score(
            self.y_true, 
            self.y_pred,
            weights=None,
            sample_weight=self.sample_weight
        )
    
    # ==================== Probability-Based Metrics ====================
    
    def roc_auc(self) -> Optional[float]:
        """
        Calculate Area Under ROC Curve.
        
        Requires probability scores.
        """
        if self.y_score is None:
            warnings.warn("y_score not provided. Cannot calculate ROC AUC.")
            return None
        
        try:
            return roc_auc_score(
                self.y_true,
                self.y_score,
                sample_weight=self.sample_weight
            )
        except Exception as e:
            warnings.warn(f"Error calculating ROC AUC: {e}")
            return None
    
    def pr_auc(self) -> Optional[float]:
        """
        Calculate Area Under Precision-Recall Curve.
        
        Better than ROC AUC for imbalanced fraud detection.
        Requires probability scores.
        """
        if self.y_score is None:
            warnings.warn("y_score not provided. Cannot calculate PR AUC.")
            return None
        
        try:
            return average_precision_score(
                self.y_true,
                self.y_score,
                sample_weight=self.sample_weight,
                pos_label=self.pos_label
            )
        except Exception as e:
            warnings.warn(f"Error calculating PR AUC: {e}")
            return None
    
    def log_loss_score(self) -> Optional[float]:
        """
        Calculate logarithmic loss.
        
        Measures the performance of a classifier where the prediction
        is a probability value between 0 and 1.
        """
        if self.y_score is None:
            warnings.warn("y_score not provided. Cannot calculate log loss.")
            return None
        
        try:
            return log_loss(
                self.y_true,
                self.y_score,
                sample_weight=self.sample_weight,
                labels=[0, 1]
            )
        except Exception as e:
            warnings.warn(f"Error calculating log loss: {e}")
            return None
    
    def brier_score(self) -> Optional[float]:
        """
        Calculate Brier score.
        
        Mean squared difference between predicted probability and actual outcome.
        Lower is better. Range: [0, 1]
        """
        if self.y_score is None:
            warnings.warn("y_score not provided. Cannot calculate Brier score.")
            return None
        
        try:
            return brier_score_loss(
                self.y_true,
                self.y_score,
                sample_weight=self.sample_weight,
                pos_label=self.pos_label
            )
        except Exception as e:
            warnings.warn(f"Error calculating Brier score: {e}")
            return None
    
    def expected_calibration_error(self, n_bins: int = 10) -> Optional[float]:
        """
        Calculate Expected Calibration Error (ECE).
        
        Measures how well probabilities are calibrated.
        Lower is better. Range: [0, 1]
        
        Parameters:
            n_bins: Number of bins for calibration
        """
        if self.y_score is None:
            warnings.warn("y_score not provided. Cannot calculate ECE.")
            return None
        
        # Bin predictions
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        total_samples = len(self.y_true)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (self.y_score >= bin_lower) & (self.y_score < bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                # Average predicted probability in bin
                avg_pred_prob = np.mean(self.y_score[in_bin])
                
                # Actual fraction of positives in bin
                avg_true = np.mean(self.y_true[in_bin])
                
                # Add to ECE
                ece += np.abs(avg_true - avg_pred_prob) * prop_in_bin
        
        return ece
    
    # ==================== Cost-Sensitive Metrics ====================
    
    def cost_savings(
        self,
        fraud_cost: float = 100.0,
        investigation_cost: float = 10.0,
        missed_fraud_cost: float = 100.0
    ) -> Dict[str, float]:
        """
        Calculate cost-based metrics for business impact.
        
        Parameters:
            fraud_cost: Average cost per fraud transaction
            investigation_cost: Cost to investigate each alert
            missed_fraud_cost: Cost of missing a fraud (could be different from fraud_cost)
            
        Returns:
            Dictionary with cost metrics
        """
        # Calculate costs
        detected_frauds = self.tp
        missed_frauds = self.fn
        false_alerts = self.fp
        correct_rejections = self.tn
        
        # Cost calculations
        fraud_loss = missed_frauds * missed_fraud_cost
        investigation_cost_total = false_alerts * investigation_cost
        savings_from_detected = detected_frauds * fraud_cost
        
        # Net savings
        net_savings = savings_from_detected - fraud_loss - investigation_cost_total
        
        # Return on investment
        if investigation_cost_total > 0:
            roi = (net_savings - investigation_cost_total) / investigation_cost_total
        else:
            roi = 0.0
        
        # Cost per alerted transaction
        total_alerts = detected_frauds + false_alerts
        if total_alerts > 0:
            cost_per_alert = investigation_cost_total / total_alerts
        else:
            cost_per_alert = 0.0
        
        return {
            'net_savings': net_savings,
            'fraud_loss': fraud_loss,
            'investigation_cost': investigation_cost_total,
            'savings_from_detected': savings_from_detected,
            'roi': roi,
            'cost_per_alert': cost_per_alert,
            'detected_frauds': detected_frauds,
            'missed_frauds': missed_frauds,
            'false_alerts': false_alerts
        }
    
    def profit_curve(
        self,
        fraud_amounts: Optional[np.ndarray] = None,
        investigation_cost: float = 10.0,
        n_thresholds: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate profit curve across different thresholds.
        
        Parameters:
            fraud_amounts: Actual fraud amounts per transaction
            investigation_cost: Cost to investigate each alert
            n_thresholds: Number of thresholds to evaluate
            
        Returns:
            thresholds, profits: Arrays of thresholds and corresponding profits
        """
        if self.y_score is None:
            raise ValueError("y_score required for profit curve")
        
        thresholds = np.linspace(0, 1, n_thresholds)
        profits = []
        
        for threshold in thresholds:
            y_pred_thresh = (self.y_score >= threshold).astype(int)
            
            # Recalculate confusion matrix at this threshold
            tn, fp, fn, tp = confusion_matrix(
                self.y_true, 
                y_pred_thresh
            ).ravel()
            
            # Calculate profit
            if fraud_amounts is not None:
                # Use actual fraud amounts
                detected_fraud_amounts = fraud_amounts[
                    (self.y_true == 1) & (y_pred_thresh == 1)
                ].sum()
                missed_fraud_amounts = fraud_amounts[
                    (self.y_true == 1) & (y_pred_thresh == 0)
                ].sum()
                
                profit = detected_fraud_amounts - missed_fraud_amounts - (fp * investigation_cost)
            else:
                # Use counts with assumed fixed cost per fraud
                profit = tp * 100 - fn * 100 - fp * investigation_cost
            
            profits.append(profit)
        
        return thresholds, np.array(profits)
    
    # ==================== Detection Rate Metrics ====================
    
    def precision_at_k(self, k: int) -> float:
        """
        Calculate precision at top-k predictions.
        
        Useful for fraud detection where we might only investigate
        the most suspicious transactions.
        
        Parameters:
            k: Number of top predictions to consider
        """
        if self.y_score is None:
            raise ValueError("y_score required for precision@k")
        
        # Get indices of top-k predictions
        top_k_idx = np.argsort(self.y_score)[-k:]
        
        # Calculate precision among these
        return np.mean(self.y_true[top_k_idx] == 1)
    
    def recall_at_k(self, k: int) -> float:
        """
        Calculate recall at top-k predictions.
        
        Parameters:
            k: Number of top predictions to consider
        """
        if self.y_score is None:
            raise ValueError("y_score required for recall@k")
        
        # Get indices of top-k predictions
        top_k_idx = np.argsort(self.y_score)[-k:]
        
        # Calculate recall
        detected_frauds = np.sum(self.y_true[top_k_idx] == 1)
        total_frauds = np.sum(self.y_true == 1)
        
        if total_frauds == 0:
            return 0.0
        
        return detected_frauds / total_frauds
    
    def fdr_at_k(self, k: int) -> float:
        """
        Calculate False Discovery Rate at top-k predictions.
        
        Parameters:
            k: Number of top predictions to consider
        """
        return 1 - self.precision_at_k(k)
    
    def lift_at_k(self, k: int) -> float:
        """
        Calculate lift at top-k predictions.
        
        Lift = (precision@k) / (overall fraud rate)
        
        Parameters:
            k: Number of top predictions to consider
        """
        precision_k = self.precision_at_k(k)
        overall_rate = self.n_pos / self.n_samples
        
        if overall_rate == 0:
            return 0.0
        
        return precision_k / overall_rate
    
    # ==================== Threshold-Independent Metrics ====================
    
    def max_f1_score(self) -> Tuple[float, float]:
        """
        Find maximum possible F1 score by threshold optimization.
        
        Returns:
            max_f1, optimal_threshold
        """
        if self.y_score is None:
            raise ValueError("y_score required for max F1 calculation")
        
        precisions, recalls, thresholds = precision_recall_curve(
            self.y_true, 
            self.y_score
        )
        
        # Calculate F1 for each point (excluding last point where recall=0)
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (
            precisions[:-1] + recalls[:-1] + 1e-10
        )
        
        best_idx = np.argmax(f1_scores)
        
        return f1_scores[best_idx], thresholds[best_idx]
    
    def optimal_threshold(
        self,
        cost_matrix: Optional[np.ndarray] = None,
        objective: str = 'f1'
    ) -> float:
        """
        Find optimal threshold based on objective.
        
        Parameters:
            cost_matrix: 2x2 cost matrix [TN, FP; FN, TP]
            objective: 'f1', 'precision', 'recall', 'cost', 'profit'
            
        Returns:
            optimal_threshold
        """
        if self.y_score is None:
            raise ValueError("y_score required for threshold optimization")
        
        thresholds = np.linspace(0, 1, 1000)
        scores = []
        
        for threshold in thresholds:
            y_pred_thresh = (self.y_score >= threshold).astype(int)
            
            if objective == 'f1':
                score = f1_score(self.y_true, y_pred_thresh)
            elif objective == 'precision':
                score = precision_score(self.y_true, y_pred_thresh)
            elif objective == 'recall':
                score = recall_score(self.y_true, y_pred_thresh)
            elif objective == 'cost' and cost_matrix is not None:
                # Calculate total cost
                tn, fp, fn, tp = confusion_matrix(
                    self.y_true, y_pred_thresh
                ).ravel()
                total_cost = (
                    tn * cost_matrix[0, 0] +
                    fp * cost_matrix[0, 1] +
                    fn * cost_matrix[1, 0] +
                    tp * cost_matrix[1, 1]
                )
                score = -total_cost  # Negative because we want to maximize
            else:
                raise ValueError(f"Unknown objective: {objective}")
            
            scores.append(score)
        
        best_idx = np.argmax(scores)
        return thresholds[best_idx]
    
    # ==================== Statistical Significance ====================
    
    def confidence_interval(
        self,
        metric_func: Callable,
        alpha: float = 0.05,
        n_bootstrap: int = 1000,
        random_state: int = 42
    ) -> Tuple[float, float, float]:
        """
        Calculate bootstrap confidence interval for a metric.
        
        Parameters:
            metric_func: Function that returns a metric value
            alpha: Significance level
            n_bootstrap: Number of bootstrap samples
            random_state: Random seed
            
        Returns:
            lower_bound, upper_bound, point_estimate
        """
        np.random.seed(random_state)
        
        # Point estimate
        point_estimate = metric_func()
        
        # Bootstrap
        bootstrap_estimates = []
        n_samples = len(self.y_true)
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            
            # Create temporary metrics object for bootstrap sample
            temp_metrics = FraudMetrics(
                self.y_true[indices],
                self.y_pred[indices],
                self.y_score[indices] if self.y_score is not None else None,
                self.sample_weight[indices] if self.sample_weight is not None else None
            )
            
            # Calculate metric
            try:
                bootstrap_estimates.append(metric_func(temp_metrics))
            except:
                continue
        
        if len(bootstrap_estimates) == 0:
            return point_estimate, point_estimate, 0.0
        
        # Calculate confidence interval
        bootstrap_estimates = np.array(bootstrap_estimates)
        lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
        upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))
        std_error = np.std(bootstrap_estimates)
        
        return lower, upper, std_error
    
    # ==================== Comprehensive Report ====================
    
    def get_comprehensive_report(
        self,
        include_ci: bool = True,
        alpha: float = 0.05,
        n_bootstrap: int = 100
    ) -> Dict[str, MetricResult]:
        """
        Generate comprehensive evaluation report.
        
        Parameters:
            include_ci: Whether to include confidence intervals
            alpha: Significance level for confidence intervals
            n_bootstrap: Number of bootstrap samples for CI
            
        Returns:
            Dictionary of metric results
        """
        report = {}
        
        # Basic metrics
        metrics_to_calc = {
            'accuracy': (self.accuracy, "Overall accuracy"),
            'precision': (self.precision, "Precision (Positive Predictive Value)"),
            'recall': (self.recall, "Recall (Sensitivity)"),
            'specificity': (self.specificity, "Specificity (True Negative Rate)"),
            'f1_score': (lambda: self.f1_score(), "F1 Score"),
            'f2_score': (lambda: self.f1_score(beta=2), "F2 Score (weights recall more)"),
            'f05_score': (lambda: self.f1_score(beta=0.5), "F0.5 Score (weights precision more)"),
            'matthews_correlation': (self.matthews_correlation, "Matthews Correlation Coefficient"),
            'cohen_kappa': (self.cohen_kappa, "Cohen's Kappa")
        }
        
        for name, (func, desc) in metrics_to_calc.items():
            try:
                value = func()
                if include_ci:
                    lower, upper, std = self.confidence_interval(
                        lambda x: func(), alpha, n_bootstrap
                    )
                    report[name] = MetricResult(
                        name=name,
                        value=value,
                        confidence_interval=(lower, upper),
                        std_error=std,
                        n_samples=self.n_samples,
                        description=desc
                    )
                else:
                    report[name] = MetricResult(
                        name=name,
                        value=value,
                        n_samples=self.n_samples,
                        description=desc
                    )
            except Exception as e:
                warnings.warn(f"Error calculating {name}: {e}")
        
        # Probability-based metrics
        if self.y_score is not None:
            prob_metrics = {
                'roc_auc': (self.roc_auc, "Area Under ROC Curve"),
                'pr_auc': (self.pr_auc, "Area Under Precision-Recall Curve"),
                'log_loss': (self.log_loss_score, "Logarithmic Loss"),
                'brier_score': (self.brier_score, "Brier Score"),
                'ece': (self.expected_calibration_error, "Expected Calibration Error")
            }
            
            for name, (func, desc) in prob_metrics.items():
                try:
                    value = func()
                    if value is not None:
                        if include_ci:
                            lower, upper, std = self.confidence_interval(
                                lambda x: func(), alpha, n_bootstrap
                            )
                            report[name] = MetricResult(
                                name=name,
                                value=value,
                                confidence_interval=(lower, upper),
                                std_error=std,
                                n_samples=self.n_samples,
                                description=desc
                            )
                        else:
                            report[name] = MetricResult(
                                name=name,
                                value=value,
                                n_samples=self.n_samples,
                                description=desc
                            )
                except Exception as e:
                    warnings.warn(f"Error calculating {name}: {e}")
        
        # Add confusion matrix counts
        report['confusion_matrix'] = {
            'true_negatives': self.tn,
            'false_positives': self.fp,
            'false_negatives': self.fn,
            'true_positives': self.tp
        }
        
        return report


def compare_models(
    y_true: np.ndarray,
    predictions: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]],
    metrics: List[str] = ['f1_score', 'roc_auc', 'pr_auc'],
    alpha: float = 0.05,
    n_bootstrap: int = 1000
) -> pd.DataFrame:
    """
    Compare multiple models using statistical tests.
    
    Parameters:
        y_true: True labels
        predictions: Dictionary mapping model names to (y_pred, y_score)
        metrics: List of metrics to compare
        alpha: Significance level
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for model_name, (y_pred, y_score) in predictions.items():
        metrics_calc = FraudMetrics(y_true, y_pred, y_score)
        report = metrics_calc.get_comprehensive_report(
            include_ci=True,
            alpha=alpha,
            n_bootstrap=n_bootstrap
        )
        
        for metric in metrics:
            if metric in report:
                result = report[metric]
                results.append({
                    'model': model_name,
                    'metric': metric,
                    'value': result.value,
                    'ci_lower': result.confidence_interval[0] if result.confidence_interval else None,
                    'ci_upper': result.confidence_interval[1] if result.confidence_interval else None,
                    'std_error': result.std_error
                })
    
    # Perform statistical tests for pairwise comparisons
    df_results = pd.DataFrame(results)
    
    # Add significance stars based on confidence intervals
    if len(predictions) > 1:
        for metric in metrics:
            metric_results = df_results[df_results['metric'] == metric]
            if len(metric_results) >= 2:
                best_value = metric_results['value'].max()
                best_model = metric_results.loc[
                    metric_results['value'].idxmax(), 'model'
                ]
                
                # Check if best model is significantly better than others
                best_ci_lower = metric_results.loc[
                    metric_results['model'] == best_model, 'ci_lower'
                ].iloc[0]
                
                for idx, row in metric_results.iterrows():
                    if row['model'] != best_model:
                        if row['ci_upper'] < best_ci_lower:
                            df_results.loc[idx, 'significance'] = '**'  # Significant at alpha
                        else:
                            df_results.loc[idx, 'significance'] = ''
    
    return df_results


def calculate_lift_chart(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_deciles: int = 10
) -> pd.DataFrame:
    """
    Calculate lift chart data for model evaluation.
    
    Parameters:
        y_true: True labels
        y_score: Prediction scores
        n_deciles: Number of deciles
        
    Returns:
        DataFrame with lift chart data
    """
    # Sort by score descending
    sorted_idx = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[sorted_idx]
    
    # Calculate cumulative statistics
    n_samples = len(y_true)
    decile_size = n_samples // n_deciles
    
    lift_data = []
    
    for i in range(n_deciles):
        start_idx = i * decile_size
        end_idx = min((i + 1) * decile_size, n_samples)
        
        # Samples in this decile
        decile_true = y_true_sorted[start_idx:end_idx]
        
        # Calculate metrics
        n_samples_decile = len(decile_true)
        n_fraud_decile = decile_true.sum()
        fraud_rate_decile = n_fraud_decile / n_samples_decile if n_samples_decile > 0 else 0
        
        # Cumulative up to this decile
        cum_true = y_true_sorted[:end_idx]
        cum_fraud = cum_true.sum()
        cum_samples = end_idx
        cum_fraud_rate = cum_fraud / cum_samples if cum_samples > 0 else 0
        
        # Overall fraud rate
        overall_fraud_rate = y_true.mean()
        
        # Lift
        lift = fraud_rate_decile / overall_fraud_rate if overall_fraud_rate > 0 else 0
        
        lift_data.append({
            'decile': i + 1,
            'samples': n_samples_decile,
            'fraud_count': n_fraud_decile,
            'fraud_rate': fraud_rate_decile,
            'cumulative_samples': cum_samples,
            'cumulative_fraud': cum_fraud,
            'cumulative_fraud_rate': cum_fraud_rate,
            'cumulative_fraud_percentage': cum_fraud / y_true.sum() if y_true.sum() > 0 else 0,
            'lift': lift
        })
    
    return pd.DataFrame(lift_data)


def calculate_gini_coefficient(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Calculate Gini coefficient (normalized).
    
    Gini = 2 * AUC - 1
    
    Parameters:
        y_true: True labels
        y_score: Prediction scores
        
    Returns:
        Gini coefficient
    """
    auc = roc_auc_score(y_true, y_score)
    return 2 * auc - 1


def calculate_kolmogorov_smirnov_statistic(
    y_true: np.ndarray,
    y_score: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate Kolmogorov-Smirnov statistic for model separation.
    
    KS = max(TPR - FPR)
    
    Parameters:
        y_true: True labels
        y_score: Prediction scores
        
    Returns:
        ks_statistic, optimal_threshold
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    ks = np.max(tpr - fpr)
    optimal_idx = np.argmax(tpr - fpr)
    
    return ks, thresholds[optimal_idx]


def calculate_hmeasure(
    y_true: np.ndarray,
    y_score: np.ndarray,
    c: float = 1.0
) -> float:
    """
    Calculate H-measure (alternative to AUC that accounts for class imbalance).
    
    Parameters:
        y_true: True labels
        y_score: Prediction scores
        c: Cost ratio parameter
        
    Returns:
        H-measure value
    """
    # Implementation based on Hand (2009)
    # This is a simplified version
    from sklearn.metrics import roc_curve
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    
    # Calculate the minimum cost line
    pi = y_true.mean()  # proportion of positives
    cost = pi * (1 - tpr) + (1 - pi) * c * fpr
    
    h_measure = 1 - np.min(cost) / min(pi, (1 - pi) * c)
    
    return h_measure


# Example usage
if __name__ == "__main__":
    """
    Example demonstrating metrics calculation.
    """
    
    print("=" * 60)
    print("Fraud Detection Metrics Examples")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # True labels (5% fraud)
    y_true = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    
    # Predictions (imperfect model)
    y_pred = y_true.copy()
    # Add some errors
    error_idx = np.random.choice(n_samples, 100, replace=False)
    y_pred[error_idx] = 1 - y_pred[error_idx]
    
    # Probabilities
    y_score = y_true + 0.3 * np.random.randn(n_samples)
    y_score = expit(y_score)  # Convert to probabilities
    
    # Initialize metrics calculator
    metrics = FraudMetrics(y_true, y_pred, y_score)
    
    # Calculate basic metrics
    print("\n1. Basic Metrics:")
    print(f"  Accuracy: {metrics.accuracy():.4f}")
    print(f"  Precision: {metrics.precision():.4f}")
    print(f"  Recall: {metrics.recall():.4f}")
    print(f"  Specificity: {metrics.specificity():.4f}")
    print(f"  F1 Score: {metrics.f1_score():.4f}")
    print(f"  MCC: {metrics.matthews_correlation():.4f}")
    
    # Calculate probability-based metrics
    print("\n2. Probability-Based Metrics:")
    print(f"  ROC AUC: {metrics.roc_auc():.4f}")
    print(f"  PR AUC: {metrics.pr_auc():.4f}")
    print(f"  Log Loss: {metrics.log_loss_score():.4f}")
    print(f"  Brier Score: {metrics.brier_score():.4f}")
    print(f"  ECE: {metrics.expected_calibration_error():.4f}")
    
    # Calculate cost metrics
    print("\n3. Cost Metrics:")
    costs = metrics.cost_savings(
        fraud_cost=1000,
        investigation_cost=50,
        missed_fraud_cost=1000
    )
    for key, value in costs.items():
        if isinstance(value, float):
            print(f"  {key}: ${value:,.2f}" if 'cost' in key or 'savings' in key else f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Calculate detection rate metrics
    print("\n4. Detection Rate Metrics:")
    print(f"  Precision@100: {metrics.precision_at_k(100):.4f}")
    print(f"  Recall@100: {metrics.recall_at_k(100):.4f}")
    print(f"  Lift@100: {metrics.lift_at_k(100):.4f}")
    
    # Find optimal threshold
    max_f1, opt_thresh = metrics.max_f1_score()
    print(f"\n5. Optimal Threshold:")
    print(f"  Max F1 Score: {max_f1:.4f}")
    print(f"  Optimal Threshold: {opt_thresh:.4f}")
    
    # Get comprehensive report
    print("\n6. Comprehensive Report (with 95% CI):")
    report = metrics.get_comprehensive_report(include_ci=True, n_bootstrap=100)
    for name, result in report.items():
        if isinstance(result, MetricResult):
            print(f"  {name}: {result.value:.4f} "
                  f"(95% CI: [{result.confidence_interval[0]:.4f}, "
                  f"{result.confidence_interval[1]:.4f}])")
    
    print("\n" + "=" * 60)
    print("Metrics module loaded successfully!")
    print("=" * 60)