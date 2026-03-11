# scripts/evaluate_model.py
#!/usr/bin/env python3
"""
Model Evaluation Script for VeritasFinancial Fraud Detection System

This script provides comprehensive evaluation of trained fraud detection models,
including multiple metrics, visualizations, and business impact analysis.
It handles both classical ML and deep learning models with extensive reporting.

Author: VeritasFinancial Data Science Team
Version: 2.0.0
"""

# ============================================================================
# IMPORTS SECTION
# ============================================================================
# Standard library imports
import os
import sys
import json
import argparse
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import pickle
import joblib

# ============================================================================
# THIRD-PARTY IMPORTS
# ============================================================================
# Data manipulation and analysis
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
    auc,
    log_loss,
    brier_score_loss,
    matthews_corrcoef,
    cohen_kappa_score
)

# Advanced metrics
from sklearn.calibration import calibration_curve
from sklearn.metrics import det_curve

# For SHAP explainability
import shap

# For feature importance
import eli5
from eli5.sklearn import PermutationImportance

# For time series evaluation
from statsmodels.tsa.stattools import acf

# ============================================================================
# DEEP LEARNING IMPORTS
# ============================================================================
import torch
import torch.nn.functional as F

# ============================================================================
# UTILITY IMPORTS
# ============================================================================
from tqdm import tqdm
import yaml

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Project imports
from src.models.classical_ml.xgboost_model import FraudXGBoostModel
from src.models.classical_ml.lightgbm_model import FraudLightGBMModel
from src.models.deep_learning.neural_networks import FraudNeuralNetwork
from src.utils.logger import setup_logger
from src.utils.config_manager import ConfigManager
from src.data_preprocessing.pipelines.preprocessing_pipeline import FraudPreprocessingPipeline

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
sns.set_context("talk")

# ============================================================================
# LOGGING SETUP
# ============================================================================
logger = setup_logger(
    name='model_evaluation',
    log_file='logs/evaluation.log',
    level=logging.INFO
)


# ============================================================================
# EVALUATION METRICS CALCULATOR CLASS
# ============================================================================
class EvaluationMetricsCalculator:
    """
    Comprehensive metrics calculator for fraud detection models.
    
    This class calculates all relevant metrics for fraud detection,
    including standard classification metrics, business-specific metrics,
    and provides confidence intervals through bootstrapping.
    
    Attributes:
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels
        y_proba (np.array): Prediction probabilities
        metrics (dict): Stores all calculated metrics
    """
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray):
        """
        Initialize the metrics calculator with predictions and true labels.
        
        Args:
            y_true: Ground truth labels (0 for non-fraud, 1 for fraud)
            y_pred: Predicted labels from model
            y_proba: Prediction probabilities for positive class
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_proba = np.array(y_proba)
        self.metrics = {}
        
        # Validate inputs
        self._validate_inputs()
        
    def _validate_inputs(self):
        """Validate input arrays for consistency and quality."""
        # Check lengths match
        assert len(self.y_true) == len(self.y_pred) == len(self.y_proba), \
            "All input arrays must have the same length"
        
        # Check for NaN values
        assert not np.any(pd.isnull(self.y_true)), "y_true contains NaN values"
        assert not np.any(pd.isnull(self.y_pred)), "y_pred contains NaN values"
        assert not np.any(pd.isnull(self.y_proba)), "y_proba contains NaN values"
        
        # Check value ranges
        assert np.all(np.isin(self.y_true, [0, 1])), "y_true must contain only 0 and 1"
        assert np.all(np.isin(self.y_pred, [0, 1])), "y_pred must contain only 0 and 1"
        assert np.all((self.y_proba >= 0) & (self.y_proba <= 1)), \
            "y_proba must be between 0 and 1"
            
        logger.info(f"Input validation passed. Shape: {self.y_true.shape}")
    
    def calculate_all_metrics(self) -> Dict:
        """
        Calculate comprehensive set of evaluation metrics.
        
        Returns:
            Dictionary containing all calculated metrics organized by category
        """
        logger.info("Starting comprehensive metrics calculation...")
        
        # Calculate metrics by category
        self._calculate_confusion_metrics()
        self._calculate_probabilistic_metrics()
        self._calculate_ranking_metrics()
        self._calculate_cost_sensitive_metrics()
        self._calculate_calibration_metrics()
        self._calculate_confidence_intervals()
        self._calculate_statistical_tests()
        
        # Add summary statistics
        self.metrics['summary'] = self._generate_summary()
        
        logger.info("Metrics calculation completed successfully")
        return self.metrics
    
    def _calculate_confusion_metrics(self):
        """
        Calculate confusion matrix based metrics.
        
        Computes:
        - True Positives (TP), False Positives (FP)
        - True Negatives (TN), False Negatives (FN)
        - Derived metrics: accuracy, precision, recall, F1, etc.
        """
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred).ravel()
        
        # Store raw counts
        confusion_counts = {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'total_samples': len(self.y_true),
            'positive_samples': int(np.sum(self.y_true)),
            'negative_samples': int(len(self.y_true) - np.sum(self.y_true))
        }
        self.metrics['confusion_counts'] = confusion_counts
        
        # Calculate derived metrics with epsilon to avoid division by zero
        epsilon = 1e-15
        
        # Accuracy: (TP + TN) / Total
        accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
        
        # Precision: TP / (TP + FP)
        precision = tp / (tp + fp + epsilon)
        
        # Recall (Sensitivity): TP / (TP + FN)
        recall = tp / (tp + fn + epsilon)
        
        # Specificity: TN / (TN + FP)
        specificity = tn / (tn + fp + epsilon)
        
        # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        
        # F2 Score (weights recall higher)
        f2 = (5 * precision * recall) / (4 * precision + recall + epsilon)
        
        # F0.5 Score (weights precision higher)
        f05 = (1.25 * precision * recall) / (0.25 * precision + recall + epsilon)
        
        # Matthews Correlation Coefficient (MCC)
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + epsilon)
        
        # Cohen's Kappa
        kappa = cohen_kappa_score(self.y_true, self.y_pred)
        
        # Store metrics
        derived_metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'specificity': float(specificity),
            'f1_score': float(f1),
            'f2_score': float(f2),
            'f05_score': float(f05),
            'mcc': float(mcc),
            'cohen_kappa': float(kappa)
        }
        self.metrics['derived_metrics'] = derived_metrics
        
        logger.info(f"Confusion metrics: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    
    def _calculate_probabilistic_metrics(self):
        """
        Calculate metrics based on prediction probabilities.
        
        Computes:
        - ROC-AUC, PR-AUC
        - Log Loss, Brier Score
        - Cross-entropy metrics
        """
        # ROC-AUC (Area Under ROC Curve)
        roc_auc = roc_auc_score(self.y_true, self.y_proba)
        
        # PR-AUC (Area Under Precision-Recall Curve)
        pr_auc = average_precision_score(self.y_true, self.y_proba)
        
        # Log Loss (Cross-Entropy Loss)
        logloss = log_loss(self.y_true, self.y_proba)
        
        # Brier Score (Mean Squared Error of probabilities)
        brier = brier_score_loss(self.y_true, self.y_proba)
        
        # Calculate ROC curve for detailed analysis
        fpr, tpr, roc_thresholds = roc_curve(self.y_true, self.y_proba)
        
        # Calculate PR curve for detailed analysis
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
            self.y_true, self.y_proba
        )
        
        # Calculate optimal threshold using Youden's J statistic
        youden_j = tpr - fpr
        optimal_youden_idx = np.argmax(youden_j)
        optimal_threshold_youden = roc_thresholds[optimal_youden_idx]
        
        # Calculate optimal threshold for F1 score
        f1_scores = 2 * (precision_curve * recall_curve) / \
                    (precision_curve + recall_curve + 1e-10)
        optimal_f1_idx = np.argmax(f1_scores)
        optimal_threshold_f1 = pr_thresholds[optimal_f1_idx] \
            if optimal_f1_idx < len(pr_thresholds) else 0.5
        
        # Store probabilistic metrics
        probabilistic_metrics = {
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'log_loss': float(logloss),
            'brier_score': float(brier),
            'optimal_threshold_youden': float(optimal_threshold_youden),
            'optimal_threshold_f1': float(optimal_threshold_f1)
        }
        self.metrics['probabilistic_metrics'] = probabilistic_metrics
        
        # Store curves for visualization
        self.metrics['curves'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'roc_thresholds': roc_thresholds.tolist(),
            'precision_curve': precision_curve.tolist(),
            'recall_curve': recall_curve.tolist(),
            'pr_thresholds': pr_thresholds.tolist()
        }
        
        logger.info(f"Probabilistic metrics: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")
    
    def _calculate_ranking_metrics(self):
        """
        Calculate ranking-based metrics useful for fraud detection.
        
        Computes:
        - Precision@K
        - Average Precision@K
        - Normalized Discounted Cumulative Gain (NDCG)
        """
        # Get indices sorted by probability descending
        sorted_indices = np.argsort(self.y_proba)[::-1]
        y_true_sorted = self.y_true[sorted_indices]
        
        # Calculate Precision@K for different K values
        precision_at_k = {}
        ks = [10, 50, 100, 500, 1000, 5000, 10000]
        
        for k in ks:
            if k <= len(y_true_sorted):
                precision_at_k[f'precision@{k}'] = float(
                    np.mean(y_true_sorted[:k])
                )
        
        # Calculate Average Precision@K
        avg_precision_at_k = {}
        for k in ks:
            if k <= len(y_true_sorted):
                cum_precision = np.cumsum(y_true_sorted[:k]) / np.arange(1, k + 1)
                avg_precision_at_k[f'avg_precision@{k}'] = float(np.mean(cum_precision))
        
        # Calculate NDCG@K
        ndcg_at_k = {}
        for k in ks:
            if k <= len(y_true_sorted):
                dcg = np.sum(y_true_sorted[:k] / np.log2(np.arange(2, k + 2)))
                idcg = np.sum(np.sort(y_true_sorted)[::-1][:k] / np.log2(np.arange(2, k + 2)))
                ndcg_at_k[f'ndcg@{k}'] = float(dcg / (idcg + 1e-10))
        
        self.metrics['ranking_metrics'] = {
            'precision_at_k': precision_at_k,
            'avg_precision_at_k': avg_precision_at_k,
            'ndcg_at_k': ndcg_at_k
        }
    
    def _calculate_cost_sensitive_metrics(self):
        """
        Calculate cost-sensitive metrics for business impact analysis.
        
        Considers:
        - Cost of fraud (average transaction amount)
        - Cost of investigation (false positives)
        - Net savings from fraud detection
        """
        # Define costs (these would typically come from business stakeholders)
        fraud_cost_multiplier = 100  # Cost per fraud transaction
        investigation_cost = 10       # Cost per false positive investigation
        
        # Calculate raw counts
        tn = self.metrics['confusion_counts']['true_negatives']
        fp = self.metrics['confusion_counts']['false_positives']
        fn = self.metrics['confusion_counts']['false_negatives']
        tp = self.metrics['confusion_counts']['true_positives']
        
        # Calculate costs and savings
        fraud_prevention_savings = tp * fraud_cost_multiplier
        investigation_costs = fp * investigation_cost
        missed_fraud_costs = fn * fraud_cost_multiplier
        
        net_savings = fraud_prevention_savings - investigation_costs - missed_fraud_costs
        
        # Calculate cost-sensitive metrics
        cost_metrics = {
            'fraud_prevention_savings': float(fraud_prevention_savings),
            'investigation_costs': float(investigation_costs),
            'missed_fraud_costs': float(missed_fraud_costs),
            'net_savings': float(net_savings),
            'roi': float(net_savings / (investigation_costs + 1e-10)),
            'cost_per_true_positive': float(investigation_costs / (tp + 1e-10)),
            'savings_per_transaction': float(net_savings / len(self.y_true))
        }
        
        self.metrics['cost_metrics'] = cost_metrics
    
    def _calculate_calibration_metrics(self):
        """
        Calculate model calibration metrics.
        
        Checks how well predicted probabilities match actual frequencies.
        """
        # Calculate calibration curve
        fraction_positives, mean_predicted = calibration_curve(
            self.y_true, self.y_proba, n_bins=10, strategy='uniform'
        )
        
        # Calculate expected calibration error (ECE)
        bin_counts, _ = np.histogram(self.y_proba, bins=10)
        ece = np.sum(bin_counts * np.abs(fraction_positives - mean_predicted)) / np.sum(bin_counts)
        
        # Calculate maximum calibration error (MCE)
        mce = np.max(np.abs(fraction_positives - mean_predicted))
        
        # Store calibration metrics
        self.metrics['calibration_metrics'] = {
            'ece': float(ece),
            'mce': float(mce),
            'calibration_curve': {
                'fraction_positives': fraction_positives.tolist(),
                'mean_predicted': mean_predicted.tolist()
            }
        }
    
    def _calculate_confidence_intervals(self, n_bootstraps: int = 1000, confidence_level: float = 0.95):
        """
        Calculate confidence intervals for key metrics using bootstrapping.
        
        Args:
            n_bootstraps: Number of bootstrap samples
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        """
        logger.info(f"Calculating confidence intervals with {n_bootstraps} bootstraps...")
        
        # Metrics to bootstrap
        key_metrics = ['roc_auc', 'pr_auc', 'f1_score', 'precision', 'recall']
        bootstrap_results = {metric: [] for metric in key_metrics}
        
        # Perform bootstrapping
        rng = np.random.RandomState(42)
        n_samples = len(self.y_true)
        
        for i in tqdm(range(n_bootstraps), desc="Bootstrapping"):
            # Sample with replacement
            indices = rng.choice(n_samples, n_samples, replace=True)
            y_true_boot = self.y_true[indices]
            y_proba_boot = self.y_proba[indices]
            
            # Calculate metrics for this bootstrap sample
            if np.sum(y_true_boot) > 0:  # Ensure we have both classes
                roc_auc_boot = roc_auc_score(y_true_boot, y_proba_boot)
                pr_auc_boot = average_precision_score(y_true_boot, y_proba_boot)
                
                # For classification metrics, need to use optimal threshold
                y_pred_boot = (y_proba_boot >= 0.5).astype(int)
                f1_boot = f1_score(y_true_boot, y_pred_boot)
                precision_boot = precision_score(y_true_boot, y_pred_boot)
                recall_boot = recall_score(y_true_boot, y_pred_boot)
                
                bootstrap_results['roc_auc'].append(roc_auc_boot)
                bootstrap_results['pr_auc'].append(pr_auc_boot)
                bootstrap_results['f1_score'].append(f1_boot)
                bootstrap_results['precision'].append(precision_boot)
                bootstrap_results['recall'].append(recall_boot)
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        confidence_intervals = {}
        for metric, values in bootstrap_results.items():
            if values:  # Check if list is not empty
                confidence_intervals[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'ci_lower': float(np.percentile(values, lower_percentile)),
                    'ci_upper': float(np.percentile(values, upper_percentile)),
                    'ci_level': confidence_level
                }
        
        self.metrics['confidence_intervals'] = confidence_intervals
    
    def _calculate_statistical_tests(self):
        """
        Perform statistical tests on model predictions.
        
        Includes:
        - McNemar's test for model comparison
        - Delong's test for ROC-AUC comparison
        - Hosmer-Lemeshow test for goodness-of-fit
        """
        from statsmodels.stats.contingency_tables import mcnemar
        from scipy import stats
        
        # McNemar's test for model performance
        tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred).ravel()
        contingency_table = [[tn, fp], [fn, tp]]
        
        # Apply continuity correction for small samples
        result = mcnemar(contingency_table, exact=False, correction=True)
        
        self.metrics['statistical_tests'] = {
            'mcnemar_statistic': float(result.statistic),
            'mcnemar_pvalue': float(result.pvalue)
        }
        
        # Hosmer-Lemeshow test for goodness-of-fit
        # Group by predicted probability deciles
        n_groups = 10
        y_true_array = self.y_true
        y_proba_array = self.y_proba
        
        # Sort by predicted probability
        sorted_indices = np.argsort(y_proba_array)
        y_true_sorted = y_true_array[sorted_indices]
        y_proba_sorted = y_proba_array[sorted_indices]
        
        # Split into deciles
        groups = np.array_split(range(len(y_true_sorted)), n_groups)
        
        # Calculate observed and expected
        observed = []
        expected = []
        
        for group in groups:
            observed.append(np.sum(y_true_sorted[group]))
            expected.append(np.sum(y_proba_sorted[group]))
        
        observed = np.array(observed)
        expected = np.array(expected)
        
        # Calculate Hosmer-Lemeshow statistic
        hl_statistic = np.sum((observed - expected) ** 2 / (expected * (1 - expected/np.mean(expected))))
        hl_pvalue = 1 - stats.chi2.cdf(hl_statistic, n_groups - 2)
        
        self.metrics['statistical_tests'].update({
            'hosmer_lemeshow_statistic': float(hl_statistic),
            'hosmer_lemeshow_pvalue': float(hl_pvalue)
        })
    
    def _generate_summary(self) -> Dict:
        """
        Generate a summary of key metrics.
        
        Returns:
            Dictionary with summary statistics
        """
        # Get key metrics
        key_metrics = {}
        
        # Add derived metrics
        if 'derived_metrics' in self.metrics:
            key_metrics.update({
                'f1': self.metrics['derived_metrics']['f1_score'],
                'precision': self.metrics['derived_metrics']['precision'],
                'recall': self.metrics['derived_metrics']['recall'],
                'accuracy': self.metrics['derived_metrics']['accuracy']
            })
        
        # Add probabilistic metrics
        if 'probabilistic_metrics' in self.metrics:
            key_metrics.update({
                'roc_auc': self.metrics['probabilistic_metrics']['roc_auc'],
                'pr_auc': self.metrics['probabilistic_metrics']['pr_auc']
            })
        
        # Calculate overall grade (custom scoring)
        # Weight different metrics based on importance for fraud detection
        weights = {
            'f1': 0.25,
            'recall': 0.30,  # Recall is very important for fraud
            'precision': 0.20,
            'roc_auc': 0.15,
            'pr_auc': 0.10
        }
        
        grade = 0
        for metric, weight in weights.items():
            if metric in key_metrics:
                grade += key_metrics[metric] * weight
        
        key_metrics['overall_grade'] = float(grade)
        
        return key_metrics


# ============================================================================
# MODEL EVALUATOR CLASS
# ============================================================================
class ModelEvaluator:
    """
    Main model evaluator class for fraud detection systems.
    
    This class orchestrates the entire evaluation process:
    - Loads trained models
    - Prepares test data
    - Runs comprehensive evaluation
    - Generates reports and visualizations
    - Saves results for deployment decisions
    """
    
    def __init__(self, config_path: str = 'configs/model_config.yaml'):
        """
        Initialize the model evaluator.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load_config()
        
        # Setup paths
        self.models_dir = Path('artifacts/models')
        self.reports_dir = Path('artifacts/reports')
        self.visualizations_dir = Path('artifacts/visualizations')
        
        # Create directories if they don't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize containers
        self.models = {}
        self.test_data = None
        self.results = {}
        
        logger.info("ModelEvaluator initialized successfully")
    
    def load_models(self, model_names: Optional[List[str]] = None):
        """
        Load trained models from disk.
        
        Args:
            model_names: List of model names to load (None loads all)
        """
        logger.info("Loading trained models...")
        
        # Find all model files
        model_files = list(self.models_dir.glob('*.pkl')) + \
                     list(self.models_dir.glob('*.joblib')) + \
                     list(self.models_dir.glob('*.pt')) + \
                     list(self.models_dir.glob('*.pth'))
        
        if not model_files:
            logger.error(f"No model files found in {self.models_dir}")
            return
        
        for model_file in model_files:
            model_name = model_file.stem
            
            # Filter if specific models requested
            if model_names and model_name not in model_names:
                continue
            
            try:
                # Load based on file extension
                if model_file.suffix in ['.pkl', '.joblib']:
                    # Classical ML models
                    if model_file.suffix == '.pkl':
                        with open(model_file, 'rb') as f:
                            model = pickle.load(f)
                    else:
                        model = joblib.load(model_file)
                    
                    self.models[model_name] = {
                        'model': model,
                        'type': 'classical',
                        'path': model_file
                    }
                    
                elif model_file.suffix in ['.pt', '.pth']:
                    # PyTorch models
                    model = torch.load(model_file, map_location='cpu')
                    
                    # Try to determine model type
                    if hasattr(model, '__class__'):
                        model_type = 'pytorch'
                    else:
                        model_type = 'pytorch_state_dict'
                    
                    self.models[model_name] = {
                        'model': model,
                        'type': model_type,
                        'path': model_file
                    }
                
                logger.info(f"Loaded model: {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {str(e)}")
        
        logger.info(f"Successfully loaded {len(self.models)} models")
    
    def load_test_data(self, data_path: str):
        """
        Load test data for evaluation.
        
        Args:
            data_path: Path to test data file
        """
        logger.info(f"Loading test data from {data_path}")
        
        # Load based on file extension
        if data_path.endswith('.csv'):
            self.test_data = pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            self.test_data = pd.read_parquet(data_path)
        elif data_path.endswith('.feather'):
            self.test_data = pd.read_feather(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        logger.info(f"Test data loaded: {self.test_data.shape}")
        
        # Separate features and target
        if 'Class' in self.test_data.columns:
            self.X_test = self.test_data.drop('Class', axis=1)
            self.y_test = self.test_data['Class']
        elif 'is_fraud' in self.test_data.columns:
            self.X_test = self.test_data.drop('is_fraud', axis=1)
            self.y_test = self.test_data['is_fraud']
        else:
            logger.warning("No target column found. Assuming all columns are features.")
            self.X_test = self.test_data
            self.y_test = None
    
    def evaluate_all_models(self) -> Dict:
        """
        Evaluate all loaded models.
        
        Returns:
            Dictionary with evaluation results for each model
        """
        logger.info("Starting comprehensive model evaluation...")
        
        for model_name, model_info in self.models.items():
            logger.info(f"Evaluating model: {model_name}")
            
            try:
                # Get predictions based on model type
                if model_info['type'] in ['classical', 'pytorch']:
                    # Get probability predictions
                    if hasattr(model_info['model'], 'predict_proba'):
                        # Classical ML with predict_proba
                        y_proba = model_info['model'].predict_proba(self.X_test)
                        
                        # Handle binary classification
                        if y_proba.shape[1] == 2:
                            y_proba = y_proba[:, 1]
                        else:
                            y_proba = y_proba.ravel()
                        
                        y_pred = (y_proba >= 0.5).astype(int)
                        
                    elif hasattr(model_info['model'], 'predict'):
                        # Models without predict_proba
                        y_pred = model_info['model'].predict(self.X_test)
                        
                        # Try to get probabilities
                        if hasattr(model_info['model'], 'decision_function'):
                            y_proba = model_info['model'].decision_function(self.X_test)
                            # Normalize to [0, 1]
                            y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
                        else:
                            y_proba = y_pred
                    
                    else:
                        # PyTorch model
                        model_info['model'].eval()
                        with torch.no_grad():
                            X_tensor = torch.FloatTensor(self.X_test.values)
                            outputs = model_info['model'](X_tensor)
                            y_proba = torch.softmax(outputs, dim=1)[:, 1].numpy()
                            y_pred = (y_proba >= 0.5).astype(int)
                    
                    # Calculate metrics
                    calculator = EvaluationMetricsCalculator(self.y_test, y_pred, y_proba)
                    results = calculator.calculate_all_metrics()
                    
                    # Store results
                    self.results[model_name] = {
                        'metrics': results,
                        'predictions': {
                            'y_pred': y_pred.tolist(),
                            'y_proba': y_proba.tolist()
                        }
                    }
                    
                    # Generate visualizations for this model
                    self._generate_model_visualizations(model_name, results, y_pred, y_proba)
                    
                else:
                    logger.warning(f"Unsupported model type for {model_name}")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Compare models
        self._compare_models()
        
        # Generate comprehensive report
        self._generate_report()
        
        return self.results
    
    def _generate_model_visualizations(self, model_name: str, results: Dict, 
                                      y_pred: np.ndarray, y_proba: np.ndarray):
        """
        Generate comprehensive visualizations for model evaluation.
        
        Args:
            model_name: Name of the model
            results: Evaluation results
            y_pred: Predicted labels
            y_proba: Prediction probabilities
        """
        logger.info(f"Generating visualizations for {model_name}")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(f'Comprehensive Evaluation - {model_name}', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        ax1 = plt.subplot(3, 3, 1)
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['Non-Fraud', 'Fraud'],
                   yticklabels=['Non-Fraud', 'Fraud'])
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # 2. ROC Curve
        ax2 = plt.subplot(3, 3, 2)
        fpr = results['curves']['fpr']
        tpr = results['curves']['tpr']
        roc_auc = results['probabilistic_metrics']['roc_auc']
        
        ax2.plot(fpr, tpr, 'b-', label=f'ROC (AUC = {roc_auc:.4f})')
        ax2.plot([0, 1], [0, 1], 'r--', label='Random')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        ax3 = plt.subplot(3, 3, 3)
        precision = results['curves']['precision_curve']
        recall = results['curves']['recall_curve']
        pr_auc = results['probabilistic_metrics']['pr_auc']
        
        ax3.plot(recall, precision, 'g-', label=f'PR (AUC = {pr_auc:.4f})')
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall Curve')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Probability Distribution
        ax4 = plt.subplot(3, 3, 4)
        fraud_probs = y_proba[self.y_test == 1]
        non_fraud_probs = y_proba[self.y_test == 0]
        
        ax4.hist(non_fraud_probs, bins=50, alpha=0.7, label='Non-Fraud', density=True)
        ax4.hist(fraud_probs, bins=50, alpha=0.7, label='Fraud', density=True)
        ax4.axvline(x=0.5, color='red', linestyle='--', label='Default Threshold')
        ax4.set_xlabel('Fraud Probability')
        ax4.set_ylabel('Density')
        ax4.set_title('Probability Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Calibration Curve
        ax5 = plt.subplot(3, 3, 5)
        if 'calibration_metrics' in results:
            cal = results['calibration_metrics']['calibration_curve']
            ax5.plot(cal['mean_predicted'], cal['fraction_positives'], 'mo-', label='Model')
            ax5.plot([0, 1], [0, 1], 'k--', label='Perfect')
            ax5.set_xlabel('Mean Predicted Probability')
            ax5.set_ylabel('Fraction of Positives')
            ax5.set_title(f"Calibration Curve (ECE={results['calibration_metrics']['ece']:.4f})")
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Precision, Recall, F1 by Threshold
        ax6 = plt.subplot(3, 3, 6)
        thresholds = np.linspace(0, 1, 100)
        precisions = []
        recalls = []
        f1_scores = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_proba >= threshold).astype(int)
            precisions.append(precision_score(self.y_test, y_pred_thresh, zero_division=0))
            recalls.append(recall_score(self.y_test, y_pred_thresh, zero_division=0))
            f1_scores.append(f1_score(self.y_test, y_pred_thresh, zero_division=0))
        
        ax6.plot(thresholds, precisions, 'b-', label='Precision')
        ax6.plot(thresholds, recalls, 'g-', label='Recall')
        ax6.plot(thresholds, f1_scores, 'r-', label='F1')
        ax6.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
        ax6.set_xlabel('Threshold')
        ax6.set_ylabel('Score')
        ax6.set_title('Metrics by Threshold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Cumulative Gains Chart
        ax7 = plt.subplot(3, 3, 7)
        sorted_indices = np.argsort(y_proba)[::-1]
        y_true_sorted = self.y_test.iloc[sorted_indices].values
        
        cumulative_gains = np.cumsum(y_true_sorted) / np.sum(y_true_sorted)
        
        ax7.plot(cumulative_gains, 'b-', label='Model')
        ax7.plot(np.linspace(0, 1, len(cumulative_gains)), 'r--', label='Random')
        ax7.set_xlabel('Sample Fraction')
        ax7.set_ylabel('Cumulative Gains')
        ax7.set_title('Cumulative Gains Chart')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Lift Chart
        ax8 = plt.subplot(3, 3, 8)
        lift = cumulative_gains / (np.arange(len(cumulative_gains)) + 1) * len(self.y_test)
        
        ax8.plot(lift, 'b-')
        ax8.set_xlabel('Sample Fraction')
        ax8.set_ylabel('Lift')
        ax8.set_title('Lift Chart')
        ax8.grid(True, alpha=0.3)
        
        # 9. Feature Importance (if available)
        ax9 = plt.subplot(3, 3, 9)
        
        # Try to get feature importance from model
        model_info = self.models[model_name]
        model = model_info['model']
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
            feature_names = self.X_test.columns
            
            # Sort by importance
            indices = np.argsort(importances)[-10:]  # Top 10
            
            ax9.barh(range(len(indices)), importances[indices])
            ax9.set_yticks(range(len(indices)))
            ax9.set_yticklabels([feature_names[i] for i in indices])
            ax9.set_xlabel('Feature Importance')
            ax9.set_title('Top 10 Feature Importances')
            ax9.grid(True, alpha=0.3)
        else:
            ax9.text(0.5, 0.5, 'No Feature Importance Available',
                    ha='center', va='center', transform=ax9.transAxes)
            ax9.set_title('Feature Importance')
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.visualizations_dir / f'{model_name}_evaluation.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {save_path}")
    
    def _compare_models(self):
        """
        Compare all evaluated models and generate comparison visualizations.
        """
        if len(self.results) < 2:
            logger.info("Not enough models for comparison")
            return
        
        logger.info("Generating model comparison...")
        
        # Extract key metrics for comparison
        comparison_data = []
        
        for model_name, result in self.results.items():
            metrics = result['metrics']
            
            comparison_data.append({
                'model': model_name,
                'accuracy': metrics['derived_metrics']['accuracy'],
                'precision': metrics['derived_metrics']['precision'],
                'recall': metrics['derived_metrics']['recall'],
                'f1_score': metrics['derived_metrics']['f1_score'],
                'roc_auc': metrics['probabilistic_metrics']['roc_auc'],
                'pr_auc': metrics['probabilistic_metrics']['pr_auc'],
                'log_loss': metrics['probabilistic_metrics']['log_loss']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create comparison visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics_to_plot = ['f1_score', 'recall', 'precision', 'roc_auc', 'pr_auc', 'accuracy']
        positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        
        for metric, pos in zip(metrics_to_plot, positions):
            ax = axes[pos[0], pos[1]]
            
            # Sort by metric value
            sorted_df = comparison_df.sort_values(metric, ascending=True)
            
            bars = ax.barh(sorted_df['model'], sorted_df[metric])
            
            # Color bars based on value
            for bar, val in zip(bars, sorted_df[metric]):
                if val >= 0.9:
                    bar.set_color('darkgreen')
                elif val >= 0.8:
                    bar.set_color('lightgreen')
                elif val >= 0.7:
                    bar.set_color('yellow')
                elif val >= 0.6:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
            
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        # Save comparison
        save_path = self.visualizations_dir / 'model_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save comparison data
        comparison_df.to_csv(self.reports_dir / 'model_comparison.csv', index=False)
        
        # Identify best model
        best_model = comparison_df.loc[comparison_df['f1_score'].idxmax()]
        self.best_model_name = best_model['model']
        self.best_model_score = best_model['f1_score']
        
        logger.info(f"Best model: {self.best_model_name} (F1 = {self.best_model_score:.4f})")
    
    def _generate_report(self):
        """
        Generate comprehensive evaluation report.
        """
        logger.info("Generating evaluation report...")
        
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'config': self.config,
            'test_data_info': {
                'shape': self.test_data.shape if self.test_data is not None else None,
                'columns': list(self.test_data.columns) if self.test_data is not None else None
            },
            'models_evaluated': len(self.results),
            'results': {}
        }
        
        # Add detailed results for each model
        for model_name, result in self.results.items():
            report['results'][model_name] = {
                'metrics': result['metrics'],
                'summary': result['metrics']['summary']
            }
        
        # Add comparison if multiple models
        if len(self.results) > 1:
            comparison = {}
            for model_name, result in self.results.items():
                comparison[model_name] = result['metrics']['summary']
            
            report['comparison'] = comparison
            report['best_model'] = {
                'name': self.best_model_name,
                'f1_score': self.best_model_score
            }
        
        # Save report as JSON
        report_path = self.reports_dir / f'evaluation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save as markdown for readability
        self._save_markdown_report(report)
        
        logger.info(f"Report saved to {report_path}")
    
    def _save_markdown_report(self, report: Dict):
        """
        Save evaluation report as markdown for easy reading.
        
        Args:
            report: Report dictionary
        """
        markdown_path = self.reports_dir / f'evaluation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
        
        with open(markdown_path, 'w') as f:
            f.write("# VeritasFinancial Model Evaluation Report\n\n")
            f.write(f"**Generated:** {report['evaluation_timestamp']}\n\n")
            
            f.write("## Test Data Information\n\n")
            if report['test_data_info']['shape']:
                f.write(f"- **Shape:** {report['test_data_info']['shape']}\n")
                f.write(f"- **Columns:** {', '.join(report['test_data_info']['columns'][:10])}\n")
            
            f.write(f"\n## Models Evaluated: {report['models_evaluated']}\n\n")
            
            for model_name, model_results in report['results'].items():
                f.write(f"### {model_name}\n\n")
                
                summary = model_results['summary']
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                for metric, value in summary.items():
                    if isinstance(value, float):
                        f.write(f"| {metric} | {value:.4f} |\n")
                    else:
                        f.write(f"| {metric} | {value} |\n")
                f.write("\n")
            
            if 'comparison' in report:
                f.write("## Model Comparison\n\n")
                
                # Create comparison table
                f.write("| Model | " + " | ".join(report['comparison'][list(report['comparison'].keys())[0]].keys()) + " |\n")
                f.write("|-------|" + "|".join(["---"] * len(report['comparison'][list(report['comparison'].keys())[0]])) + "|\n")
                
                for model_name, metrics in report['comparison'].items():
                    f.write(f"| {model_name} | ")
                    f.write(" | ".join([f"{v:.4f}" if isinstance(v, float) else str(v) for v in metrics.values()]))
                    f.write(" |\n")
                
                f.write(f"\n**Best Model:** {report['best_model']['name']} (F1 = {report['best_model']['f1_score']:.4f})\n")
        
        logger.info(f"Markdown report saved to {markdown_path}")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================
def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='VeritasFinancial Model Evaluation Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_model.py --data data/processed/test.csv --models xgboost lightgbm
  python evaluate_model.py --data data/processed/test.parquet --all
  python evaluate_model.py --data data/processed/test.csv --config configs/custom_config.yaml
        """
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to test data file (CSV, Parquet, or Feather)'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        help='Specific models to evaluate (e.g., xgboost lightgbm)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Evaluate all available models'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/model_config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='artifacts',
        help='Directory to save evaluation results'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        default=True,
        help='Generate visualizations'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed output'
    )
    
    return parser.parse_args()


def main():
    """
    Main execution function.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Configure logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info("=" * 60)
    logger.info("VeritasFinancial Model Evaluation")
    logger.info("=" * 60)
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(config_path=args.config)
        
        # Load test data
        evaluator.load_test_data(args.data)
        
        # Load models
        if args.all:
            evaluator.load_models()
        elif args.models:
            evaluator.load_models(args.models)
        else:
            logger.error("Please specify --models or --all")
            sys.exit(1)
        
        # Evaluate models
        results = evaluator.evaluate_all_models()
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        
        for model_name, result in results.items():
            summary = result['metrics']['summary']
            logger.info(f"\n{model_name}:")
            logger.info(f"  F1 Score: {summary['f1']:.4f}")
            logger.info(f"  Precision: {summary['precision']:.4f}")
            logger.info(f"  Recall: {summary['recall']:.4f}")
            logger.info(f"  ROC-AUC: {summary['roc_auc']:.4f}")
            logger.info(f"  PR-AUC: {summary['pr_auc']:.4f}")
        
        if hasattr(evaluator, 'best_model_name'):
            logger.info("\n" + "=" * 60)
            logger.info(f"BEST MODEL: {evaluator.best_model_name}")
            logger.info(f"Best F1 Score: {evaluator.best_model_score:.4f}")
            logger.info("=" * 60)
        
        logger.info(f"\nReports saved to: {evaluator.reports_dir}")
        logger.info(f"Visualizations saved to: {evaluator.visualizations_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()