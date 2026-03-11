# =============================================================================
# Module: business_metrics.py
# Location: src/modeling/evaluation/business_metrics.py
# Purpose: Calculate business-centric metrics for fraud detection model evaluation
# Author: VeritasFinancial Data Science Team
# Version: 2.0.0
# Last Updated: 2024-01-15
# 
# Description:
# This module provides comprehensive business metrics calculation for fraud
# detection models. Unlike traditional ML metrics (accuracy, precision, recall),
# these metrics focus on the actual financial impact and operational efficiency
# of the fraud detection system. These metrics are crucial for:
#   1. ROI analysis of the fraud detection system
#   2. Setting optimal thresholds based on business constraints
#   3. Communicating model performance to business stakeholders
#   4. Making data-driven decisions about model deployment
# =============================================================================

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum
import json
import warnings
from pathlib import Path

# Configure logging for production monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class FraudType(Enum):
    """Enumeration of different fraud types for granular analysis"""
    CARD_NOT_PRESENT = "card_not_present"
    ACCOUNT_TAKEOVER = "account_takeover"
    IDENTITY_THEFT = "identity_theft"
    SYNTHETIC_ID = "synthetic_id"
    FIRST_PARTY = "first_party"
    TRIANGULATION = "triangulation"
    REFUND = "refund"
    CHARGEBACK = "chargeback"
    UNKNOWN = "unknown"


class DecisionAction(Enum):
    """Enumeration of possible decision actions"""
    APPROVE = "approve"
    REVIEW = "review"
    BLOCK = "block"
    STEP_UP_AUTH = "step_up_auth"
    MANUAL_REVIEW = "manual_review"


# =============================================================================
# DATA CLASSES FOR STRUCTURED CONFIGURATION
# =============================================================================

@dataclass
class BusinessCostConfig:
    """
    Configuration class for business costs associated with fraud detection.
    
    This class centralizes all cost-related parameters that affect business
    metrics. Each cost should be based on actual business data and regularly
    updated.
    
    Attributes:
        avg_fraud_loss (float): Average monetary loss per fraudulent transaction
        investigation_cost (float): Cost to investigate a suspicious transaction
        customer_acquisition_cost (float): Cost to acquire a new customer
        customer_lifetime_value (float): Average CLV for context
        false_positive_soft_cost (float): Soft cost of customer friction
        chargeback_processing_cost (float): Cost to process a chargeback
        manual_review_hourly_rate (float): Analyst hourly rate for reviews
        time_value_of_money_rate (float): Daily discount rate for time value
    """
    avg_fraud_loss: float = 500.0  # Average loss per fraud incident in USD
    investigation_cost: float = 25.0  # Cost to investigate one transaction
    customer_acquisition_cost: float = 200.0  # CAC in USD
    customer_lifetime_value: float = 5000.0  # CLV in USD
    false_positive_soft_cost: float = 5.0  # Customer friction cost
    chargeback_processing_cost: float = 50.0  # Cost to process chargeback
    manual_review_hourly_rate: float = 45.0  # Analyst hourly rate
    time_value_of_money_rate: float = 0.0001  # Daily discount rate
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization"""
        return {
            'avg_fraud_loss': self.avg_fraud_loss,
            'investigation_cost': self.investigation_cost,
            'customer_acquisition_cost': self.customer_acquisition_cost,
            'customer_lifetime_value': self.customer_lifetime_value,
            'false_positive_soft_cost': self.false_positive_soft_cost,
            'chargeback_processing_cost': self.chargeback_processing_cost,
            'manual_review_hourly_rate': self.manual_review_hourly_rate,
            'time_value_of_money_rate': self.time_value_of_money_rate
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BusinessCostConfig':
        """Create config from dictionary"""
        return cls(**{k: v for k, v in data.items() 
                     if k in cls.__dataclass_fields__})


@dataclass
class BusinessThresholdConfig:
    """
    Configuration for business-driven thresholds.
    
    These thresholds help translate model scores into business actions
    based on risk tolerance and operational capacity.
    
    Attributes:
        low_risk_threshold (float): Below this, auto-approve
        medium_risk_threshold (float): Between low and high, step-up auth
        high_risk_threshold (float): Above this, block transaction
        max_daily_reviews (int): Maximum reviews per day capacity
        max_review_cost_per_day (float): Budget for manual reviews
        target_precision (float): Desired precision for fraud detection
        target_recall (float): Desired recall for fraud detection
        max_false_positive_rate (float): Maximum acceptable FPR
    """
    low_risk_threshold: float = 0.3
    medium_risk_threshold: float = 0.7
    high_risk_threshold: float = 0.9
    max_daily_reviews: int = 1000
    max_review_cost_per_day: float = 50000.0
    target_precision: float = 0.8
    target_recall: float = 0.9
    max_false_positive_rate: float = 0.05
    
    def validate(self) -> bool:
        """Validate threshold consistency"""
        if not (0 <= self.low_risk_threshold <= 1):
            raise ValueError("low_risk_threshold must be between 0 and 1")
        if not (0 <= self.medium_risk_threshold <= 1):
            raise ValueError("medium_risk_threshold must be between 0 and 1")
        if not (0 <= self.high_risk_threshold <= 1):
            raise ValueError("high_risk_threshold must be between 0 and 1")
        if self.low_risk_threshold >= self.medium_risk_threshold:
            raise ValueError("low_risk_threshold must be less than medium_risk_threshold")
        if self.medium_risk_threshold >= self.high_risk_threshold:
            raise ValueError("medium_risk_threshold must be less than high_risk_threshold")
        return True


# =============================================================================
# MAIN BUSINESS METRICS CALCULATOR CLASS
# =============================================================================

class BusinessMetricsCalculator:
    """
    Comprehensive calculator for business metrics in fraud detection.
    
    This class provides methods to calculate various business-centric metrics
    that go beyond traditional ML metrics. It helps bridge the gap between
    model performance and business impact.
    
    The calculator handles:
    1. Financial metrics (cost savings, ROI)
    2. Operational metrics (review capacity, efficiency)
    3. Customer impact metrics (friction, satisfaction)
    4. Risk-based metrics (loss prevention, exposure)
    
    Args:
        cost_config (BusinessCostConfig): Configuration for business costs
        threshold_config (BusinessThresholdConfig): Configuration for thresholds
        currency (str): Currency code for monetary values
        business_unit (str): Identifier for business unit/segment
    """
    
    def __init__(
        self,
        cost_config: Optional[BusinessCostConfig] = None,
        threshold_config: Optional[BusinessThresholdConfig] = None,
        currency: str = 'USD',
        business_unit: str = 'default'
    ):
        """
        Initialize the BusinessMetricsCalculator with configurations.
        
        Args:
            cost_config: Business cost configuration (uses defaults if None)
            threshold_config: Threshold configuration (uses defaults if None)
            currency: Currency code for monetary values
            business_unit: Identifier for business unit/segment
        """
        self.cost_config = cost_config or BusinessCostConfig()
        self.threshold_config = threshold_config or BusinessThresholdConfig()
        self.currency = currency
        self.business_unit = business_unit
        self.metrics_history: List[Dict] = []
        self.last_calculation_time: Optional[datetime] = None
        
        # Validate thresholds
        self.threshold_config.validate()
        
        logger.info(f"Initialized BusinessMetricsCalculator for {business_unit}")
        logger.info(f"Cost config: {self.cost_config}")
        logger.info(f"Threshold config: {self.threshold_config}")
    
    # =========================================================================
    # SECTION 1: FINANCIAL IMPACT METRICS
    # =========================================================================
    
    def calculate_financial_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        transaction_amounts: np.ndarray,
        fraud_types: Optional[List[FraudType]] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive financial impact metrics.
        
        This method computes the actual financial impact of the fraud detection
        system by considering:
        - Fraud losses prevented
        - Operational costs
        - Customer impact costs
        - Net savings/profit
        
        Args:
            y_true: Ground truth labels (1 for fraud, 0 for legitimate)
            y_pred: Model predictions (1 for predicted fraud, 0 for legitimate)
            transaction_amounts: Monetary amounts of each transaction
            fraud_types: Optional list of fraud types for granular analysis
            
        Returns:
            Dictionary containing all financial metrics
            
        Raises:
            ValueError: If input arrays have incompatible shapes
        """
        # Validate inputs
        self._validate_inputs(y_true, y_pred, transaction_amounts)
        
        # Convert to numpy arrays if needed
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        amounts = np.array(transaction_amounts)
        
        # Calculate confusion matrix elements with amounts
        true_positives_mask = (y_true == 1) & (y_pred == 1)
        false_negatives_mask = (y_true == 1) & (y_pred == 0)
        false_positives_mask = (y_true == 0) & (y_pred == 1)
        true_negatives_mask = (y_true == 0) & (y_pred == 0)
        
        # Amount-based calculations
        tp_amount = np.sum(amounts[true_positives_mask])
        fn_amount = np.sum(amounts[false_negatives_mask])
        fp_amount = np.sum(amounts[false_positives_mask])
        tn_amount = np.sum(amounts[true_negatives_mask])
        
        # Count-based calculations
        tp_count = np.sum(true_positives_mask)
        fn_count = np.sum(false_negatives_mask)
        fp_count = np.sum(false_positives_mask)
        tn_count = np.sum(true_negatives_mask)
        
        # =====================================================================
        # 1. Fraud Loss Prevention Metrics
        # =====================================================================
        
        # Total fraud amount in the dataset
        total_fraud_amount = tp_amount + fn_amount
        
        # Prevented fraud amount (true positives)
        prevented_fraud_amount = tp_amount
        
        # Missed fraud amount (false negatives)
        missed_fraud_amount = fn_amount
        
        # Fraud prevention rate (by amount)
        fraud_prevention_rate = (
            prevented_fraud_amount / total_fraud_amount 
            if total_fraud_amount > 0 else 0.0
        )
        
        # =====================================================================
        # 2. Operational Cost Metrics
        # =====================================================================
        
        # Investigation costs for flagged transactions (TP + FP)
        investigation_cost_per_tx = self.cost_config.investigation_cost
        total_investigations = tp_count + fp_count
        total_investigation_cost = total_investigations * investigation_cost_per_tx
        
        # Manual review costs (if applicable)
        # Assuming some flagged transactions require manual review
        manual_review_rate = 0.3  # 30% of flagged need manual review
        manual_reviews = int(total_investigations * manual_review_rate)
        review_time_hours = manual_reviews * 0.5  # 30 minutes per review
        manual_review_cost = (
            review_time_hours * self.cost_config.manual_review_hourly_rate
        )
        
        # Chargeback processing costs
        # Chargebacks typically occur on false negatives (missed fraud)
        chargeback_cost = fn_count * self.cost_config.chargeback_processing_cost
        
        # Total operational costs
        total_operational_cost = (
            total_investigation_cost + 
            manual_review_cost + 
            chargeback_cost
        )
        
        # =====================================================================
        # 3. Customer Impact Metrics
        # =====================================================================
        
        # False positive customer friction cost
        # Each false positive causes customer friction and potential churn
        fp_customer_impact_cost = (
            fp_count * self.cost_config.false_positive_soft_cost
        )
        
        # Customer churn risk from false positives
        # Assuming X% of false positives lead to customer churn
        fp_churn_rate = 0.05  # 5% of false positives churn
        estimated_churned_customers = fp_count * fp_churn_rate
        churn_cost = (
            estimated_churned_customers * self.cost_config.customer_acquisition_cost
        )
        
        # Customer trust impact (hard to quantify, but important)
        # This is a proxy metric for customer satisfaction impact
        customer_trust_score = max(0, 1 - (fp_count / max(tn_count, 1)))
        
        # =====================================================================
        # 4. Net Financial Impact
        # =====================================================================
        
        # Gross savings from prevented fraud
        gross_savings = prevented_fraud_amount
        
        # Total costs (operational + customer impact)
        total_costs = total_operational_cost + fp_customer_impact_cost + churn_cost
        
        # Net savings/profit
        net_savings = gross_savings - total_costs
        
        # Return on Investment (ROI)
        roi = (net_savings / total_costs * 100) if total_costs > 0 else float('inf')
        
        # Savings per transaction
        total_transactions = len(y_true)
        savings_per_tx = net_savings / total_transactions if total_transactions > 0 else 0.0
        
        # =====================================================================
        # 5. Compile all financial metrics
        # =====================================================================
        
        financial_metrics = {
            # Fraud loss metrics
            'total_fraud_amount': float(total_fraud_amount),
            'prevented_fraud_amount': float(prevented_fraud_amount),
            'missed_fraud_amount': float(missed_fraud_amount),
            'fraud_prevention_rate': float(fraud_prevention_rate),
            
            # Operational metrics
            'total_investigations': int(total_investigations),
            'total_investigation_cost': float(total_investigation_cost),
            'manual_review_cost': float(manual_review_cost),
            'chargeback_cost': float(chargeback_cost),
            'total_operational_cost': float(total_operational_cost),
            
            # Customer impact metrics
            'fp_customer_impact_cost': float(fp_customer_impact_cost),
            'estimated_churned_customers': float(estimated_churned_customers),
            'churn_cost': float(churn_cost),
            'customer_trust_score': float(customer_trust_score),
            
            # Net financial metrics
            'gross_savings': float(gross_savings),
            'total_costs': float(total_costs),
            'net_savings': float(net_savings),
            'roi_percentage': float(roi),
            'savings_per_transaction': float(savings_per_tx),
            
            # Metadata
            'currency': self.currency,
            'business_unit': self.business_unit,
            'calculation_timestamp': datetime.now().isoformat()
        }
        
        # Log summary for monitoring
        logger.info(f"Financial metrics calculated: Net savings = {net_savings:.2f} {self.currency}")
        logger.info(f"ROI = {roi:.2f}%")
        
        return financial_metrics
    
    # =========================================================================
    # SECTION 2: OPERATIONAL METRICS
    # =========================================================================
    
    def calculate_operational_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        review_times: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate operational efficiency metrics.
        
        These metrics focus on the operational aspects of fraud detection:
        - Review queue management
        - Analyst workload
        - Response times
        - Capacity utilization
        
        Args:
            y_true: Ground truth labels
            y_pred: Model predictions
            timestamps: Optional timestamps for temporal analysis
            review_times: Optional review completion times
            
        Returns:
            Dictionary containing operational metrics
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Basic counts
        total_transactions = len(y_true)
        flagged_transactions = np.sum(y_pred == 1)
        fraud_transactions = np.sum(y_true == 1)
        detected_fraud = np.sum((y_true == 1) & (y_pred == 1))
        
        # =====================================================================
        # 1. Workload Metrics
        # =====================================================================
        
        # Flag rate (percentage of transactions flagged for review)
        flag_rate = flagged_transactions / total_transactions if total_transactions > 0 else 0.0
        
        # Review rate relative to capacity
        if self.threshold_config.max_daily_reviews > 0:
            capacity_utilization = flagged_transactions / self.threshold_config.max_daily_reviews
        else:
            capacity_utilization = float('inf')
        
        # Fraud hit rate (percentage of flagged that are actually fraud)
        fraud_hit_rate = detected_fraud / flagged_transactions if flagged_transactions > 0 else 0.0
        
        # =====================================================================
        # 2. Temporal Metrics (if timestamps provided)
        # =====================================================================
        
        temporal_metrics = {}
        if timestamps is not None:
            timestamps = pd.to_datetime(timestamps)
            
            # Transactions per hour
            df_temp = pd.DataFrame({
                'timestamp': timestamps,
                'flagged': y_pred
            })
            df_temp['hour'] = df_temp['timestamp'].dt.floor('H')
            hourly_flagged = df_temp.groupby('hour')['flagged'].sum()
            
            temporal_metrics.update({
                'avg_flagged_per_hour': float(hourly_flagged.mean()),
                'max_flagged_per_hour': float(hourly_flagged.max()),
                'min_flagged_per_hour': float(hourly_flagged.min()),
                'std_flagged_per_hour': float(hourly_flagged.std())
            })
        
        # =====================================================================
        # 3. Review Efficiency Metrics (if review times provided)
        # =====================================================================
        
        efficiency_metrics = {}
        if review_times is not None and len(review_times) > 0:
            review_times = np.array(review_times)
            
            efficiency_metrics.update({
                'avg_review_time_minutes': float(np.mean(review_times) / 60),
                'median_review_time_minutes': float(np.median(review_times) / 60),
                'p95_review_time_minutes': float(np.percentile(review_times, 95) / 60),
                'p99_review_time_minutes': float(np.percentile(review_times, 99) / 60),
                'total_review_hours': float(np.sum(review_times) / 3600)
            })
        
        # =====================================================================
        # 4. Compile all operational metrics
        # =====================================================================
        
        operational_metrics = {
            'total_transactions': int(total_transactions),
            'flagged_transactions': int(flagged_transactions),
            'fraud_transactions': int(fraud_transactions),
            'detected_fraud': int(detected_fraud),
            'flag_rate': float(flag_rate),
            'capacity_utilization': float(capacity_utilization),
            'fraud_hit_rate': float(fraud_hit_rate),
            **temporal_metrics,
            **efficiency_metrics
        }
        
        logger.info(f"Operational metrics calculated: Flag rate = {flag_rate:.2%}")
        
        return operational_metrics
    
    # =========================================================================
    # SECTION 3: RISK-BASED METRICS
    # =========================================================================
    
    def calculate_risk_metrics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        transaction_amounts: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate risk-based metrics for portfolio management.
        
        These metrics help understand the risk profile of transactions and
        the effectiveness of risk-based decisioning.
        
        Args:
            y_true: Ground truth labels
            y_pred_proba: Model probability scores (0 to 1)
            transaction_amounts: Transaction amounts
            
        Returns:
            Dictionary containing risk metrics
        """
        y_true = np.array(y_true)
        y_pred_proba = np.array(y_pred_proba)
        amounts = np.array(transaction_amounts)
        
        # =====================================================================
        # 1. Risk Distribution Metrics
        # =====================================================================
        
        # Risk score statistics for fraud vs non-fraud
        fraud_scores = y_pred_proba[y_true == 1]
        non_fraud_scores = y_pred_proba[y_true == 0]
        
        risk_metrics = {
            'avg_risk_score_fraud': float(np.mean(fraud_scores)) if len(fraud_scores) > 0 else 0.0,
            'avg_risk_score_non_fraud': float(np.mean(non_fraud_scores)) if len(non_fraud_scores) > 0 else 0.0,
            'median_risk_score_fraud': float(np.median(fraud_scores)) if len(fraud_scores) > 0 else 0.0,
            'median_risk_score_non_fraud': float(np.median(non_fraud_scores)) if len(non_fraud_scores) > 0 else 0.0,
            'p95_risk_score_fraud': float(np.percentile(fraud_scores, 95)) if len(fraud_scores) > 0 else 0.0,
            'p95_risk_score_non_fraud': float(np.percentile(non_fraud_scores, 95)) if len(non_fraud_scores) > 0 else 0.0,
        }
        
        # =====================================================================
        # 2. Exposure Metrics
        # =====================================================================
        
        # Total exposure (amount at risk)
        total_exposure = np.sum(amounts)
        
        # Exposure at different risk levels
        low_risk_mask = y_pred_proba < self.threshold_config.low_risk_threshold
        medium_risk_mask = (
            (y_pred_proba >= self.threshold_config.low_risk_threshold) & 
            (y_pred_proba < self.threshold_config.medium_risk_threshold)
        )
        high_risk_mask = y_pred_proba >= self.threshold_config.medium_risk_threshold
        
        exposure_metrics = {
            'total_exposure': float(total_exposure),
            'low_risk_exposure': float(np.sum(amounts[low_risk_mask])),
            'medium_risk_exposure': float(np.sum(amounts[medium_risk_mask])),
            'high_risk_exposure': float(np.sum(amounts[high_risk_mask])),
            'low_risk_percentage': float(np.sum(amounts[low_risk_mask]) / total_exposure * 100),
            'medium_risk_percentage': float(np.sum(amounts[medium_risk_mask]) / total_exposure * 100),
            'high_risk_percentage': float(np.sum(amounts[high_risk_mask]) / total_exposure * 100),
        }
        
        # =====================================================================
        # 3. Risk-Adjusted Metrics
        # =====================================================================
        
        # Expected loss (probability * amount)
        expected_loss = np.sum(y_pred_proba * amounts)
        
        # Value at Risk (VaR) at different confidence levels
        sorted_amounts = np.sort(amounts)
        var_95 = np.percentile(sorted_amounts, 95)
        var_99 = np.percentile(sorted_amounts, 99)
        
        # Conditional VaR (Expected Shortfall)
        cvar_95 = np.mean(sorted_amounts[sorted_amounts >= var_95])
        cvar_99 = np.mean(sorted_amounts[sorted_amounts >= var_99])
        
        risk_adjusted_metrics = {
            'expected_loss': float(expected_loss),
            'var_95': float(var_95),
            'var_99': float(var_99),
            'cvar_95': float(cvar_95),
            'cvar_99': float(cvar_99),
            'risk_adjusted_return': float((total_exposure - expected_loss) / total_exposure)
        }
        
        # Combine all risk metrics
        all_risk_metrics = {**risk_metrics, **exposure_metrics, **risk_adjusted_metrics}
        
        logger.info(f"Risk metrics calculated: Expected loss = {expected_loss:.2f} {self.currency}")
        
        return all_risk_metrics
    
    # =========================================================================
    # SECTION 4: THRESHOLD OPTIMIZATION METRICS
    # =========================================================================
    
    def calculate_threshold_metrics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        transaction_amounts: np.ndarray,
        thresholds: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Calculate metrics across different thresholds to find optimal cutoff.
        
        This method helps in selecting the optimal threshold based on
        business constraints and objectives.
        
        Args:
            y_true: Ground truth labels
            y_pred_proba: Model probability scores
            transaction_amounts: Transaction amounts
            thresholds: List of thresholds to evaluate (default: 0.1 to 0.9 step 0.05)
            
        Returns:
            DataFrame with metrics for each threshold
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 0.95, 0.05)
        
        results = []
        
        for threshold in thresholds:
            # Apply threshold
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # Calculate basic metrics
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            
            # Calculate financial impact at this threshold
            financial_metrics = self.calculate_financial_metrics(
                y_true, y_pred, transaction_amounts
            )
            
            # Calculate operational metrics
            operational_metrics = self.calculate_operational_metrics(
                y_true, y_pred
            )
            
            # Combine all metrics
            threshold_metrics = {
                'threshold': float(threshold),
                'true_positives': int(tp),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_negatives': int(tn),
                'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
                'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
                'f1_score': float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0,
                'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
                'net_savings': financial_metrics['net_savings'],
                'roi': financial_metrics['roi_percentage'],
                'flag_rate': operational_metrics['flag_rate'],
                'fraud_hit_rate': operational_metrics['fraud_hit_rate']
            }
            
            results.append(threshold_metrics)
        
        # Convert to DataFrame for easy analysis
        results_df = pd.DataFrame(results)
        
        # Find optimal threshold based on different criteria
        optimal_by_net_savings = results_df.loc[results_df['net_savings'].idxmax()]
        optimal_by_f1 = results_df.loc[results_df['f1_score'].idxmax()]
        optimal_by_roi = results_df.loc[results_df['roi'].idxmax()]
        
        logger.info(f"Threshold optimization complete")
        logger.info(f"Optimal by net savings: {optimal_by_net_savings['threshold']:.3f}")
        logger.info(f"Optimal by F1: {optimal_by_f1['threshold']:.3f}")
        logger.info(f"Optimal by ROI: {optimal_by_roi['threshold']:.3f}")
        
        return results_df
    
    # =========================================================================
    # SECTION 5: CUSTOMER SEGMENTATION METRICS
    # =========================================================================
    
    def calculate_segment_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        transaction_amounts: np.ndarray,
        customer_segments: np.ndarray,
        segment_names: Optional[Dict[int, str]] = None
    ) -> Dict[str, Dict]:
        """
        Calculate business metrics segmented by customer groups.
        
        Different customer segments may have different fraud patterns and
        business impacts. This method provides granular analysis.
        
        Args:
            y_true: Ground truth labels
            y_pred: Model predictions
            transaction_amounts: Transaction amounts
            customer_segments: Array of segment IDs for each transaction
            segment_names: Optional mapping from segment ID to name
            
        Returns:
            Dictionary with metrics for each segment
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        amounts = np.array(transaction_amounts)
        segments = np.array(customer_segments)
        
        unique_segments = np.unique(segments)
        segment_metrics = {}
        
        for segment in unique_segments:
            # Get mask for this segment
            segment_mask = segments == segment
            
            # Get segment name or use ID
            segment_name = (
                segment_names.get(segment, f"Segment_{segment}") 
                if segment_names else f"Segment_{segment}"
            )
            
            # Calculate metrics for this segment
            segment_y_true = y_true[segment_mask]
            segment_y_pred = y_pred[segment_mask]
            segment_amounts = amounts[segment_mask]
            
            # Skip if no transactions in this segment
            if len(segment_y_true) == 0:
                continue
            
            # Calculate financial metrics for segment
            financial = self.calculate_financial_metrics(
                segment_y_true, segment_y_pred, segment_amounts
            )
            
            # Calculate operational metrics
            operational = self.calculate_operational_metrics(
                segment_y_true, segment_y_pred
            )
            
            # Calculate segment-specific metrics
            segment_fraud_rate = np.mean(segment_y_true)
            segment_approval_rate = np.mean(segment_y_pred == 0)
            
            # Store segment metrics
            segment_metrics[segment_name] = {
                'transaction_count': int(len(segment_y_true)),
                'fraud_rate': float(segment_fraud_rate),
                'approval_rate': float(segment_approval_rate),
                'total_amount': float(np.sum(segment_amounts)),
                'avg_amount': float(np.mean(segment_amounts)),
                'financial_metrics': financial,
                'operational_metrics': operational
            }
        
        logger.info(f"Segment metrics calculated for {len(segment_metrics)} segments")
        
        return segment_metrics
    
    # =========================================================================
    # SECTION 6: TIME-BASED METRICS (DETECTION DELAY)
    # =========================================================================
    
    def calculate_time_based_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        transaction_times: np.ndarray,
        detection_times: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate time-based metrics focusing on detection speed.
        
        In fraud detection, faster detection leads to less loss. These metrics
        quantify the time efficiency of the detection system.
        
        Args:
            y_true: Ground truth labels
            y_pred: Model predictions
            transaction_times: Timestamps of transactions
            detection_times: Optional timestamps when fraud was detected
            
        Returns:
            Dictionary containing time-based metrics
        """
        transaction_times = pd.to_datetime(transaction_times)
        
        # Find fraud transactions that were detected
        fraud_mask = y_true == 1
        detected_fraud_mask = (y_true == 1) & (y_pred == 1)
        
        # =====================================================================
        # 1. Detection Delay Metrics
        # =====================================================================
        
        if detection_times is not None and np.any(detected_fraud_mask):
            detection_times = pd.to_datetime(detection_times)
            
            # Calculate detection delay for each detected fraud
            fraud_transaction_times = transaction_times[detected_fraud_mask]
            fraud_detection_times = detection_times[detected_fraud_mask]
            
            detection_delays = (
                fraud_detection_times - fraud_transaction_times
            ).total_seconds()
            
            time_metrics = {
                'avg_detection_delay_seconds': float(np.mean(detection_delays)),
                'median_detection_delay_seconds': float(np.median(detection_delays)),
                'min_detection_delay_seconds': float(np.min(detection_delays)),
                'max_detection_delay_seconds': float(np.max(detection_delays)),
                'p95_detection_delay_seconds': float(np.percentile(detection_delays, 95)),
                'p99_detection_delay_seconds': float(np.percentile(detection_delays, 99)),
            }
        else:
            time_metrics = {
                'avg_detection_delay_seconds': 0.0,
                'median_detection_delay_seconds': 0.0,
                'min_detection_delay_seconds': 0.0,
                'max_detection_delay_seconds': 0.0,
                'p95_detection_delay_seconds': 0.0,
                'p99_detection_delay_seconds': 0.0,
            }
        
        # =====================================================================
        # 2. Fraud Occurrence Patterns
        # =====================================================================
        
        # Time of day analysis
        hours = transaction_times[fraud_mask].hour
        if len(hours) > 0:
            time_metrics.update({
                'peak_fraud_hour': int(hours.mode()[0] if len(hours.mode()) > 0 else 0),
                'fraud_hours_std': float(hours.std()),
            })
        
        # Day of week analysis
        days = transaction_times[fraud_mask].dayofweek
        if len(days) > 0:
            time_metrics['weekend_fraud_ratio'] = float(
                np.mean(days >= 5)  # Saturday and Sunday
            )
        
        logger.info(f"Time-based metrics calculated")
        
        return time_metrics
    
    # =========================================================================
    # SECTION 7: COMPREHENSIVE BUSINESS REPORT
    # =========================================================================
    
    def generate_comprehensive_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        transaction_amounts: np.ndarray,
        transaction_times: Optional[np.ndarray] = None,
        customer_segments: Optional[np.ndarray] = None,
        fraud_types: Optional[List[FraudType]] = None,
        include_threshold_analysis: bool = True,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive business metrics report.
        
        This method combines all individual metric calculations into a single
        comprehensive report suitable for business stakeholders.
        
        Args:
            y_true: Ground truth labels
            y_pred: Model predictions
            y_pred_proba: Model probability scores
            transaction_amounts: Transaction amounts
            transaction_times: Optional transaction timestamps
            customer_segments: Optional customer segment labels
            fraud_types: Optional fraud type labels
            include_threshold_analysis: Whether to include threshold optimization
            output_file: Optional file path to save report
            
        Returns:
            Dictionary containing all business metrics
        """
        logger.info("Generating comprehensive business metrics report...")
        
        # Initialize report
        report = {
            'report_generated': datetime.now().isoformat(),
            'business_unit': self.business_unit,
            'currency': self.currency,
            'data_summary': self._generate_data_summary(
                y_true, transaction_amounts
            )
        }
        
        # =====================================================================
        # 1. Financial Impact Metrics
        # =====================================================================
        logger.info("Calculating financial metrics...")
        report['financial_metrics'] = self.calculate_financial_metrics(
            y_true, y_pred, transaction_amounts, fraud_types
        )
        
        # =====================================================================
        # 2. Operational Metrics
        # =====================================================================
        logger.info("Calculating operational metrics...")
        report['operational_metrics'] = self.calculate_operational_metrics(
            y_true, y_pred, transaction_times
        )
        
        # =====================================================================
        # 3. Risk Metrics
        # =====================================================================
        logger.info("Calculating risk metrics...")
        report['risk_metrics'] = self.calculate_risk_metrics(
            y_true, y_pred_proba, transaction_amounts
        )
        
        # =====================================================================
        # 4. Time-based Metrics (if timestamps provided)
        # =====================================================================
        if transaction_times is not None:
            logger.info("Calculating time-based metrics...")
            report['time_metrics'] = self.calculate_time_based_metrics(
                y_true, y_pred, transaction_times
            )
        
        # =====================================================================
        # 5. Segment Metrics (if segments provided)
        # =====================================================================
        if customer_segments is not None:
            logger.info("Calculating segment metrics...")
            report['segment_metrics'] = self.calculate_segment_metrics(
                y_true, y_pred, transaction_amounts, customer_segments
            )
        
        # =====================================================================
        # 6. Threshold Analysis (optional)
        # =====================================================================
        if include_threshold_analysis:
            logger.info("Performing threshold analysis...")
            threshold_df = self.calculate_threshold_metrics(
                y_true, y_pred_proba, transaction_amounts
            )
            
            # Find optimal thresholds
            optimal_by_savings = threshold_df.loc[threshold_df['net_savings'].idxmax()].to_dict()
            optimal_by_f1 = threshold_df.loc[threshold_df['f1_score'].idxmax()].to_dict()
            
            report['threshold_analysis'] = {
                'threshold_data': threshold_df.to_dict('records'),
                'optimal_by_net_savings': optimal_by_savings,
                'optimal_by_f1': optimal_by_f1,
                'recommended_threshold': optimal_by_savings['threshold'],
                'recommendation_reason': (
                    f"Threshold {optimal_by_savings['threshold']:.3f} "
                    f"maximizes net savings at {optimal_by_savings['net_savings']:.2f} {self.currency}"
                )
            }
        
        # =====================================================================
        # 7. Executive Summary
        # =====================================================================
        report['executive_summary'] = self._generate_executive_summary(report)
        
        # =====================================================================
        # 8. Save report if output file specified
        # =====================================================================
        if output_file:
            self._save_report(report, output_file)
            logger.info(f"Report saved to {output_file}")
        
        # Store in history
        self.metrics_history.append({
            'timestamp': datetime.now().isoformat(),
            'report_summary': report['executive_summary']
        })
        self.last_calculation_time = datetime.now()
        
        logger.info("Comprehensive report generation complete")
        
        return report
    
    # =========================================================================
    # SECTION 8: UTILITY METHODS
    # =========================================================================
    
    def _validate_inputs(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        amounts: np.ndarray
    ) -> None:
        """
        Validate input arrays for consistency.
        
        Args:
            y_true: Ground truth labels
            y_pred: Model predictions
            amounts: Transaction amounts
            
        Raises:
            ValueError: If inputs are invalid
        """
        if len(y_true) != len(y_pred) or len(y_true) != len(amounts):
            raise ValueError(
                f"Input length mismatch: y_true={len(y_true)}, "
                f"y_pred={len(y_pred)}, amounts={len(amounts)}"
            )
        
        if not np.all(np.isin(y_true, [0, 1])):
            raise ValueError("y_true must contain only 0 and 1")
        
        if not np.all(np.isin(y_pred, [0, 1])):
            raise ValueError("y_pred must contain only 0 and 1")
        
        if np.any(amounts < 0):
            raise ValueError("Transaction amounts cannot be negative")
    
    def _generate_data_summary(
        self,
        y_true: np.ndarray,
        amounts: np.ndarray
    ) -> Dict[str, Any]:
        """
        Generate summary statistics of the input data.
        
        Args:
            y_true: Ground truth labels
            amounts: Transaction amounts
            
        Returns:
            Dictionary with data summary
        """
        fraud_count = np.sum(y_true == 1)
        legitimate_count = np.sum(y_true == 0)
        total_count = len(y_true)
        
        fraud_amount = np.sum(amounts[y_true == 1])
        legitimate_amount = np.sum(amounts[y_true == 0])
        total_amount = np.sum(amounts)
        
        return {
            'total_transactions': int(total_count),
            'fraud_transactions': int(fraud_count),
            'legitimate_transactions': int(legitimate_count),
            'fraud_rate': float(fraud_count / total_count if total_count > 0 else 0),
            'total_amount': float(total_amount),
            'fraud_amount': float(fraud_amount),
            'legitimate_amount': float(legitimate_amount),
            'avg_transaction_amount': float(np.mean(amounts)),
            'median_transaction_amount': float(np.median(amounts)),
            'min_transaction_amount': float(np.min(amounts)),
            'max_transaction_amount': float(np.max(amounts)),
            'std_transaction_amount': float(np.std(amounts))
        }
    
    def _generate_executive_summary(self, report: Dict) -> Dict[str, Any]:
        """
        Generate an executive summary of key metrics.
        
        This summary is designed for business stakeholders who need
        high-level insights without technical details.
        
        Args:
            report: Complete metrics report
            
        Returns:
            Dictionary with key metrics and insights
        """
        financial = report.get('financial_metrics', {})
        operational = report.get('operational_metrics', {})
        risk = report.get('risk_metrics', {})
        
        # Calculate key performance indicators
        fraud_prevention_efficiency = (
            financial.get('prevented_fraud_amount', 0) / 
            financial.get('total_fraud_amount', 1) * 100
        )
        
        cost_per_prevented_fraud = (
            financial.get('total_costs', 0) / 
            max(financial.get('prevented_fraud_amount', 1), 1)
        )
        
        review_efficiency = (
            operational.get('detected_fraud', 0) / 
            max(operational.get('flagged_transactions', 1), 1) * 100
        )
        
        # Generate summary
        summary = {
            'key_metrics': {
                'total_savings': f"{financial.get('net_savings', 0):,.2f} {self.currency}",
                'roi': f"{financial.get('roi_percentage', 0):.1f}%",
                'fraud_prevention_rate': f"{fraud_prevention_efficiency:.1f}%",
                'review_efficiency': f"{review_efficiency:.1f}%",
                'cost_per_prevented_fraud': f"{cost_per_prevented_fraud:.2f} {self.currency}",
                'expected_loss': f"{risk.get('expected_loss', 0):,.2f} {self.currency}"
            },
            'performance_rating': self._calculate_performance_rating(financial, operational),
            'key_insights': self._generate_key_insights(report),
            'recommendations': self._generate_recommendations(report)
        }
        
        return summary
    
    def _calculate_performance_rating(
        self,
        financial: Dict,
        operational: Dict
    ) -> str:
        """
        Calculate overall performance rating based on multiple factors.
        
        Args:
            financial: Financial metrics
            operational: Operational metrics
            
        Returns:
            String rating (Excellent, Good, Fair, Poor)
        """
        # Define thresholds for different ratings
        if financial.get('roi_percentage', 0) > 200 and operational.get('fraud_hit_rate', 0) > 0.5:
            return "Excellent"
        elif financial.get('roi_percentage', 0) > 100 and operational.get('fraud_hit_rate', 0) > 0.3:
            return "Good"
        elif financial.get('roi_percentage', 0) > 0 and operational.get('fraud_hit_rate', 0) > 0.1:
            return "Fair"
        else:
            return "Poor"
    
    def _generate_key_insights(self, report: Dict) -> List[str]:
        """
        Generate key insights from the metrics.
        
        Args:
            report: Complete metrics report
            
        Returns:
            List of insight statements
        """
        insights = []
        financial = report.get('financial_metrics', {})
        operational = report.get('operational_metrics', {})
        risk = report.get('risk_metrics', {})
        
        # Fraud prevention insight
        prevention_rate = (
            financial.get('prevented_fraud_amount', 0) / 
            max(financial.get('total_fraud_amount', 1), 1) * 100
        )
        if prevention_rate > 80:
            insights.append(f"Excellent fraud prevention rate of {prevention_rate:.1f}%")
        elif prevention_rate > 50:
            insights.append(f"Good fraud prevention rate of {prevention_rate:.1f}%")
        else:
            insights.append(f"Fraud prevention rate of {prevention_rate:.1f}% needs improvement")
        
        # Operational efficiency insight
        hit_rate = operational.get('fraud_hit_rate', 0) * 100
        if hit_rate > 50:
            insights.append(f"High review efficiency: {hit_rate:.1f}% of flagged are fraud")
        elif hit_rate > 30:
            insights.append(f"Moderate review efficiency: {hit_rate:.1f}% of flagged are fraud")
        else:
            insights.append(f"Low review efficiency: Only {hit_rate:.1f}% of flagged are fraud")
        
        # Risk exposure insight
        high_risk_percentage = risk.get('high_risk_percentage', 0)
        if high_risk_percentage > 30:
            insights.append(f"High risk exposure: {high_risk_percentage:.1f}% of amount is high-risk")
        else:
            insights.append(f"Managed risk exposure: {high_risk_percentage:.1f}% high-risk")
        
        return insights
    
    def _generate_recommendations(self, report: Dict) -> List[str]:
        """
        Generate actionable recommendations based on metrics.
        
        Args:
            report: Complete metrics report
            
        Returns:
            List of recommendation statements
        """
        recommendations = []
        financial = report.get('financial_metrics', {})
        operational = report.get('operational_metrics', {})
        
        # Threshold recommendations
        if 'threshold_analysis' in report:
            optimal = report['threshold_analysis'].get('recommended_threshold')
            current = self.threshold_config.medium_risk_threshold
            
            if abs(optimal - current) > 0.1:
                recommendations.append(
                    f"Consider adjusting threshold from {current:.2f} to {optimal:.2f} "
                    f"to optimize net savings"
                )
        
        # Review capacity recommendations
        capacity_util = operational.get('capacity_utilization', 0)
        if capacity_util > 0.9:
            recommendations.append(
                "Review team at capacity. Consider increasing review resources "
                "or tightening thresholds to reduce volume"
            )
        elif capacity_util < 0.3:
            recommendations.append(
                "Review team underutilized. Consider loosening thresholds "
                "to catch more fraud"
            )
        
        # Fraud detection recommendations
        hit_rate = operational.get('fraud_hit_rate', 0)
        if hit_rate < 0.2:
            recommendations.append(
                "Low fraud hit rate. Investigate model performance and "
                "consider feature engineering improvements"
            )
        
        return recommendations
    
    def _save_report(self, report: Dict, output_file: str) -> None:
        """
        Save report to file in JSON format.
        
        Args:
            report: Report dictionary
            output_file: Output file path
        """
        # Create directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            else:
                return obj
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(report, f, default=convert_to_serializable, indent=2)
    
    def get_metrics_history(self) -> pd.DataFrame:
        """
        Get historical metrics as DataFrame.
        
        Returns:
            DataFrame with historical metrics
        """
        if not self.metrics_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.metrics_history)
    
    def compare_with_benchmark(
        self,
        current_metrics: Dict,
        benchmark_metrics: Dict
    ) -> Dict[str, Any]:
        """
        Compare current metrics with benchmark values.
        
        Args:
            current_metrics: Current period metrics
            benchmark_metrics: Benchmark metrics for comparison
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {}
        
        # Compare key financial metrics
        for metric in ['net_savings', 'roi_percentage', 'fraud_prevention_rate']:
            if metric in current_metrics and metric in benchmark_metrics:
                current = current_metrics[metric]
                benchmark = benchmark_metrics[metric]
                
                if benchmark != 0:
                    change_pct = (current - benchmark) / abs(benchmark) * 100
                else:
                    change_pct = float('inf') if current > 0 else 0
                
                comparison[f'{metric}_vs_benchmark'] = {
                    'current': current,
                    'benchmark': benchmark,
                    'absolute_change': current - benchmark,
                    'percentage_change': change_pct,
                    'status': 'improved' if current > benchmark else 'declined'
                }
        
        return comparison


# =============================================================================
# SECTION 9: USAGE EXAMPLE AND TESTING
# =============================================================================

def example_usage():
    """
    Example showing how to use the BusinessMetricsCalculator.
    """
    # Generate sample data
    np.random.seed(42)
    n_samples = 10000
    
    # Simulate ground truth (1% fraud rate)
    y_true = np.random.choice([0, 1], n_samples, p=[0.99, 0.01])
    
    # Simulate predictions with reasonable accuracy
    y_pred_proba = np.random.random(n_samples)
    # Make fraud cases have higher scores on average
    y_pred_proba[y_true == 1] += 0.3
    y_pred_proba = np.clip(y_pred_proba, 0, 1)
    
    # Apply default threshold
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Simulate transaction amounts (positive skew)
    transaction_amounts = np.random.gamma(2, 100, n_samples)
    
    # Simulate timestamps over 30 days
    base_time = datetime.now() - timedelta(days=30)
    timestamps = [
        base_time + timedelta(
            days=np.random.randint(0, 30),
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60)
        )
        for _ in range(n_samples)
    ]
    
    # Simulate customer segments
    customer_segments = np.random.choice([1, 2, 3, 4], n_samples)
    
    # Initialize calculator with custom configs
    cost_config = BusinessCostConfig(
        avg_fraud_loss=750.0,
        investigation_cost=30.0,
        customer_acquisition_cost=250.0
    )
    
    threshold_config = BusinessThresholdConfig(
        low_risk_threshold=0.3,
        medium_risk_threshold=0.7,
        high_risk_threshold=0.9
    )
    
    calculator = BusinessMetricsCalculator(
        cost_config=cost_config,
        threshold_config=threshold_config,
        business_unit="retail_banking"
    )
    
    # Generate comprehensive report
    report = calculator.generate_comprehensive_report(
        y_true=y_true,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        transaction_amounts=transaction_amounts,
        transaction_times=timestamps,
        customer_segments=customer_segments,
        include_threshold_analysis=True,
        output_file="reports/business_metrics_report.json"
    )
    
    # Print executive summary
    print("\n" + "="*60)
    print("EXECUTIVE SUMMARY")
    print("="*60)
    
    summary = report['executive_summary']
    print(f"Performance Rating: {summary['performance_rating']}")
    print("\nKey Metrics:")
    for key, value in summary['key_metrics'].items():
        print(f"  {key}: {value}")
    
    print("\nKey Insights:")
    for insight in summary['key_insights']:
        print(f"  • {insight}")
    
    print("\nRecommendations:")
    for rec in summary['recommendations']:
        print(f"  • {rec}")
    
    # Show threshold optimization results
    if 'threshold_analysis' in report:
        print("\n" + "="*60)
        print("THRESHOLD OPTIMIZATION")
        print("="*60)
        optimal = report['threshold_analysis']['optimal_by_net_savings']
        print(f"Optimal threshold: {optimal['threshold']:.3f}")
        print(f"Net savings at optimal: {optimal['net_savings']:.2f} USD")
        print(f"F1 score at optimal: {optimal['f1_score']:.3f}")
        print(f"Flag rate at optimal: {optimal['flag_rate']:.2%}")
    
    return report


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    When run directly, this module will execute the example usage
    to demonstrate its functionality.
    """
    print("Running BusinessMetricsCalculator example...")
    example_usage()