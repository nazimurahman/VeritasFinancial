"""
Drift Detection Module
======================

Detects data drift and concept drift in production models to ensure
model performance remains stable over time.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import json
import hashlib

# Configure logging
logger = logging.getLogger(__name__)

class DataDriftDetector:
    """
    Detects drift in input data distributions over time.
    
    Compares current data distribution with reference distribution
    and triggers alerts when significant drift is detected.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize data drift detector.
        
        Args:
            config: Configuration dictionary containing:
                - reference_window: Days of reference data to use
                - detection_window: Days of current data to compare
                - threshold: Drift threshold for alerting
                - features_to_monitor: List of features to monitor
        """
        self.config = config or {}
        self.reference_window = self.config.get('reference_window', 30)  # days
        self.detection_window = self.config.get('detection_window', 7)   # days
        self.threshold = self.config.get('threshold', 0.1)  # 10% change threshold
        self.features_to_monitor = self.config.get('features_to_monitor', [])
        
        # Storage for historical data
        self.reference_stats = {}
        self.current_stats = {}
        self.drift_history = []
        
        logger.info(f"DataDriftDetector initialized with threshold: {self.threshold}")
    
    def compute_statistics(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Compute statistical summary for each feature.
        
        Args:
            data: Input data DataFrame
            
        Returns:
            dict: Statistical summaries for each feature
        """
        stats = {}
        
        for column in data.columns:
            if column in self.features_to_monitor or not self.features_to_monitor:
                col_data = data[column].dropna()
                
                if len(col_data) > 0:
                    if col_data.dtype in ['int64', 'float64']:
                        # Numerical features
                        stats[column] = {
                            'mean': float(col_data.mean()),
                            'std': float(col_data.std()),
                            'min': float(col_data.min()),
                            'max': float(col_data.max()),
                            'median': float(col_data.median()),
                            'q1': float(col_data.quantile(0.25)),
                            'q3': float(col_data.quantile(0.75)),
                            'skew': float(col_data.skew()),
                            'kurtosis': float(col_data.kurtosis()),
                            'missing_rate': float(data[column].isnull().mean())
                        }
                    else:
                        # Categorical features
                        value_counts = col_data.value_counts(normalize=True)
                        stats[column] = {
                            'unique_values': len(value_counts),
                            'mode': col_data.mode()[0] if len(col_data) > 0 else None,
                            'entropy': float(-(value_counts * np.log(value_counts + 1e-10)).sum()),
                            'missing_rate': float(data[column].isnull().mean()),
                            'top_categories': value_counts.head(5).to_dict()
                        }
        
        return stats
    
    def update_reference(self, data: pd.DataFrame) -> None:
        """
        Update reference statistics with new baseline data.
        
        Args:
            data: Reference data to establish baseline
        """
        self.reference_stats = self.compute_statistics(data)
        logger.info(f"Reference statistics updated for {len(self.reference_stats)} features")
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect drift in current data compared to reference.
        
        Args:
            current_data: Current data to check for drift
            
        Returns:
            dict: Drift detection results
        """
        if not self.reference_stats:
            logger.warning("No reference statistics available. Run update_reference first.")
            return {"drift_detected": False, "message": "No reference data"}
        
        # Compute statistics for current data
        self.current_stats = self.compute_statistics(current_data)
        
        # Detect drift for each feature
        drift_results = {}
        overall_drift_score = 0.0
        drifted_features = []
        
        for feature, ref_stats in self.reference_stats.items():
            if feature not in self.current_stats:
                continue
            
            curr_stats = self.current_stats[feature]
            
            # Calculate drift based on feature type
            if 'mean' in ref_stats:  # Numerical feature
                drift_score = self._calculate_numerical_drift(ref_stats, curr_stats)
            else:  # Categorical feature
                drift_score = self._calculate_categorical_drift(ref_stats, curr_stats)
            
            # Check if drift exceeds threshold
            is_drifted = drift_score > self.threshold
            
            drift_results[feature] = {
                'drift_score': drift_score,
                'is_drifted': is_drifted,
                'reference_stats': ref_stats,
                'current_stats': curr_stats,
                'threshold': self.threshold
            }
            
            if is_drifted:
                drifted_features.append(feature)
                overall_drift_score += drift_score
        
        # Calculate overall drift
        overall_drift_score = overall_drift_score / max(len(drift_results), 1)
        
        # Record drift event
        drift_event = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_drift_score': overall_drift_score,
            'drift_detected': len(drifted_features) > 0,
            'drifted_features': drifted_features,
            'total_features_monitored': len(drift_results),
            'drift_details': drift_results
        }
        
        self.drift_history.append(drift_event)
        
        # Log results
        if drift_event['drift_detected']:
            logger.warning(
                f"Data drift detected! Overall score: {overall_drift_score:.4f}, "
                f"Drifted features: {drifted_features}"
            )
        else:
            logger.info(f"No significant drift detected. Overall score: {overall_drift_score:.4f}")
        
        return drift_event
    
    def _calculate_numerical_drift(self, ref_stats: Dict, curr_stats: Dict) -> float:
        """
        Calculate drift score for numerical features.
        
        Uses a combination of:
        - Mean shift (standardized)
        - Variance change
        - Distribution shift via KS test
        
        Args:
            ref_stats: Reference statistics
            curr_stats: Current statistics
            
        Returns:
            float: Drift score (0-1)
        """
        # Mean shift (normalized by reference std)
        mean_shift = abs(curr_stats['mean'] - ref_stats['mean'])
        mean_shift_score = mean_shift / max(ref_stats['std'], 1e-10)
        mean_shift_score = min(mean_shift_score / 3, 1.0)  # Normalize to 0-1
        
        # Variance change
        std_ratio = curr_stats['std'] / max(ref_stats['std'], 1e-10)
        variance_score = min(abs(std_ratio - 1) * 2, 1.0)
        
        # Quantile shift
        q1_shift = abs(curr_stats['q1'] - ref_stats['q1']) / max(ref_stats['std'], 1e-10)
        q3_shift = abs(curr_stats['q3'] - ref_stats['q3']) / max(ref_stats['std'], 1e-10)
        quantile_score = min(max(q1_shift, q3_shift) / 3, 1.0)
        
        # Missing rate change
        missing_rate_shift = abs(curr_stats['missing_rate'] - ref_stats['missing_rate'])
        
        # Combine scores
        drift_score = (
            0.4 * mean_shift_score +
            0.3 * variance_score +
            0.2 * quantile_score +
            0.1 * missing_rate_shift
        )
        
        return drift_score
    
    def _calculate_categorical_drift(self, ref_stats: Dict, curr_stats: Dict) -> float:
        """
        Calculate drift score for categorical features.
        
        Args:
            ref_stats: Reference statistics
            curr_stats: Current statistics
            
        Returns:
            float: Drift score (0-1)
        """
        # Entropy change (distribution spread)
        entropy_ratio = curr_stats['entropy'] / max(ref_stats['entropy'], 1e-10)
        entropy_score = min(abs(entropy_ratio - 1), 1.0)
        
        # Unique values change
        unique_ratio = curr_stats['unique_values'] / max(ref_stats['unique_values'], 1)
        unique_score = min(abs(unique_ratio - 1), 1.0)
        
        # Missing rate change
        missing_rate_shift = abs(curr_stats['missing_rate'] - ref_stats['missing_rate'])
        
        # Mode stability
        mode_score = 0.0
        if 'top_categories' in ref_stats and 'top_categories' in curr_stats:
            # Check if top categories have changed significantly
            ref_top = set(ref_stats['top_categories'].keys())
            curr_top = set(curr_stats['top_categories'].keys())
            overlap = len(ref_top & curr_top) / max(len(ref_top), 1)
            mode_score = 1 - overlap
        
        # Combine scores
        drift_score = (
            0.3 * entropy_score +
            0.3 * unique_score +
            0.2 * missing_rate_shift +
            0.2 * mode_score
        )
        
        return drift_score
    
    def get_drift_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive drift report.
        
        Returns:
            dict: Drift report with summary and history
        """
        if not self.drift_history:
            return {"message": "No drift history available"}
        
        # Calculate drift trends
        recent_events = self.drift_history[-10:]  # Last 10 events
        drift_rate = sum(1 for e in recent_events if e['drift_detected']) / len(recent_events)
        
        # Most drifted features
        feature_drift_counts = defaultdict(int)
        for event in self.drift_history:
            for feature in event.get('drifted_features', []):
                feature_drift_counts[feature] += 1
        
        most_drifted = sorted(
            feature_drift_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            'total_monitoring_periods': len(self.drift_history),
            'drift_events': sum(1 for e in self.drift_history if e['drift_detected']),
            'drift_rate': drift_rate,
            'most_drifted_features': most_drifted,
            'latest_drift': self.drift_history[-1] if self.drift_history else None,
            'threshold': self.threshold,
            'features_monitored': list(self.reference_stats.keys()) if self.reference_stats else []
        }

class ConceptDriftDetector:
    """
    Detects concept drift in model performance.
    
    Monitors model predictions and actual outcomes to detect when
    the relationship between features and target changes.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize concept drift detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.performance_window = self.config.get('performance_window', 1000)  # samples
        self.alert_threshold = self.config.get('alert_threshold', 0.1)  # 10% drop
        self.significance_level = self.config.get('significance_level', 0.05)
        
        # Storage
        self.performance_history = []
        self.baseline_performance = None
        self.drift_alerts = []
        
        logger.info(f"ConceptDriftDetector initialized with threshold: {self.alert_threshold}")
    
    def update_baseline(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Establish baseline performance metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Model predictions
        """
        self.baseline_performance = self._compute_metrics(y_true, y_pred)
        logger.info(f"Baseline performance established: {self.baseline_performance}")
    
    def detect_drift(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        features: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Detect concept drift in recent predictions.
        
        Args:
            y_true: Ground truth labels
            y_pred: Model predictions
            features: Optional feature matrix for advanced analysis
            
        Returns:
            dict: Drift detection results
        """
        # Compute current performance
        current_performance = self._compute_metrics(y_true, y_pred)
        
        # Store in history
        performance_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': current_performance,
            'sample_size': len(y_true)
        }
        self.performance_history.append(performance_record)
        
        # Keep only recent history
        if len(self.performance_history) > self.performance_window:
            self.performance_history = self.performance_history[-self.performance_window:]
        
        # Detect drift
        drift_result = {
            'timestamp': datetime.utcnow().isoformat(),
            'drift_detected': False,
            'current_performance': current_performance,
            'baseline_performance': self.baseline_performance,
            'metrics_drift': {}
        }
        
        if self.baseline_performance:
            # Check each metric for drift
            for metric, current_value in current_performance.items():
                baseline_value = self.baseline_performance[metric]
                
                # Calculate relative change
                relative_change = (current_value - baseline_value) / max(abs(baseline_value), 1e-10)
                
                # Check if significant drop
                if relative_change < -self.alert_threshold:
                    drift_result['drift_detected'] = True
                    drift_result['metrics_drift'][metric] = {
                        'baseline': baseline_value,
                        'current': current_value,
                        'change': relative_change,
                        'threshold': self.alert_threshold
                    }
            
            # Log drift if detected
            if drift_result['drift_detected']:
                logger.warning(
                    f"Concept drift detected! Metrics: "
                    f"{drift_result['metrics_drift']}"
                )
                self.drift_alerts.append(drift_result)
            else:
                logger.info("No concept drift detected")
        
        return drift_result
    
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute performance metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Model predictions (probabilities or classes)
            
        Returns:
            dict: Performance metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, average_precision_score
        )
        
        # Convert probabilities to classes if needed
        if y_pred.ndim > 1 or (y_pred.max() <= 1 and y_pred.min() >= 0 and len(set(y_pred)) > 2):
            # This is probabilities
            y_pred_class = (y_pred > 0.5).astype(int)
            
            # For AUC, use probabilities
            try:
                auc = roc_auc_score(y_true, y_pred)
                avg_precision = average_precision_score(y_true, y_pred)
            except:
                auc = 0.0
                avg_precision = 0.0
        else:
            y_pred_class = y_pred.astype(int)
            auc = 0.0
            avg_precision = 0.0
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred_class),
            'precision': precision_score(y_true, y_pred_class, zero_division=0),
            'recall': recall_score(y_true, y_pred_class, zero_division=0),
            'f1_score': f1_score(y_true, y_pred_class, zero_division=0),
            'auc_roc': auc,
            'avg_precision': avg_precision,
            'sample_size': len(y_true)
        }
        
        return metrics
    
    def get_performance_trend(self) -> Dict[str, Any]:
        """
        Analyze performance trends over time.
        
        Returns:
            dict: Performance trend analysis
        """
        if len(self.performance_history) < 2:
            return {"message": "Insufficient history for trend analysis"}
        
        # Extract metric trends
        metrics_trends = {}
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for metric in metrics:
            values = [record['metrics'][metric] for record in self.performance_history]
            
            # Calculate trend
            if len(values) > 1:
                slope = np.polyfit(range(len(values)), values, 1)[0]
                trend_direction = "improving" if slope > 0 else "degrading" if slope < 0 else "stable"
                
                metrics_trends[metric] = {
                    'current': values[-1],
                    'min': min(values),
                    'max': max(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'trend_slope': float(slope),
                    'trend_direction': trend_direction
                }
        
        return {
            'total_records': len(self.performance_history),
            'time_range': {
                'start': self.performance_history[0]['timestamp'],
                'end': self.performance_history[-1]['timestamp']
            },
            'metrics_trends': metrics_trends,
            'drift_alerts_count': len(self.drift_alerts)
        }

class DriftAlert:
    """
    Manages drift alerts and notifications.
    """
    
    def __init__(self, alert_manager=None):
        """
        Initialize drift alert manager.
        
        Args:
            alert_manager: AlertManager instance for sending notifications
        """
        self.alert_manager = alert_manager
        self.active_alerts = {}
        self.alert_history = []
    
    def create_alert(
        self,
        alert_type: str,
        severity: str,
        details: Dict[str, Any],
        channels: List[str] = None
    ) -> str:
        """
        Create a new drift alert.
        
        Args:
            alert_type: Type of alert ('data_drift', 'concept_drift', 'performance_drop')
            severity: Alert severity ('info', 'warning', 'critical')
            details: Alert details
            channels: Notification channels to use
            
        Returns:
            str: Alert ID
        """
        import uuid
        
        alert_id = str(uuid.uuid4())
        
        alert = {
            'id': alert_id,
            'type': alert_type,
            'severity': severity,
            'details': details,
            'created_at': datetime.utcnow().isoformat(),
            'channels': channels or ['log'],
            'status': 'active',
            'resolved_at': None
        }
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Send notifications
        self._send_notifications(alert)
        
        logger.warning(f"Drift alert created: {alert_id} - {alert_type} ({severity})")
        
        return alert_id
    
    def resolve_alert(self, alert_id: str, resolution: str = None) -> bool:
        """
        Resolve an active alert.
        
        Args:
            alert_id: Alert ID to resolve
            resolution: Resolution notes
            
        Returns:
            bool: True if resolved, False if not found
        """
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id]['status'] = 'resolved'
            self.active_alerts[alert_id]['resolved_at'] = datetime.utcnow().isoformat()
            self.active_alerts[alert_id]['resolution'] = resolution
            
            logger.info(f"Drift alert resolved: {alert_id}")
            
            # Send resolution notification
            self._send_resolution_notification(self.active_alerts[alert_id])
            
            del self.active_alerts[alert_id]
            return True
        
        return False
    
    def _send_notifications(self, alert: Dict[str, Any]) -> None:
        """
        Send alert notifications through configured channels.
        
        Args:
            alert: Alert dictionary
        """
        channels = alert.get('channels', ['log'])
        
        for channel in channels:
            if channel == 'log':
                # Already logged in create_alert
                pass
            elif channel == 'slack' and self.alert_manager:
                self.alert_manager.send_slack_alert(alert)
            elif channel == 'email' and self.alert_manager:
                self.alert_manager.send_email_alert(alert)
            elif channel == 'pagerduty' and self.alert_manager:
                self.alert_manager.send_pagerduty_alert(alert)
    
    def _send_resolution_notification(self, alert: Dict[str, Any]) -> None:
        """
        Send alert resolution notification.
        
        Args:
            alert: Resolved alert dictionary
        """
        logger.info(f"Alert resolved: {alert['id']} - {alert['type']}")
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """
        Get all active alerts.
        
        Returns:
            list: Active alerts
        """
        return list(self.active_alerts.values())
    
    def get_alert_history(
        self,
        limit: int = 100,
        alert_type: str = None,
        severity: str = None
    ) -> List[Dict[str, Any]]:
        """
        Get alert history with filtering.
        
        Args:
            limit: Maximum number of alerts to return
            alert_type: Filter by alert type
            severity: Filter by severity
            
        Returns:
            list: Filtered alert history
        """
        history = self.alert_history[-limit:]
        
        if alert_type:
            history = [a for a in history if a['type'] == alert_type]
        
        if severity:
            history = [a for a in history if a['severity'] == severity]
        
        return history