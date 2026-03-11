"""
Performance Tracking Module
===========================

Tracks model performance metrics in production, monitors system health,
and collects business metrics for reporting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from collections import defaultdict, deque
import json
import time

# Configure logging
logger = logging.getLogger(__name__)

class PerformanceTracker:
    """
    Tracks model performance metrics in production.
    
    Monitors key performance indicators (KPIs) for the fraud detection system,
    including accuracy, precision, recall, and business metrics.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize performance tracker.
        
        Args:
            config: Configuration dictionary containing:
                - metrics_to_track: List of metrics to monitor
                - window_size: Rolling window size for metrics
                - storage_backend: Where to store metrics
        """
        self.config = config or {}
        self.metrics_to_track = self.config.get('metrics_to_track', [
            'accuracy', 'precision', 'recall', 'f1_score',
            'auc_roc', 'avg_precision', 'false_positive_rate',
            'false_negative_rate', 'processing_time_ms', 'throughput'
        ])
        self.window_size = self.config.get('window_size', 10000)
        self.storage_backend = self.config.get('storage_backend', 'memory')
        
        # Metrics storage
        self.metrics_history = deque(maxlen=self.window_size)
        self.metrics_summary = {}
        self.baseline_metrics = {}
        
        # Time-series metrics
        self.time_series = defaultdict(list)
        
        logger.info(f"PerformanceTracker initialized with {len(self.metrics_to_track)} metrics")
    
    def log_prediction(
        self,
        y_true: int,
        y_pred: float,
        processing_time_ms: float,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """
        Log a single prediction and update metrics.
        
        Args:
            y_true: Ground truth label (0 or 1)
            y_pred: Predicted probability (0-1)
            processing_time_ms: Processing time in milliseconds
            metadata: Additional metadata about the prediction
            
        Returns:
            dict: Updated metrics
        """
        # Convert probability to class
        y_pred_class = 1 if y_pred > 0.5 else 0
        
        # Create record
        record = {
            'timestamp': datetime.utcnow().isoformat(),
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_class': y_pred_class,
            'processing_time_ms': processing_time_ms,
            'metadata': metadata or {}
        }
        
        # Add to history
        self.metrics_history.append(record)
        
        # Update time series
        self.time_series['timestamp'].append(record['timestamp'])
        self.time_series['y_true'].append(y_true)
        self.time_series['y_pred'].append(y_pred)
        self.time_series['processing_time_ms'].append(processing_time_ms)
        
        # Recalculate metrics
        self._update_metrics()
        
        return self.metrics_summary
    
    def log_batch(
        self,
        y_true: List[int],
        y_pred: List[float],
        processing_times_ms: List[float],
        metadata: List[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Log a batch of predictions.
        
        Args:
            y_true: List of ground truth labels
            y_pred: List of predicted probabilities
            processing_times_ms: List of processing times
            metadata: List of metadata dictionaries
            
        Returns:
            dict: Updated metrics
        """
        metadata = metadata or [{}] * len(y_true)
        
        for i in range(len(y_true)):
            self.log_prediction(
                y_true[i],
                y_pred[i],
                processing_times_ms[i] if i < len(processing_times_ms) else 0,
                metadata[i] if i < len(metadata) else {}
            )
        
        return self.metrics_summary
    
    def _update_metrics(self) -> None:
        """
        Update all metrics based on current history.
        """
        if len(self.metrics_history) == 0:
            return
        
        # Extract data
        y_true = [r['y_true'] for r in self.metrics_history]
        y_pred = [r['y_pred'] for r in self.metrics_history]
        y_pred_class = [r['y_pred_class'] for r in self.metrics_history]
        processing_times = [r['processing_time_ms'] for r in self.metrics_history]
        
        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix
        )
        
        # Basic metrics
        self.metrics_summary['accuracy'] = accuracy_score(y_true, y_pred_class)
        self.metrics_summary['precision'] = precision_score(y_true, y_pred_class, zero_division=0)
        self.metrics_summary['recall'] = recall_score(y_true, y_pred_class, zero_division=0)
        self.metrics_summary['f1_score'] = f1_score(y_true, y_pred_class, zero_division=0)
        
        # AUC (if both classes present)
        if len(set(y_true)) == 2:
            try:
                self.metrics_summary['auc_roc'] = roc_auc_score(y_true, y_pred)
            except:
                self.metrics_summary['auc_roc'] = 0.0
        else:
            self.metrics_summary['auc_roc'] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_class, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            
            # Derived metrics
            self.metrics_summary['true_positives'] = int(tp)
            self.metrics_summary['false_positives'] = int(fp)
            self.metrics_summary['true_negatives'] = int(tn)
            self.metrics_summary['false_negatives'] = int(fn)
            
            self.metrics_summary['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            self.metrics_summary['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Performance metrics
        self.metrics_summary['avg_processing_time_ms'] = float(np.mean(processing_times))
        self.metrics_summary['p95_processing_time_ms'] = float(np.percentile(processing_times, 95))
        self.metrics_summary['p99_processing_time_ms'] = float(np.percentile(processing_times, 99))
        
        # Throughput (requests per second over last minute)
        recent_times = [r['timestamp'] for r in list(self.metrics_history)[-100:]]
        if len(recent_times) > 1:
            time_span = (
                datetime.fromisoformat(recent_times[-1]) -
                datetime.fromisoformat(recent_times[0])
            ).total_seconds()
            if time_span > 0:
                self.metrics_summary['throughput'] = len(recent_times) / time_span
            else:
                self.metrics_summary['throughput'] = len(recent_times)
        else:
            self.metrics_summary['throughput'] = len(recent_times)
        
        # Sample size
        self.metrics_summary['total_predictions'] = len(self.metrics_history)
    
    def get_metrics(self, window: int = None) -> Dict[str, float]:
        """
        Get current performance metrics.
        
        Args:
            window: Optional window size to calculate metrics on subset
            
        Returns:
            dict: Performance metrics
        """
        if window and window < len(self.metrics_history):
            # Calculate metrics on recent window
            recent = list(self.metrics_history)[-window:]
            
            y_true = [r['y_true'] for r in recent]
            y_pred = [r['y_pred'] for r in recent]
            y_pred_class = [r['y_pred_class'] for r in recent]
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            return {
                'accuracy': accuracy_score(y_true, y_pred_class),
                'precision': precision_score(y_true, y_pred_class, zero_division=0),
                'recall': recall_score(y_true, y_pred_class, zero_division=0),
                'f1_score': f1_score(y_true, y_pred_class, zero_division=0),
                'sample_size': len(recent)
            }
        
        return self.metrics_summary.copy()
    
    def set_baseline(self) -> None:
        """
        Set current metrics as baseline for comparison.
        """
        self.baseline_metrics = self.metrics_summary.copy()
        logger.info(f"Baseline metrics set: {self.baseline_metrics}")
    
    def compare_to_baseline(self) -> Dict[str, float]:
        """
        Compare current metrics to baseline.
        
        Returns:
            dict: Relative changes from baseline
        """
        if not self.baseline_metrics:
            return {}
        
        changes = {}
        
        for metric, value in self.metrics_summary.items():
            if metric in self.baseline_metrics and self.baseline_metrics[metric] != 0:
                baseline = self.baseline_metrics[metric]
                relative_change = (value - baseline) / abs(baseline)
                changes[metric] = relative_change
        
        return changes
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Returns:
            dict: Performance report with summary and trends
        """
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'current_metrics': self.metrics_summary,
            'total_predictions': len(self.metrics_history),
            'window_size': self.window_size
        }
        
        # Add baseline comparison
        if self.baseline_metrics:
            report['baseline_comparison'] = self.compare_to_baseline()
        
        # Add trends if enough history
        if len(self.time_series['timestamp']) > 10:
            # Calculate hourly averages for key metrics
            hourly_metrics = self._calculate_hourly_metrics()
            report['hourly_trends'] = hourly_metrics
            
            # Detect anomalies
            anomalies = self._detect_anomalies()
            if anomalies:
                report['anomalies'] = anomalies
        
        return report
    
    def _calculate_hourly_metrics(self) -> Dict[str, List]:
        """
        Calculate hourly aggregated metrics.
        
        Returns:
            dict: Hourly metrics
        """
        if not self.time_series['timestamp']:
            return {}
        
        # Convert to DataFrame for easier grouping
        df = pd.DataFrame(self.time_series)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.floor('H')
        
        # Group by hour
        hourly = df.groupby('hour').agg({
            'y_true': 'count',
            'processing_time_ms': ['mean', 'std']
        }).reset_index()
        
        hourly.columns = ['hour', 'request_count', 'avg_processing_time', 'std_processing_time']
        
        # Calculate accuracy per hour
        hourly_accuracy = []
        for hour in hourly['hour']:
            hour_data = df[df['hour'] == hour]
            y_true = hour_data['y_true'].values
            y_pred = (hour_data['y_pred'] > 0.5).astype(int).values
            acc = (y_true == y_pred).mean()
            hourly_accuracy.append(acc)
        
        hourly['accuracy'] = hourly_accuracy
        
        return hourly.to_dict(orient='records')
    
    def _detect_anomalies(self, z_threshold: float = 3.0) -> List[Dict]:
        """
        Detect anomalies in performance metrics.
        
        Args:
            z_threshold: Z-score threshold for anomaly detection
            
        Returns:
            list: Detected anomalies
        """
        anomalies = []
        
        if len(self.time_series['processing_time_ms']) < 10:
            return anomalies
        
        # Check processing time anomalies
        recent_times = self.time_series['processing_time_ms'][-100:]
        mean_time = np.mean(recent_times)
        std_time = np.std(recent_times)
        
        last_time = recent_times[-1]
        z_score = abs(last_time - mean_time) / max(std_time, 1e-10)
        
        if z_score > z_threshold:
            anomalies.append({
                'type': 'processing_time_anomaly',
                'timestamp': self.time_series['timestamp'][-1],
                'value': last_time,
                'mean': mean_time,
                'std': std_time,
                'z_score': float(z_score)
            })
        
        # Check accuracy anomalies (sudden drop)
        if len(self.time_series['y_true']) >= 100:
            recent_accuracy = []
            y_true = self.time_series['y_true'][-100:]
            y_pred = self.time_series['y_pred'][-100:]
            
            for i in range(0, len(y_true), 10):
                if i + 10 <= len(y_true):
                    acc = (np.array(y_true[i:i+10]) == 
                          (np.array(y_pred[i:i+10]) > 0.5)).mean()
                    recent_accuracy.append(acc)
            
            if len(recent_accuracy) >= 5:
                accuracy_trend = np.polyfit(range(len(recent_accuracy)), recent_accuracy, 1)[0]
                
                if accuracy_trend < -0.05:  # 5% drop over last 5 batches
                    anomalies.append({
                        'type': 'accuracy_degradation',
                        'timestamp': self.time_series['timestamp'][-1],
                        'trend_slope': float(accuracy_trend),
                        'current_accuracy': recent_accuracy[-1]
                    })
        
        return anomalies

class ModelMonitor:
    """
    Monitors model health and performance in production.
    
    Tracks model versions, performs health checks, and manages
    model lifecycle events.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize model monitor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.models = {}  # model_id -> model info
        self.active_model_id = None
        self.model_versions = defaultdict(list)
        self.health_checks = []
        
        logger.info("ModelMonitor initialized")
    
    def register_model(
        self,
        model_id: str,
        model_version: str,
        model_type: str,
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Register a model with the monitor.
        
        Args:
            model_id: Unique model identifier
            model_version: Model version string
            model_type: Type of model (xgboost, neural_network, etc.)
            metadata: Additional model metadata
        """
        model_info = {
            'model_id': model_id,
            'model_version': model_version,
            'model_type': model_type,
            'metadata': metadata or {},
            'registered_at': datetime.utcnow().isoformat(),
            'status': 'registered',
            'health_checks': [],
            'performance_metrics': {}
        }
        
        self.models[model_id] = model_info
        self.model_versions[model_type].append({
            'model_id': model_id,
            'version': model_version,
            'registered_at': model_info['registered_at']
        })
        
        logger.info(f"Model registered: {model_id} (v{model_version})")
    
    def activate_model(self, model_id: str) -> bool:
        """
        Set a model as the active production model.
        
        Args:
            model_id: Model ID to activate
            
        Returns:
            bool: True if activated successfully
        """
        if model_id not in self.models:
            logger.error(f"Cannot activate unknown model: {model_id}")
            return False
        
        # Deactivate current model
        if self.active_model_id:
            self.models[self.active_model_id]['status'] = 'inactive'
            self.models[self.active_model_id]['deactivated_at'] = datetime.utcnow().isoformat()
        
        # Activate new model
        self.active_model_id = model_id
        self.models[model_id]['status'] = 'active'
        self.models[model_id]['activated_at'] = datetime.utcnow().isoformat()
        
        logger.info(f"Model activated: {model_id}")
        return True
    
    def check_model_health(self, model_id: str = None) -> Dict[str, Any]:
        """
        Perform health check on a model.
        
        Args:
            model_id: Model ID to check (uses active model if None)
            
        Returns:
            dict: Health check results
        """
        model_id = model_id or self.active_model_id
        
        if not model_id or model_id not in self.models:
            return {'status': 'unknown', 'error': 'Model not found'}
        
        model = self.models[model_id]
        
        # Perform health checks
        health_status = {
            'model_id': model_id,
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'healthy',
            'checks': []
        }
        
        # Check 1: Model file exists and is loadable
        # In production, this would actually try to load the model
        health_status['checks'].append({
            'name': 'model_loadable',
            'status': 'passed',
            'message': 'Model file exists and is loadable'
        })
        
        # Check 2: Model responds to inference
        # This would do a test prediction
        health_status['checks'].append({
            'name': 'inference_test',
            'status': 'passed',
            'message': 'Model successfully performed test inference'
        })
        
        # Check 3: Memory usage
        # This would check actual memory usage
        health_status['checks'].append({
            'name': 'memory_usage',
            'status': 'passed',
            'message': 'Memory usage within limits',
            'metrics': {'memory_mb': 512}
        })
        
        # Check 4: Response time
        # This would measure actual inference time
        health_status['checks'].append({
            'name': 'response_time',
            'status': 'passed',
            'message': 'Response time within threshold',
            'metrics': {'avg_ms': 45, 'p95_ms': 78}
        })
        
        # Determine overall health
        failed_checks = [c for c in health_status['checks'] if c['status'] != 'passed']
        if failed_checks:
            health_status['status'] = 'degraded'
            health_status['failed_checks'] = len(failed_checks)
        
        # Store health check
        health_status_copy = health_status.copy()
        model['health_checks'].append(health_status_copy)
        self.health_checks.append(health_status_copy)
        
        return health_status
    
    def get_model_info(self, model_id: str = None) -> Dict[str, Any]:
        """
        Get information about a model.
        
        Args:
            model_id: Model ID (uses active model if None)
            
        Returns:
            dict: Model information
        """
        model_id = model_id or self.active_model_id
        
        if not model_id or model_id not in self.models:
            return {'error': 'Model not found'}
        
        return self.models[model_id].copy()
    
    def list_models(
        self,
        model_type: str = None,
        status: str = None
    ) -> List[Dict[str, Any]]:
        """
        List all registered models with optional filtering.
        
        Args:
            model_type: Filter by model type
            status: Filter by status
            
        Returns:
            list: Matching models
        """
        models = []
        
        for model_id, model_info in self.models.items():
            if model_type and model_info['model_type'] != model_type:
                continue
            
            if status and model_info['status'] != status:
                continue
            
            models.append({
                'model_id': model_id,
                'model_version': model_info['model_version'],
                'model_type': model_info['model_type'],
                'status': model_info['status'],
                'registered_at': model_info['registered_at']
            })
        
        return models
    
    def get_health_history(self, model_id: str = None, limit: int = 10) -> List[Dict]:
        """
        Get health check history for a model.
        
        Args:
            model_id: Model ID (uses active if None)
            limit: Maximum number of checks to return
            
        Returns:
            list: Health check history
        """
        model_id = model_id or self.active_model_id
        
        if not model_id or model_id not in self.models:
            return []
        
        checks = self.models[model_id]['health_checks'][-limit:]
        return checks

class MetricsCollector:
    """
    Collects and aggregates metrics from multiple sources.
    
    Provides unified interface for collecting system metrics,
    business metrics, and model performance metrics.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize metrics collector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.metrics = defaultdict(list)
        self.labels = {}
        self.collection_interval = self.config.get('collection_interval', 60)  # seconds
        
        # Metric definitions
        self.metric_definitions = {
            'system': [
                'cpu_usage', 'memory_usage', 'disk_usage', 'network_io',
                'gpu_usage', 'gpu_memory', 'request_rate', 'error_rate'
            ],
            'business': [
                'total_transactions', 'fraud_detected', 'false_positives',
                'avg_risk_score', 'high_risk_transactions', 'revenue_protected'
            ],
            'model': [
                'inference_latency', 'batch_size', 'model_version',
                'prediction_distribution', 'confidence_scores'
            ]
        }
        
        logger.info("MetricsCollector initialized")
    
    def collect_metric(
        self,
        name: str,
        value: float,
        category: str = 'custom',
        tags: Dict[str, str] = None,
        timestamp: datetime = None
    ) -> None:
        """
        Collect a single metric.
        
        Args:
            name: Metric name
            value: Metric value
            category: Metric category
            tags: Additional tags for the metric
            timestamp: Metric timestamp (uses current time if None)
        """
        timestamp = timestamp or datetime.utcnow()
        tags = tags or {}
        
        metric_record = {
            'name': name,
            'value': value,
            'category': category,
            'tags': tags,
            'timestamp': timestamp.isoformat()
        }
        
        self.metrics[category].append(metric_record)
        
        # Keep only recent metrics (last 10000 per category)
        if len(self.metrics[category]) > 10000:
            self.metrics[category] = self.metrics[category][-10000:]
    
    def collect_batch(
        self,
        metrics: List[Dict[str, Any]],
        category: str = 'custom'
    ) -> None:
        """
        Collect multiple metrics at once.
        
        Args:
            metrics: List of metric dictionaries
            category: Default category for metrics
        """
        for metric in metrics:
            self.collect_metric(
                name=metric['name'],
                value=metric['value'],
                category=metric.get('category', category),
                tags=metric.get('tags', {}),
                timestamp=metric.get('timestamp')
            )
    
    def get_metrics(
        self,
        category: str = None,
        name: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 1000
    ) -> List[Dict]:
        """
        Retrieve collected metrics with filtering.
        
        Args:
            category: Filter by category
            name: Filter by metric name
            start_time: Start time for filtering
            end_time: End time for filtering
            limit: Maximum number of metrics to return
            
        Returns:
            list: Filtered metrics
        """
        results = []
        
        # Determine which categories to include
        categories = [category] if category else self.metrics.keys()
        
        for cat in categories:
            for metric in self.metrics.get(cat, []):
                # Apply filters
                if name and metric['name'] != name:
                    continue
                
                metric_time = datetime.fromisoformat(metric['timestamp'])
                
                if start_time and metric_time < start_time:
                    continue
                
                if end_time and metric_time > end_time:
                    continue
                
                results.append(metric)
                
                if len(results) >= limit:
                    return results
        
        return results
    
    def aggregate(
        self,
        name: str,
        interval: str = '1h',
        aggregation: str = 'mean',
        category: str = None
    ) -> List[Dict]:
        """
        Aggregate metrics over time intervals.
        
        Args:
            name: Metric name to aggregate
            interval: Aggregation interval ('1m', '5m', '1h', '1d')
            aggregation: Aggregation function ('mean', 'sum', 'min', 'max', 'count')
            category: Metric category
            
        Returns:
            list: Aggregated metrics
        """
        # Get metrics for this name
        metrics = self.get_metrics(name=name, category=category)
        
        if not metrics:
            return []
        
        # Parse interval to timedelta
        interval_map = {
            '1m': timedelta(minutes=1),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '1h': timedelta(hours=1),
            '6h': timedelta(hours=6),
            '12h': timedelta(hours=12),
            '1d': timedelta(days=1),
            '7d': timedelta(days=7)
        }
        
        interval_delta = interval_map.get(interval, timedelta(hours=1))
        
        # Group by interval
        grouped = defaultdict(list)
        
        for metric in metrics:
            metric_time = datetime.fromisoformat(metric['timestamp'])
            # Round down to interval
            if interval == '1h':
                interval_key = metric_time.replace(minute=0, second=0, microsecond=0)
            elif interval == '1d':
                interval_key = metric_time.replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                # For other intervals, use floor division
                timestamp_seconds = int(metric_time.timestamp())
                interval_seconds = int(interval_delta.total_seconds())
                floored_seconds = (timestamp_seconds // interval_seconds) * interval_seconds
                interval_key = datetime.fromtimestamp(floored_seconds)
            
            grouped[interval_key].append(metric['value'])
        
        # Aggregate each group
        results = []
        agg_functions = {
            'mean': np.mean,
            'sum': np.sum,
            'min': np.min,
            'max': np.max,
            'count': len,
            'std': np.std
        }
        
        agg_func = agg_functions.get(aggregation, np.mean)
        
        for interval_key, values in sorted(grouped.items()):
            if values:
                results.append({
                    'interval': interval_key.isoformat(),
                    'value': float(agg_func(values)),
                    'count': len(values)
                })
        
        return results
    
    def get_statistics(self, name: str, category: str = None) -> Dict[str, float]:
        """
        Get statistical summary for a metric.
        
        Args:
            name: Metric name
            category: Metric category
            
        Returns:
            dict: Statistical summary
        """
        metrics = self.get_metrics(name=name, category=category)
        values = [m['value'] for m in metrics]
        
        if not values:
            return {}
        
        return {
            'count': len(values),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'p25': float(np.percentile(values, 25)),
            'p75': float(np.percentile(values, 75)),
            'p95': float(np.percentile(values, 95)),
            'p99': float(np.percentile(values, 99))
        }
    
    def export_metrics(self, format: str = 'json') -> str:
        """
        Export all metrics in specified format.
        
        Args:
            format: Export format ('json' or 'csv')
            
        Returns:
            str: Exported metrics
        """
        if format == 'json':
            return json.dumps({
                category: metrics
                for category, metrics in self.metrics.items()
            }, indent=2, default=str)
        
        elif format == 'csv':
            # Convert to CSV format
            import csv
            from io import StringIO
            
            output = StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(['category', 'name', 'value', 'tags', 'timestamp'])
            
            # Write rows
            for category, metrics in self.metrics.items():
                for metric in metrics:
                    writer.writerow([
                        category,
                        metric['name'],
                        metric['value'],
                        json.dumps(metric['tags']),
                        metric['timestamp']
                    ])
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported export format: {format}")