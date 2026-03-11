# src/modeling/training/early_stopping.py
"""
Advanced Early Stopping Module for Fraud Detection

This module provides sophisticated early stopping mechanisms specifically
designed for imbalanced fraud detection tasks, including:
1. Gradient-based stopping
2. Plateau detection with patience
3. Minimum improvement thresholds
4. Class-aware stopping criteria

Author: VeritasFinancial DS Team
Version: 1.0.0
"""

import numpy as np
import warnings
from typing import Optional, Dict, Any, Callable, List, Union, Tuple
from dataclasses import dataclass, field
from collections import deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EarlyStoppingHistory:
    """
    Data class to store early stopping history.
    
    Attributes:
        epoch: List of epoch numbers
        metrics: Dictionary of metric histories
        best_epoch: Best epoch number
        best_score: Best score achieved
        stopped_epoch: Epoch where training stopped
        reason: Reason for stopping
    """
    epoch: List[int] = field(default_factory=list)
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    best_epoch: int = 0
    best_score: float = 0.0
    stopped_epoch: Optional[int] = None
    reason: str = ""
    
    def add(self, epoch: int, **kwargs):
        """Add metrics for an epoch."""
        self.epoch.append(epoch)
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'epoch': self.epoch,
            'metrics': self.metrics,
            'best_epoch': self.best_epoch,
            'best_score': self.best_score,
            'stopped_epoch': self.stopped_epoch,
            'reason': self.reason
        }


class EarlyStopping:
    """
    Base Early Stopping class for fraud detection models.
    
    This class provides the foundation for various early stopping strategies
    with support for:
    1. Multiple monitoring metrics
    2. Class-aware stopping criteria
    3. Minimum improvement thresholds
    4. Cooldown periods
    
    Parameters:
        monitor: str or List[str], default='val_loss'
            Metric(s) to monitor for stopping
        mode: str, default='min'
            'min' for metrics that should decrease (loss),
            'max' for metrics that should increase (accuracy, f1)
        patience: int, default=10
            Number of epochs to wait for improvement
        min_delta: float, default=0.001
            Minimum change to qualify as improvement
        cooldown: int, default=0
            Number of epochs to wait before resuming normal operation
        baseline: float, optional
            Baseline value for the monitored quantity
        restore_best_weights: bool, default=True
            Whether to restore model weights from the best epoch
        verbose: bool, default=False
            Whether to print stopping information
    """
    
    def __init__(
        self,
        monitor: Union[str, List[str]] = 'val_loss',
        mode: str = 'min',
        patience: int = 10,
        min_delta: float = 0.001,
        cooldown: int = 0,
        baseline: Optional[float] = None,
        restore_best_weights: bool = True,
        verbose: bool = False
    ):
        # Validate inputs
        if mode not in ['min', 'max']:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        
        if patience < 0:
            raise ValueError(f"patience must be >= 0, got {patience}")
        
        if min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {min_delta}")
        
        # Convert single monitor to list for consistency
        self.monitor = [monitor] if isinstance(monitor, str) else monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = abs(min_delta)  # Ensure positive
        self.cooldown = cooldown
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        # Initialize state
        self.best_weights = None
        self.best_epoch = 0
        self.best_score = None
        self.wait = 0
        self.stopped_epoch = 0
        self.cooldown_counter = 0
        self.history = EarlyStoppingHistory()
        
        # For multiple metrics
        self.metric_weights = {}
        self.metric_thresholds = {}
        
    def _is_improvement(self, current: float, best: float) -> bool:
        """
        Check if current value is an improvement over best.
        
        Parameters:
            current: Current metric value
            best: Best metric value so far
            
        Returns:
            is_improved: Whether current is better than best
        """
        if self.mode == 'min':
            return current < best - self.min_delta
        else:  # mode == 'max'
            return current > best + self.min_delta
    
    def _get_monitor_value(self, logs: Dict[str, float]) -> Optional[float]:
        """
        Extract monitored value from logs.
        
        Parameters:
            logs: Dictionary of metrics from current epoch
            
        Returns:
            value: Monitored value or None if not found
        """
        # Try each monitor in order
        for monitor_name in self.monitor:
            if monitor_name in logs:
                return logs[monitor_name]
        
        # If none found, warn and return None
        available_metrics = list(logs.keys())
        warnings.warn(
            f"Monitored metrics {self.monitor} not found in logs. "
            f"Available metrics: {available_metrics}"
        )
        return None
    
    def on_train_begin(self):
        """Initialize at start of training."""
        self.best_weights = None
        self.best_epoch = 0
        self.best_score = None
        self.wait = 0
        self.stopped_epoch = 0
        self.cooldown_counter = 0
        self.history = EarlyStoppingHistory()
        
        if self.baseline is not None:
            self.best_score = self.baseline
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float], model: Any = None) -> bool:
        """
        Check if training should stop at the end of current epoch.
        
        Parameters:
            epoch: Current epoch number
            logs: Dictionary of metrics
            model: Model object (to save weights if needed)
            
        Returns:
            stop_training: Whether to stop training
        """
        # Get monitored value
        current = self._get_monitor_value(logs)
        
        if current is None:
            # If monitored metric not available, don't stop
            return False
        
        # Add to history
        self.history.add(epoch, **logs)
        
        # Handle cooldown period
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.wait = 0
            return False
        
        # Initialize best score if not set
        if self.best_score is None:
            self.best_score = current
            self.best_epoch = epoch
            if self.restore_best_weights and hasattr(model, 'get_weights'):
                self.best_weights = self._get_model_weights(model)
            return False
        
        # Check for improvement
        if self._is_improvement(current, self.best_score):
            self.best_score = current
            self.best_epoch = epoch
            self.wait = 0
            
            # Save best weights
            if self.restore_best_weights and hasattr(model, 'get_weights'):
                self.best_weights = self._get_model_weights(model)
            
            if self.verbose:
                logger.info(f"Epoch {epoch}: {self.monitor} improved to {current:.6f}")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.history.best_epoch = self.best_epoch
                self.history.best_score = self.best_score
                self.history.stopped_epoch = epoch
                self.history.reason = f"No improvement for {self.patience} epochs"
                
                if self.verbose:
                    logger.info(f"Early stopping at epoch {epoch}")
                
                return True
        
        return False
    
    def _get_model_weights(self, model: Any) -> Any:
        """Extract model weights based on model type."""
        if hasattr(model, 'get_weights'):  # Keras/TensorFlow
            return model.get_weights()
        elif hasattr(model, 'state_dict'):  # PyTorch
            return model.state_dict()
        elif hasattr(model, 'get_params'):  # Scikit-learn
            return model.get_params()
        else:
            return None
    
    def on_train_end(self, model: Any = None):
        """Restore best weights at end of training."""
        if self.restore_best_weights and self.best_weights is not None and model is not None:
            if self.verbose:
                logger.info(f"Restoring model weights from epoch {self.best_epoch}")
            
            self._set_model_weights(model, self.best_weights)
    
    def _set_model_weights(self, model: Any, weights: Any):
        """Set model weights based on model type."""
        if hasattr(model, 'set_weights'):  # Keras/TensorFlow
            model.set_weights(weights)
        elif hasattr(model, 'load_state_dict'):  # PyTorch
            model.load_state_dict(weights)
        elif hasattr(model, 'set_params'):  # Scikit-learn
            model.set_params(**weights)
    
    def get_history(self) -> EarlyStoppingHistory:
        """Get training history."""
        return self.history


class GradientBasedEarlyStopping(EarlyStopping):
    """
    Early stopping based on gradient information.
    
    This is particularly useful for deep learning models where
    gradient norms can indicate convergence.
    
    Parameters:
        monitor_gradient: str, default='gradient_norm'
            Gradient metric to monitor
        min_gradient_norm: float, default=1e-7
            Minimum gradient norm to continue training
        Additional parameters from EarlyStopping
    """
    
    def __init__(
        self,
        monitor_gradient: str = 'gradient_norm',
        min_gradient_norm: float = 1e-7,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.monitor_gradient = monitor_gradient
        self.min_gradient_norm = min_gradient_norm
        self.gradient_history = []
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float], model: Any = None) -> bool:
        """
        Check gradient-based stopping criteria.
        """
        # Check gradient norm
        if self.monitor_gradient in logs:
            grad_norm = logs[self.monitor_gradient]
            self.gradient_history.append(grad_norm)
            
            # Stop if gradient norm is too small
            if grad_norm < self.min_gradient_norm:
                if self.verbose:
                    logger.info(f"Gradient norm {grad_norm:.2e} below minimum threshold")
                self.stopped_epoch = epoch
                self.history.reason = f"Gradient norm below {self.min_gradient_norm}"
                return True
        
        # Also check standard metrics
        return super().on_epoch_end(epoch, logs, model)


class ClassAwareEarlyStopping(EarlyStopping):
    """
    Early stopping that considers class-specific metrics.
    
    This is crucial for fraud detection where we care more about
    minority class performance.
    
    Parameters:
        class_weight: float, default=0.7
            Weight for minority class metrics
        minority_class: int, default=1
            Label for minority class (fraud)
        Additional parameters from EarlyStopping
    """
    
    def __init__(
        self,
        class_weight: float = 0.7,
        minority_class: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.class_weight = class_weight
        self.minority_class = minority_class
        self.class_metrics = {}
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float], model: Any = None) -> bool:
        """
        Check class-aware stopping criteria.
        """
        # Extract class-specific metrics
        minority_metrics = {}
        majority_metrics = {}
        
        for key, value in logs.items():
            if f'class_{self.minority_class}' in key:
                minority_metrics[key] = value
            elif 'class_0' in key:
                majority_metrics[key] = value
        
        # Calculate weighted score if we have both
        if minority_metrics and majority_metrics:
            # Get primary metric (e.g., f1_score for each class)
            minority_f1 = minority_metrics.get(f'f1_score_class_{self.minority_class}', 0)
            majority_f1 = majority_metrics.get('f1_score_class_0', 0)
            
            # Weighted combination
            weighted_score = (self.class_weight * minority_f1 + 
                            (1 - self.class_weight) * majority_f1)
            
            logs['weighted_class_score'] = weighted_score
        
        return super().on_epoch_end(epoch, logs, model)


class PlateauDetectionEarlyStopping(EarlyStopping):
    """
    Early stopping with sophisticated plateau detection.
    
    Uses statistical tests to detect when the model has plateaued,
    rather than just waiting for patience epochs.
    
    Parameters:
        window_size: int, default=5
            Number of recent epochs to consider for plateau detection
        std_threshold: float, default=1e-4
            Standard deviation threshold for plateau
        trend_threshold: float, default=1e-4
            Slope threshold for plateau
        Additional parameters from EarlyStopping
    """
    
    def __init__(
        self,
        window_size: int = 5,
        std_threshold: float = 1e-4,
        trend_threshold: float = 1e-4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.std_threshold = std_threshold
        self.trend_threshold = trend_threshold
        self.recent_values = deque(maxlen=window_size)
    
    def _detect_plateau(self, values: List[float]) -> bool:
        """
        Detect if the values have plateaued using statistical tests.
        
        Parameters:
            values: List of recent metric values
            
        Returns:
            is_plateau: Whether a plateau is detected
        """
        if len(values) < self.window_size:
            return False
        
        # Calculate statistics
        values_array = np.array(values)
        
        # Check if standard deviation is very low
        if np.std(values_array) < self.std_threshold:
            return True
        
        # Check if trend (slope) is very low
        x = np.arange(len(values_array))
        try:
            slope, _ = np.polyfit(x, values_array, 1)
            if abs(slope) < self.trend_threshold:
                return True
        except:
            pass
        
        return False
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float], model: Any = None) -> bool:
        """
        Check plateau-based stopping criteria.
        """
        # Get monitored value
        current = self._get_monitor_value(logs)
        
        if current is not None:
            self.recent_values.append(current)
            
            # Check for plateau
            if self._detect_plateau(list(self.recent_values)):
                if self.verbose:
                    logger.info(f"Plateau detected at epoch {epoch}")
                self.stopped_epoch = epoch
                self.history.reason = "Plateau detected"
                return True
        
        # Also check standard criteria
        return super().on_epoch_end(epoch, logs, model)


class CompositeEarlyStopping:
    """
    Combine multiple early stopping strategies.
    
    This allows using different stopping criteria simultaneously,
    stopping when any of them triggers.
    
    Parameters:
        stopping_strategies: List[EarlyStopping]
            List of early stopping strategies
        stop_on_any: bool, default=True
            If True, stop when any strategy triggers
            If False, stop when all trigger
    """
    
    def __init__(
        self,
        stopping_strategies: List[EarlyStopping],
        stop_on_any: bool = True
    ):
        self.stopping_strategies = stopping_strategies
        self.stop_on_any = stop_on_any
        self.history = EarlyStoppingHistory()
    
    def on_train_begin(self):
        """Initialize all strategies."""
        for strategy in self.stopping_strategies:
            strategy.on_train_begin()
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float], model: Any = None) -> bool:
        """
        Check all stopping strategies.
        """
        stop_signals = []
        
        for strategy in self.stopping_strategies:
            should_stop = strategy.on_epoch_end(epoch, logs, model)
            stop_signals.append(should_stop)
            
            # If any strategy stops and we're in stop_on_any mode
            if should_stop and self.stop_on_any:
                self.stopped_epoch = epoch
                self.history.reason = f"Stopped by {strategy.__class__.__name__}"
                
                # Aggregate histories
                for s in self.stopping_strategies:
                    self.history.metrics.update(s.get_history().metrics)
                
                return True
        
        # If all strategies stop and we're in stop_on_all mode
        if not self.stop_on_any and all(stop_signals):
            self.stopped_epoch = epoch
            self.history.reason = "All strategies triggered"
            
            # Aggregate histories
            for s in self.stopping_strategies:
                self.history.metrics.update(s.get_history().metrics)
            
            return True
        
        return False
    
    def on_train_end(self, model: Any = None):
        """Call on_train_end for all strategies."""
        for strategy in self.stopping_strategies:
            strategy.on_train_end(model)
    
    def get_history(self) -> EarlyStoppingHistory:
        """Get combined history."""
        return self.history


def create_adaptive_early_stopping(
    initial_patience: int = 10,
    max_patience: int = 50,
    improvement_threshold: float = 0.01,
    adaptive_factor: float = 1.5,
    **kwargs
) -> EarlyStopping:
    """
    Create an early stopping strategy with adaptive patience.
    
    Patience increases when model is showing consistent improvement,
    decreases when progress stagnates.
    
    Parameters:
        initial_patience: Starting patience value
        max_patience: Maximum allowed patience
        improvement_threshold: Threshold for significant improvement
        adaptive_factor: Factor to multiply patience by on improvement
        **kwargs: Additional arguments for EarlyStopping
        
    Returns:
        adaptive_stopping: Configured EarlyStopping instance
    """
    
    class AdaptiveEarlyStopping(EarlyStopping):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.initial_patience = self.patience
            self.adaptive_factor = adaptive_factor
            self.max_patience = max_patience
            self.improvement_threshold = improvement_threshold
            self.improvement_history = []
        
        def on_epoch_end(self, epoch: int, logs: Dict[str, float], model: Any = None) -> bool:
            # Get current value
            current = self._get_monitor_value(logs)
            
            if current is not None and self.best_score is not None:
                # Calculate relative improvement
                if self.mode == 'min':
                    improvement = (self.best_score - current) / (self.best_score + 1e-8)
                else:
                    improvement = (current - self.best_score) / (self.best_score + 1e-8)
                
                self.improvement_history.append(improvement)
                
                # Adapt patience based on recent improvements
                recent_improvements = self.improvement_history[-5:] if len(self.improvement_history) >= 5 else self.improvement_history
                
                if recent_improvements:
                    avg_improvement = np.mean(recent_improvements)
                    
                    if avg_improvement > self.improvement_threshold:
                        # Significant improvement - increase patience
                        self.patience = min(int(self.patience * self.adaptive_factor), self.max_patience)
                    elif avg_improvement < self.improvement_threshold / 2:
                        # Minimal improvement - decrease patience
                        self.patience = max(int(self.patience / self.adaptive_factor), self.initial_patience)
            
            return super().on_epoch_end(epoch, logs, model)
    
    return AdaptiveEarlyStopping(
        patience=initial_patience,
        **kwargs
    )


def create_metric_aware_early_stopping(
    primary_metric: str = 'val_f1_score',
    secondary_metrics: List[str] = None,
    secondary_thresholds: Dict[str, float] = None,
    **kwargs
) -> EarlyStopping:
    """
    Create early stopping that considers multiple metrics.
    
    Parameters:
        primary_metric: Main metric to optimize
        secondary_metrics: Additional metrics to monitor
        secondary_thresholds: Minimum thresholds for secondary metrics
        **kwargs: Additional arguments for EarlyStopping
        
    Returns:
        metric_aware_stopping: Configured EarlyStopping instance
    """
    
    class MetricAwareEarlyStopping(EarlyStopping):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.primary_metric = primary_metric
            self.secondary_metrics = secondary_metrics or []
            self.secondary_thresholds = secondary_thresholds or {}
            self.secondary_history = {m: [] for m in self.secondary_metrics}
        
        def on_epoch_end(self, epoch: int, logs: Dict[str, float], model: Any = None) -> bool:
            # First, check secondary metrics thresholds
            secondary_ok = True
            
            for metric in self.secondary_metrics:
                if metric in logs:
                    self.secondary_history[metric].append(logs[metric])
                    
                    # Check if metric meets threshold
                    if metric in self.secondary_thresholds:
                        threshold = self.secondary_thresholds[metric]
                        current_value = logs[metric]
                        
                        # Determine if threshold is met based on metric direction
                        if 'loss' in metric.lower() or 'error' in metric.lower():
                            # Lower is better
                            if current_value > threshold:
                                secondary_ok = False
                        else:
                            # Higher is better
                            if current_value < threshold:
                                secondary_ok = False
            
            if not secondary_ok:
                # Secondary metrics not meeting thresholds, continue training
                if self.verbose:
                    logger.info(f"Secondary metrics below thresholds at epoch {epoch}")
                return False
            
            # Check primary metric for stopping
            return super().on_epoch_end(epoch, logs, model)
    
    return MetricAwareEarlyStopping(
        monitor=primary_metric,
        **kwargs
    )


class TrainingMonitor:
    """
    Monitor training progress and provide real-time feedback.
    
    This is useful for tracking early stopping decisions and
    providing insights into model convergence.
    
    Parameters:
        early_stopping: EarlyStopping instance
        log_interval: int, default=10
            How often to log progress
        plot: bool, default=False
            Whether to create real-time plots
    """
    
    def __init__(
        self,
        early_stopping: EarlyStopping,
        log_interval: int = 10,
        plot: bool = False
    ):
        self.early_stopping = early_stopping
        self.log_interval = log_interval
        self.plot = plot
        self.fig = None
        self.ax = None
        
        if plot:
            self._setup_plot()
    
    def _setup_plot(self):
        """Setup real-time plotting."""
        import matplotlib.pyplot as plt
        
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 6))
        plt.ion()  # Interactive mode
        plt.show()
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float]):
        """Monitor at epoch end."""
        # Log progress
        if epoch % self.log_interval == 0:
            monitor_value = self.early_stopping._get_monitor_value(logs)
            if monitor_value is not None:
                logger.info(
                    f"Epoch {epoch}: {self.early_stopping.monitor} = {monitor_value:.6f}, "
                    f"Best = {self.early_stopping.best_score:.6f}, "
                    f"Wait = {self.early_stopping.wait}/{self.early_stopping.patience}"
                )
        
        # Update plot
        if self.plot:
            self._update_plot(epoch, logs)
    
    def _update_plot(self, epoch: int, logs: Dict[str, float]):
        """Update real-time plot."""
        import matplotlib.pyplot as plt
        
        if self.fig is None:
            return
        
        history = self.early_stopping.get_history()
        
        self.ax.clear()
        
        # Plot all metrics
        for metric_name, values in history.metrics.items():
            self.ax.plot(history.epoch[:len(values)], values, label=metric_name)
        
        # Mark best epoch
        if history.best_epoch > 0:
            self.ax.axvline(x=history.best_epoch, color='r', linestyle='--', 
                           label=f"Best epoch ({history.best_epoch})")
        
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Value')
        self.ax.set_title('Training Progress')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
        plt.draw()
        plt.pause(0.01)
    
    def on_train_end(self):
        """Clean up at training end."""
        if self.plot:
            import matplotlib.pyplot as plt
            plt.ioff()
            plt.show()


# Example usage and testing
if __name__ == "__main__":
    """
    Example demonstrating how to use early stopping.
    """
    
    print("=" * 60)
    print("Early Stopping Examples for Fraud Detection")
    print("=" * 60)
    
    # Simulate training progress
    np.random.seed(42)
    
    # Generate fake training history (loss decreasing, then plateau)
    epochs = 100
    train_loss = 0.5 * np.exp(-np.arange(epochs) / 20) + 0.1 * np.random.randn(epochs)
    val_loss = 0.5 * np.exp(-np.arange(epochs) / 25) + 0.15 * np.random.randn(epochs)
    val_f1 = 0.9 * (1 - np.exp(-np.arange(epochs) / 30)) + 0.05 * np.random.randn(epochs)
    
    # Example 1: Standard early stopping
    print("\n1. Standard Early Stopping:")
    stopper = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=10,
        min_delta=0.01,
        verbose=True
    )
    
    stopper.on_train_begin()
    
    for epoch in range(epochs):
        logs = {
            'train_loss': train_loss[epoch],
            'val_loss': val_loss[epoch],
            'val_f1': val_f1[epoch]
        }
        
        should_stop = stopper.on_epoch_end(epoch, logs)
        
        if should_stop:
            print(f"  Stopped at epoch {epoch}")
            break
    
    history = stopper.get_history()
    print(f"  Best epoch: {history.best_epoch}")
    print(f"  Best score: {history.best_score:.4f}")
    
    # Example 2: Class-aware early stopping
    print("\n2. Class-Aware Early Stopping:")
    stopper2 = ClassAwareEarlyStopping(
        monitor='weighted_class_score',
        mode='max',
        patience=8,
        class_weight=0.8,
        verbose=True
    )
    
    stopper2.on_train_begin()
    
    for epoch in range(epochs):
        # Add class-specific metrics
        logs = {
            'train_loss': train_loss[epoch],
            'val_loss': val_loss[epoch],
            'f1_score_class_1': val_f1[epoch] * (0.9 + 0.1 * np.random.randn()),
            'f1_score_class_0': val_f1[epoch] * (0.95 + 0.05 * np.random.randn())
        }
        
        should_stop = stopper2.on_epoch_end(epoch, logs)
        
        if should_stop:
            print(f"  Stopped at epoch {epoch}")
            break
    
    # Example 3: Composite early stopping
    print("\n3. Composite Early Stopping:")
    stopper3 = CompositeEarlyStopping([
        EarlyStopping(monitor='val_loss', mode='min', patience=15, verbose=False),
        PlateauDetectionEarlyStopping(monitor='val_f1', mode='max', window_size=10, verbose=False)
    ])
    
    stopper3.on_train_begin()
    
    for epoch in range(epochs):
        logs = {
            'train_loss': train_loss[epoch],
            'val_loss': val_loss[epoch],
            'val_f1': val_f1[epoch]
        }
        
        should_stop = stopper3.on_epoch_end(epoch, logs)
        
        if should_stop:
            print(f"  Stopped at epoch {epoch}")
            break
    
    print("\n" + "=" * 60)
    print("Early stopping module loaded successfully!")
    print("=" * 60)