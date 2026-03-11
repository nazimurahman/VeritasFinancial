# =============================================================================
# File: hyperparameter_tuning.py
# Description: Advanced hyperparameter optimization for fraud detection models
# Author: VeritasFinancial DS Team
# Version: 1.0.0
# Last Updated: 2024-01-15
# =============================================================================

"""
HYPERPARAMETER TUNING FOR FRAUD DETECTION
==========================================
This module implements sophisticated hyperparameter optimization strategies
specifically designed for fraud detection problems with imbalanced data.

Key Challenges Addressed:
1. Class Imbalance: Need metrics that account for rare fraud
2. Computational Cost: Large datasets require efficient search
3. Correlated Parameters: Complex interactions between hyperparameters
4. Overfitting Risk: Must validate properly with time-based CV

Optimization Strategies:
1. Bayesian Optimization: Uses past trials to guide search
2. Hyperband: Early stopping for resource efficiency
3. Genetic Algorithms: Evolutionary approach for complex spaces
4. Multi-Fidelity Optimization: Approximate then refine
5. Multi-Objective Tuning: Optimize multiple metrics (precision/recall)

Custom Features:
- Cost-sensitive optimization (fraud costs > false positive costs)
- Time-based cross-validation integration
- GPU-aware tuning
- Distributed optimization support
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Callable, Optional, Tuple, Union
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
import warnings
import logging
import time
import json
from pathlib import Path
from functools import partial

# Optional imports for different optimization backends
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("optuna not installed. Install with: pip install optuna")

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    warnings.warn("scikit-optimize not installed. Install with: pip install scikit-optimize")

try:
    import hyperopt
    from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False
    warnings.warn("hyperopt not installed. Install with: pip install hyperopt")

try:
    import ray
    from ray import tune
    from ray.tune.schedulers import HyperBandScheduler, ASHAScheduler
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    warnings.warn("ray[tune] not installed. Install with: pip install ray[tune]")

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# CUSTOM SCORING FOR FRAUD DETECTION
# =============================================================================

class FraudScorer:
    """
    Custom scoring functions for fraud detection with business costs.
    
    In fraud detection, false negatives (missed fraud) are much more costly
    than false positives (legitimate transactions flagged). This class
    implements cost-sensitive scoring functions.
    
    Cost Matrix:
                    Predicted: No Fraud    Predicted: Fraud
    Actual: No Fraud    0                   cost_fp
    Actual: Fraud       cost_fn             0
    
    Where typically cost_fn = 10-100 * cost_fp
    
    Benefits:
    - Aligns model optimization with business objectives
    - Handles extreme class imbalance
    - Provides interpretable business metrics
    """
    
    def __init__(
        self,
        cost_fn: float = 100.0,  # Cost of missing a fraud
        cost_fp: float = 1.0,    # Cost of false alarm
        threshold: float = 0.5    # Classification threshold
    ):
        """
        Initialize fraud scorer with business costs.
        
        Args:
            cost_fn (float): Cost of false negative (missed fraud)
            cost_fp (float): Cost of false positive (wrongly flagged)
            threshold (float): Probability threshold for classification
        """
        self.cost_fn = cost_fn
        self.cost_fp = cost_fp
        self.threshold = threshold
        
        logger.info(f"Initialized FraudScorer: cost_fn={cost_fn}, cost_fp={cost_fp}")
    
    def cost_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """
        Calculate total cost based on business costs.
        
        Args:
            y_true: True labels (0: non-fraud, 1: fraud)
            y_pred_proba: Predicted probabilities
            
        Returns:
            float: Total cost (lower is better)
        """
        # Convert probabilities to binary predictions
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        # Calculate confusion matrix elements
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        
        # Calculate total cost
        total_cost = fn * self.cost_fn + fp * self.cost_fp
        
        # Normalize by number of samples
        normalized_cost = total_cost / len(y_true)
        
        return normalized_cost
    
    def savings_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """
        Calculate savings compared to doing nothing.
        
        If we do nothing, all frauds are missed: cost = n_fraud * cost_fn
        With model, we have some false positives and some detected frauds.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            float: Savings (higher is better)
        """
        # Baseline cost (no model, approve all transactions)
        n_fraud = np.sum(y_true)
        baseline_cost = n_fraud * self.cost_fn
        
        # Model cost
        model_cost = self.cost_score(y_true, y_pred_proba) * len(y_true)
        
        # Savings
        savings = baseline_cost - model_cost
        
        return savings
    
    def profit_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """
        Calculate profit (negative cost) for maximization.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            float: Profit (higher is better)
        """
        return -self.cost_score(y_true, y_pred_proba)


# =============================================================================
# HYPERPARAMETER SPACE DEFINITIONS
# =============================================================================

class HyperparameterSpace:
    """
    Define hyperparameter spaces for different model types.
    
    This class provides pre-defined search spaces for common fraud detection
    models with sensible ranges for imbalanced data.
    """
    
    @staticmethod
    def xgboost_space(
        use_gpu: bool = False,
        n_jobs: int = -1
    ) -> Dict[str, Any]:
        """
        Hyperparameter space for XGBoost optimized for fraud detection.
        
        Args:
            use_gpu (bool): Whether to use GPU
            n_jobs (int): Number of parallel jobs
            
        Returns:
            Dict: Parameter space
        """
        space = {
            # Tree-specific parameters
            'max_depth': Integer(3, 12),
            'min_child_weight': Real(1, 10, prior='log-uniform'),
            'subsample': Real(0.6, 1.0),
            'colsample_bytree': Real(0.6, 1.0),
            'colsample_bylevel': Real(0.6, 1.0),
            
            # Learning parameters
            'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'n_estimators': Integer(100, 1000),
            
            # Regularization
            'reg_alpha': Real(1e-5, 10, prior='log-uniform'),
            'reg_lambda': Real(1e-5, 10, prior='log-uniform'),
            'gamma': Real(0, 5),
            
            # Class imbalance
            'scale_pos_weight': Real(1, 100, prior='log-uniform'),
            
            # Sampling
            'max_delta_step': Integer(0, 10),
        }
        
        if use_gpu:
            space['tree_method'] = Categorical(['gpu_hist'])
            space['predictor'] = Categorical(['gpu_predictor'])
        
        return space
    
    @staticmethod
    def lightgbm_space(
        use_gpu: bool = False,
        n_jobs: int = -1
    ) -> Dict[str, Any]:
        """
        Hyperparameter space for LightGBM optimized for fraud detection.
        
        Args:
            use_gpu (bool): Whether to use GPU
            n_jobs (int): Number of parallel jobs
            
        Returns:
            Dict: Parameter space
        """
        space = {
            # Tree structure
            'num_leaves': Integer(20, 300),
            'max_depth': Integer(3, 20),
            'min_child_samples': Integer(5, 100),
            'min_child_weight': Real(1e-3, 10, prior='log-uniform'),
            
            # Sampling
            'subsample': Real(0.6, 1.0),
            'subsample_freq': Integer(0, 10),
            'colsample_bytree': Real(0.6, 1.0),
            
            # Learning
            'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'n_estimators': Integer(100, 1000),
            
            # Regularization
            'reg_alpha': Real(1e-5, 10, prior='log-uniform'),
            'reg_lambda': Real(1e-5, 10, prior='log-uniform'),
            
            # Class imbalance
            'scale_pos_weight': Real(1, 100, prior='log-uniform'),
            
            # Histogram binning
            'max_bin': Integer(100, 500),
        }
        
        if use_gpu:
            space['device'] = Categorical(['gpu'])
            space['gpu_platform_id'] = Integer(0, 1)
            space['gpu_device_id'] = Integer(0, 8)
        
        return space
    
    @staticmethod
    def random_forest_space() -> Dict[str, Any]:
        """
        Hyperparameter space for Random Forest.
        
        Returns:
            Dict: Parameter space
        """
        return {
            'n_estimators': Integer(100, 500),
            'max_depth': Integer(5, 50),
            'min_samples_split': Integer(2, 50),
            'min_samples_leaf': Integer(1, 20),
            'max_features': Categorical(['sqrt', 'log2', None]),
            'bootstrap': Categorical([True, False]),
            'class_weight': Categorical(['balanced', 'balanced_subsample', None])
        }
    
    @staticmethod
    def neural_network_space() -> Dict[str, Any]:
        """
        Hyperparameter space for Neural Networks.
        
        Returns:
            Dict: Parameter space
        """
        return {
            'hidden_layer_sizes': Categorical([
                (64,), (128,), (256,),
                (64, 32), (128, 64), (256, 128),
                (128, 64, 32), (256, 128, 64)
            ]),
            'activation': Categorical(['relu', 'tanh', 'elu']),
            'alpha': Real(1e-5, 1e-1, prior='log-uniform'),
            'batch_size': Categorical([32, 64, 128, 256]),
            'learning_rate_init': Real(1e-4, 1e-2, prior='log-uniform'),
            'dropout_rate': Real(0.1, 0.5),
        }


# =============================================================================
# OPTUNA OPTIMIZER
# =============================================================================

class OptunaOptimizer:
    """
    Hyperparameter optimization using Optuna with pruning.
    
    Optuna provides sophisticated optimization algorithms including:
    - Tree-structured Parzen Estimator (TPE)
    - CMA-ES
    - Hyperband pruning
    
    Benefits for Fraud Detection:
    - Efficient search in high-dimensional spaces
    - Automatic pruning of bad trials saves resources
    - Built-in support for multi-objective optimization
    - Distributed optimization support
    """
    
    def __init__(
        self,
        model_class: Any,
        param_space: Dict[str, Any],
        cv: BaseCrossValidator,
        scoring: Callable,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        direction: str = "maximize",
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        pruner: str = "hyperband",
        n_jobs: int = 1,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize Optuna optimizer.
        
        Args:
            model_class: Model class to optimize
            param_space: Parameter space definition
            cv: Cross-validation strategy
            scoring: Scoring function
            n_trials: Number of optimization trials
            timeout: Time limit in seconds
            direction: "maximize" or "minimize"
            study_name: Name for the study
            storage: Database URL for distributed optimization
            pruner: Pruning algorithm ("hyperband", "median", "none")
            n_jobs: Number of parallel jobs
            random_state: Random seed
            verbose: Whether to print progress
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("optuna is required for OptunaOptimizer")
        
        self.model_class = model_class
        self.param_space = param_space
        self.cv = cv
        self.scoring = scoring
        self.n_trials = n_trials
        self.timeout = timeout
        self.direction = direction
        self.study_name = study_name
        self.storage = storage
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        
        # Set up pruner
        self.pruner = self._create_pruner(pruner)
        
        # Results storage
        self.best_params = None
        self.best_score = None
        self.study = None
        self.trials_dataframe = None
        
        logger.info(f"Initialized OptunaOptimizer with {n_trials} trials")
    
    def _create_pruner(self, pruner: str) -> optuna.pruners.BasePruner:
        """
        Create pruning algorithm.
        
        Args:
            pruner: Pruner name
            
        Returns:
            optuna.pruners.BasePruner: Pruner instance
        """
        if pruner == "hyperband":
            return optuna.pruners.HyperbandPruner(
                min_resource=1,
                max_resource=self.n_trials,
                reduction_factor=3
            )
        elif pruner == "median":
            return optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        elif pruner == "none":
            return optuna.pruners.NopPruner()
        else:
            raise ValueError(f"Unknown pruner: {pruner}")
    
    def _objective(self, trial: optuna.Trial, X, y, groups=None) -> float:
        """
        Objective function for Optuna trial.
        
        Args:
            trial: Optuna trial object
            X: Features
            y: Target
            groups: Group labels
            
        Returns:
            float: Score to optimize
        """
        # Sample hyperparameters
        params = {}
        
        for param_name, param_spec in self.param_space.items():
            if isinstance(param_spec, Integer):
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_spec.low,
                    param_spec.high,
                    log=param_spec.prior == 'log-uniform'
                )
            elif isinstance(param_spec, Real):
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_spec.low,
                    param_spec.high,
                    log=param_spec.prior == 'log-uniform'
                )
            elif isinstance(param_spec, Categorical):
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_spec.categories
                )
            else:
                # Direct value (not a search space)
                params[param_name] = param_spec
        
        # Cross-validation scores
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X, y, groups)):
            # Report intermediate value for pruning
            trial.report(fold, step=fold)
            
            # Check if we should prune
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # Split data
            if isinstance(X, pd.DataFrame):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            else:
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model = self.model_class(**params)
            
            try:
                model.fit(X_train, y_train)
                
                # Predict
                if hasattr(model, "predict_proba"):
                    y_pred = model.predict_proba(X_val)[:, 1]
                else:
                    y_pred = model.predict(X_val)
                
                # Score
                score = self.scoring(y_val, y_pred)
                cv_scores.append(score)
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return float('-inf') if self.direction == "maximize" else float('inf')
        
        # Return mean score
        return np.mean(cv_scores)
    
    def optimize(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        groups: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            X: Features
            y: Target
            groups: Group labels
            
        Returns:
            Dict: Best parameters
        """
        # Create study
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction=self.direction,
            sampler=sampler,
            pruner=self.pruner,
            load_if_exists=True
        )
        
        # Create objective with data
        objective = partial(self._objective, X=X, y=y, groups=groups)
        
        # Run optimization
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=self.verbose
        )
        
        # Store results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        self.trials_dataframe = self.study.trials_dataframe()
        
        if self.verbose:
            print(f"\nBest score: {self.best_score:.4f}")
            print(f"Best parameters: {self.best_params}")
        
        return self.best_params
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """
        Plot optimization history.
        
        Args:
            save_path: Path to save plot
        """
        if not OPTUNA_AVAILABLE:
            return
        
        try:
            from optuna.visualization import plot_optimization_history
            fig = plot_optimization_history(self.study)
            
            if save_path:
                fig.write_image(save_path)
            else:
                fig.show()
                
        except Exception as e:
            logger.warning(f"Could not plot optimization history: {e}")
    
    def plot_param_importances(self, save_path: Optional[str] = None):
        """
        Plot parameter importances.
        
        Args:
            save_path: Path to save plot
        """
        if not OPTUNA_AVAILABLE:
            return
        
        try:
            from optuna.visualization import plot_param_importances
            fig = plot_param_importances(self.study)
            
            if save_path:
                fig.write_image(save_path)
            else:
                fig.show()
                
        except Exception as e:
            logger.warning(f"Could not plot param importances: {e}")


# =============================================================================
# HYPEROPT OPTIMIZER
# =============================================================================

class HyperoptOptimizer:
    """
    Hyperparameter optimization using Hyperopt.
    
    Hyperopt uses Tree-structured Parzen Estimator (TPE) for efficient
    search in complex spaces.
    """
    
    def __init__(
        self,
        model_class: Any,
        param_space: Dict[str, Any],
        cv: BaseCrossValidator,
        scoring: Callable,
        max_evals: int = 100,
        timeout: Optional[int] = None,
        algo: str = "tpe",
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize Hyperopt optimizer.
        """
        if not HYPEROPT_AVAILABLE:
            raise ImportError("hyperopt is required for HyperoptOptimizer")
        
        self.model_class = model_class
        self.param_space = self._convert_space(param_space)
        self.cv = cv
        self.scoring = scoring
        self.max_evals = max_evals
        self.timeout = timeout
        self.random_state = random_state
        self.verbose = verbose
        
        # Algorithm selection
        if algo == "tpe":
            self.algo = hyperopt.tpe.suggest
        elif algo == "random":
            self.algo = hyperopt.rand.suggest
        else:
            raise ValueError(f"Unknown algorithm: {algo}")
        
        # Results
        self.best_params = None
        self.best_loss = None
        self.trials = None
    
    def _convert_space(self, space: Dict) -> Dict:
        """
        Convert scikit-optimize space to hyperopt space.
        
        Args:
            space: Parameter space in skopt format
            
        Returns:
            Dict: Space in hyperopt format
        """
        hyperopt_space = {}
        
        for name, spec in space.items():
            if isinstance(spec, Integer):
                hyperopt_space[name] = hp.quniform(
                    name,
                    spec.low,
                    spec.high,
                    1
                )
            elif isinstance(spec, Real):
                if spec.prior == 'log-uniform':
                    hyperopt_space[name] = hp.loguniform(
                        name,
                        np.log(spec.low),
                        np.log(spec.high)
                    )
                else:
                    hyperopt_space[name] = hp.uniform(
                        name,
                        spec.low,
                        spec.high
                    )
            elif isinstance(spec, Categorical):
                hyperopt_space[name] = hp.choice(
                    name,
                    spec.categories
                )
            else:
                hyperopt_space[name] = spec
        
        return hyperopt_space
    
    def _objective(self, params: Dict, X, y, groups) -> Dict:
        """
        Objective function for Hyperopt.
        
        Args:
            params: Hyperparameters
            X: Features
            y: Target
            groups: Group labels
            
        Returns:
            Dict: Results with loss and status
        """
        # Cross-validation scores
        cv_scores = []
        
        for train_idx, val_idx in self.cv.split(X, y, groups):
            # Split data
            if isinstance(X, pd.DataFrame):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            else:
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model = self.model_class(**params)
            
            try:
                model.fit(X_train, y_train)
                
                # Predict
                if hasattr(model, "predict_proba"):
                    y_pred = model.predict_proba(X_val)[:, 1]
                else:
                    y_pred = model.predict(X_val)
                
                # Score (negate for minimization)
                score = self.scoring(y_val, y_pred)
                cv_scores.append(-score)  # Hyperopt minimizes
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return {'loss': float('inf'), 'status': STATUS_OK}
        
        return {
            'loss': np.mean(cv_scores),
            'status': STATUS_OK
        }
    
    def optimize(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        groups: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            X: Features
            y: Target
            groups: Group labels
            
        Returns:
            Dict: Best parameters
        """
        # Create objective with data
        objective = partial(self._objective, X=X, y=y, groups=groups)
        
        # Run optimization
        self.trials = Trials()
        
        self.best_params = fmin(
            fn=objective,
            space=self.param_space,
            algo=self.algo,
            max_evals=self.max_evals,
            trials=self.trials,
            rstate=np.random.RandomState(self.random_state),
            verbose=self.verbose
        )
        
        # Get best loss
        self.best_loss = min(self.trials.losses()) if self.trials.losses() else None
        
        if self.verbose:
            print(f"\nBest loss: {self.best_loss:.4f}")
            print(f"Best parameters: {self.best_params}")
        
        return self.best_params


# =============================================================================
# RAY TUNE OPTIMIZER
# =============================================================================

class RayTuneOptimizer:
    """
    Distributed hyperparameter optimization using Ray Tune.
    
    Ray Tune provides:
    - Distributed optimization across multiple machines
    - Advanced schedulers (HyperBand, ASHA, Population Based Training)
    - Integration with many frameworks
    """
    
    def __init__(
        self,
        model_class: Any,
        param_space: Dict[str, Any],
        cv: BaseCrossValidator,
        scoring: Callable,
        num_samples: int = 100,
        time_budget_s: Optional[int] = None,
        scheduler: str = "asha",
        resources_per_trial: Dict = {"cpu": 1},
        num_workers: int = 1,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize Ray Tune optimizer.
        """
        if not RAY_AVAILABLE:
            raise ImportError("ray[tune] is required for RayTuneOptimizer")
        
        self.model_class = model_class
        self.param_space = param_space
        self.cv = cv
        self.scoring = scoring
        self.num_samples = num_samples
        self.time_budget_s = time_budget_s
        self.resources_per_trial = resources_per_trial
        self.num_workers = num_workers
        self.random_state = random_state
        self.verbose = verbose
        
        # Scheduler
        self.scheduler = self._create_scheduler(scheduler)
        
        # Results
        self.best_config = None
        self.best_result = None
        self.analysis = None
    
    def _create_scheduler(self, scheduler: str) -> tune.schedulers.TrialScheduler:
        """
        Create Ray Tune scheduler.
        """
        if scheduler == "asha":
            return ASHAScheduler(
                time_attr="training_iteration",
                max_t=100,
                grace_period=10,
                reduction_factor=3,
                brackets=1
            )
        elif scheduler == "hyperband":
            return HyperBandScheduler(
                time_attr="training_iteration",
                max_t=100,
                reduction_factor=3,
                stop_last_trials=False
            )
        else:
            return None
    
    def _trainable(self, config: Dict, X=None, y=None, groups=None):
        """
        Trainable function for Ray Tune.
        """
        # Cross-validation scores
        cv_scores = []
        
        for train_idx, val_idx in self.cv.split(X, y, groups):
            # Split data
            if isinstance(X, pd.DataFrame):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            else:
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model = self.model_class(**config)
            
            try:
                model.fit(X_train, y_train)
                
                # Predict
                if hasattr(model, "predict_proba"):
                    y_pred = model.predict_proba(X_val)[:, 1]
                else:
                    y_pred = model.predict(X_val)
                
                # Score
                score = self.scoring(y_val, y_pred)
                cv_scores.append(score)
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                tune.report(score=float('-inf'))
                return
        
        # Report score
        tune.report(score=np.mean(cv_scores))
    
    def optimize(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        groups: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> Dict[str, Any]:
        """
        Run distributed hyperparameter optimization.
        """
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(num_cpus=self.num_workers, ignore_reinit_error=True)
        
        # Create trainable with data
        trainable = tune.with_parameters(
            self._trainable,
            X=X,
            y=y,
            groups=groups
        )
        
        # Run tuning
        self.analysis = tune.run(
            trainable,
            config=self.param_space,
            num_samples=self.num_samples,
            time_budget_s=self.time_budget_s,
            scheduler=self.scheduler,
            resources_per_trial=self.resources_per_trial,
            metric="score",
            mode="max",
            verbose=int(self.verbose),
            name="fraud_tuning"
        )
        
        # Get best config
        self.best_config = self.analysis.get_best_config(metric="score", mode="max")
        self.best_result = self.analysis.get_best_trial(metric="score", mode="max")
        
        if self.verbose:
            print(f"\nBest score: {self.best_result.metric_analysis['score']['mean']:.4f}")
            print(f"Best config: {self.best_config}")
        
        return self.best_config


# =============================================================================
# HYPERPARAMETER TUNING MANAGER
# =============================================================================

class HyperparameterTuningManager:
    """
    Unified manager for hyperparameter tuning with multiple backends.
    
    This class provides a single interface to different optimization
    algorithms and includes utilities for:
    - Grid search
    - Random search
    - Bayesian optimization
    - Early stopping
    - Result analysis
    
    Benefits for Fraud Detection:
    - Handles class imbalance with custom scoring
    - Integrates with time-based CV
    - Provides parallel/distributed optimization
    - Saves and loads tuning results
    """
    
    def __init__(
        self,
        model_class: Any,
        param_grid: Optional[Dict] = None,
        param_space: Optional[Dict] = None,
        cv: Optional[BaseCrossValidator] = None,
        scoring: Union[str, Callable] = 'f1',
        n_iter: int = 100,
        cv_strategy: str = "stratified_kfold",
        n_splits: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: bool = True,
        optimization_backend: str = "optuna",  # "grid", "random", "optuna", "hyperopt", "ray"
        distributed: bool = False
    ):
        """
        Initialize hyperparameter tuning manager.
        
        Args:
            model_class: Model class to optimize
            param_grid: Grid for grid search
            param_space: Space for random/Bayesian search
            cv: Cross-validator (if None, creates default)
            scoring: Scoring metric
            n_iter: Number of iterations for random/Bayesian search
            cv_strategy: Cross-validation strategy
            n_splits: Number of CV folds
            random_state: Random seed
            n_jobs: Number of parallel jobs
            verbose: Whether to print progress
            optimization_backend: Optimization algorithm
            distributed: Whether to use distributed optimization
        """
        self.model_class = model_class
        self.param_grid = param_grid
        self.param_space = param_space
        self.scoring = scoring
        self.n_iter = n_iter
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.optimization_backend = optimization_backend
        self.distributed = distributed
        
        # Set up cross-validation
        if cv is None:
            from src.modeling.training.cross_validation import CrossValidationManager
            cv_manager = CrossValidationManager(
                cv_strategy=cv_strategy,
                n_splits=n_splits,
                random_state=random_state
            )
            self.cv = cv_manager.splitter
        else:
            self.cv = cv
        
        # Results storage
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = None
        self.optimizer = None
    
    def _get_scorer(self) -> Callable:
        """
        Get scoring function.
        
        Returns:
            Callable: Scoring function
        """
        if isinstance(self.scoring, str):
            if self.scoring == 'f1':
                return make_scorer(f1_score)
            elif self.scoring == 'precision':
                return make_scorer(precision_score)
            elif self.scoring == 'recall':
                return make_scorer(recall_score)
            elif self.scoring == 'roc_auc':
                return make_scorer(roc_auc_score, needs_proba=True)
            elif self.scoring == 'pr_auc':
                return make_scorer(average_precision_score, needs_proba=True)
            elif self.scoring == 'cost':
                scorer = FraudScorer(cost_fn=100, cost_fp=1)
                return scorer.cost_score
            else:
                raise ValueError(f"Unknown scoring metric: {self.scoring}")
        else:
            return self.scoring
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        groups: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> 'HyperparameterTuningManager':
        """
        Run hyperparameter optimization.
        
        Args:
            X: Features
            y: Target
            groups: Group labels
            
        Returns:
            self: Fitted manager
        """
        # Get scorer
        scorer = self._get_scorer()
        
        # Convert pandas to numpy if needed
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = X
        
        if isinstance(y, pd.Series):
            y_np = y.values
        else:
            y_np = y
        
        # Choose optimization backend
        if self.optimization_backend == "grid":
            from sklearn.model_selection import GridSearchCV
            self.optimizer = GridSearchCV(
                estimator=self.model_class(),
                param_grid=self.param_grid,
                scoring=scorer,
                cv=self.cv,
                n_jobs=self.n_jobs,
                verbose=self.verbose
            )
            self.optimizer.fit(X_np, y_np, groups=groups)
            
        elif self.optimization_backend == "random":
            from sklearn.model_selection import RandomizedSearchCV
            self.optimizer = RandomizedSearchCV(
                estimator=self.model_class(),
                param_distributions=self.param_space,
                n_iter=self.n_iter,
                scoring=scorer,
                cv=self.cv,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=self.verbose
            )
            self.optimizer.fit(X_np, y_np, groups=groups)
            
        elif self.optimization_backend == "optuna":
            if not OPTUNA_AVAILABLE:
                raise ImportError("optuna not installed")
            
            self.optimizer = OptunaOptimizer(
                model_class=self.model_class,
                param_space=self.param_space,
                cv=self.cv,
                scoring=scorer,
                n_trials=self.n_iter,
                random_state=self.random_state,
                n_jobs=self.n_jobs if not self.distributed else 1,
                verbose=self.verbose
            )
            self.optimizer.optimize(X_np, y_np, groups)
            
            self.best_params_ = self.optimizer.best_params
            self.best_score_ = self.optimizer.best_score
            
        elif self.optimization_backend == "hyperopt":
            if not HYPEROPT_AVAILABLE:
                raise ImportError("hyperopt not installed")
            
            self.optimizer = HyperoptOptimizer(
                model_class=self.model_class,
                param_space=self.param_space,
                cv=self.cv,
                scoring=scorer,
                max_evals=self.n_iter,
                random_state=self.random_state,
                verbose=self.verbose
            )
            self.optimizer.optimize(X_np, y_np, groups)
            
            self.best_params_ = self.optimizer.best_params
            self.best_score_ = -self.optimizer.best_loss if self.optimizer.best_loss else None
            
        elif self.optimization_backend == "ray":
            if not RAY_AVAILABLE:
                raise ImportError("ray[tune] not installed")
            
            self.optimizer = RayTuneOptimizer(
                model_class=self.model_class,
                param_space=self.param_space,
                cv=self.cv,
                scoring=scorer,
                num_samples=self.n_iter,
                random_state=self.random_state,
                num_workers=self.n_jobs if self.distributed else 1,
                verbose=self.verbose
            )
            self.optimizer.optimize(X_np, y_np, groups)
            
            self.best_params_ = self.optimizer.best_config
            self.best_score_ = self.optimizer.best_result.metric_analysis['score']['mean'] if self.optimizer.best_result else None
        
        else:
            raise ValueError(f"Unknown backend: {self.optimization_backend}")
        
        # Extract results for sklearn-based searches
        if hasattr(self.optimizer, 'best_params_'):
            self.best_params_ = self.optimizer.best_params_
            self.best_score_ = self.optimizer.best_score_
            self.cv_results_ = self.optimizer.cv_results_
        
        return self
    
    def get_best_model(self) -> Any:
        """
        Get best model with optimal hyperparameters.
        
        Returns:
            Any: Model with best parameters
        """
        if self.best_params_ is None:
            raise ValueError("Must call fit() first")
        
        return self.model_class(**self.best_params_)
    
    def save_results(self, path: str):
        """
        Save tuning results to file.
        
        Args:
            path: Path to save results
        """
        results = {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'cv_results': self.cv_results_,
            'optimization_backend': self.optimization_backend,
            'n_iter': self.n_iter
        }
        
        with open(path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {path}")
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Plot optimization results.
        
        Args:
            save_path: Path to save plot
        """
        if self.optimization_backend == "optuna" and hasattr(self.optimizer, 'plot_optimization_history'):
            self.optimizer.plot_optimization_history(save_path)
            self.optimizer.plot_param_importances(save_path)
        
        elif self.cv_results_ is not None:
            # Plot for sklearn-based searches
            import matplotlib.pyplot as plt
            
            results_df = pd.DataFrame(self.cv_results_)
            
            # Plot parameter importance
            param_cols = [col for col in results_df.columns if col.startswith('param_')]
            
            if len(param_cols) > 0:
                fig, axes = plt.subplots(len(param_cols), 1, figsize=(10, 4*len(param_cols)))
                
                if len(param_cols) == 1:
                    axes = [axes]
                
                for i, param in enumerate(param_cols):
                    param_name = param.replace('param_', '')
                    
                    # Group by parameter value
                    grouped = results_df.groupby(param)['mean_test_score'].agg(['mean', 'std']).reset_index()
                    
                    axes[i].errorbar(
                        grouped[param].astype(str),
                        grouped['mean'],
                        yerr=grouped['std'],
                        fmt='o-',
                        capsize=5
                    )
                    axes[i].set_xlabel(param_name)
                    axes[i].set_ylabel('Score')
                    axes[i].set_title(f'Effect of {param_name} on Performance')
                    axes[i].grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(save_path)
                else:
                    plt.show()


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    """
    Example usage of hyperparameter tuning for fraud detection.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Generate imbalanced dataset (5% fraud)
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        weights=[0.95, 0.05],
        random_state=42
    )
    
    # Create groups (customers)
    groups = np.random.randint(0, 100, 10000)
    
    print("="*60)
    print("HYPERPARAMETER TUNING FOR FRAUD DETECTION")
    print("="*60)
    
    # 1. Define parameter space
    param_space = {
        'n_estimators': Integer(50, 200),
        'max_depth': Integer(3, 15),
        'min_samples_split': Integer(2, 20),
        'min_samples_leaf': Integer(1, 10),
        'max_features': Categorical(['sqrt', 'log2']),
        'class_weight': Categorical(['balanced', 'balanced_subsample'])
    }
    
    # 2. Grid Search
    print("\n1. Running Grid Search...")
    grid_manager = HyperparameterTuningManager(
        model_class=RandomForestClassifier,
        param_grid={
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'class_weight': ['balanced', None]
        },
        cv_strategy="stratified_kfold",
        n_splits=3,
        scoring='f1',
        optimization_backend="grid",
        n_jobs=-1,
        verbose=False
    )
    grid_manager.fit(X, y, groups)
    print(f"Best F1: {grid_manager.best_score_:.4f}")
    print(f"Best params: {grid_manager.best_params_}")
    
    # 3. Random Search
    print("\n2. Running Random Search...")
    random_manager = HyperparameterTuningManager(
        model_class=RandomForestClassifier,
        param_space=param_space,
        cv_strategy="stratified_kfold",
        n_splits=3,
        scoring='f1',
        n_iter=20,
        optimization_backend="random",
        n_jobs=-1,
        verbose=False
    )
    random_manager.fit(X, y, groups)
    print(f"Best F1: {random_manager.best_score_:.4f}")
    print(f"Best params: {random_manager.best_params_}")
    
    # 4. Bayesian Optimization (Optuna)
    if OPTUNA_AVAILABLE:
        print("\n3. Running Bayesian Optimization with Optuna...")
        optuna_manager = HyperparameterTuningManager(
            model_class=RandomForestClassifier,
            param_space=param_space,
            cv_strategy="stratified_kfold",
            n_splits=3,
            scoring='f1',
            n_iter=20,
            optimization_backend="optuna",
            n_jobs=1,  # Optuna handles parallelism differently
            verbose=False
        )
        optuna_manager.fit(X, y, groups)
        print(f"Best F1: {optuna_manager.best_score_:.4f}")
        print(f"Best params: {optuna_manager.best_params_}")
    
    # 5. Cost-sensitive optimization
    print("\n4. Running Cost-Sensitive Optimization...")
    cost_manager = HyperparameterTuningManager(
        model_class=RandomForestClassifier,
        param_space=param_space,
        cv_strategy="stratified_kfold",
        n_splits=3,
        scoring='cost',  # Custom cost-based scoring
        n_iter=20,
        optimization_backend="random",
        n_jobs=-1,
        verbose=False
    )
    cost_manager.fit(X, y, groups)
    print(f"Best params: {cost_manager.best_params_}")
    
    # 6. Compare results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    results = {
        'Grid Search': grid_manager.best_score_,
        'Random Search': random_manager.best_score_,
    }
    
    if OPTUNA_AVAILABLE:
        results['Optuna'] = optuna_manager.best_score_
    
    for method, score in results.items():
        print(f"{method:15s}: F1 = {score:.4f}")