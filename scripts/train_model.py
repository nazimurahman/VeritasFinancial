#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VeritasFinancial - Banking Fraud Detection System
Model Training Script
==================================================
This script handles the complete model training pipeline:
1. Data loading and validation
2. Model architecture selection and initialization
3. Hyperparameter tuning
4. Model training with advanced techniques
5. Model validation and evaluation
6. Model versioning and storage

Author: VeritasFinancial Data Science Team
Version: 1.0.0
"""

# ============================================================================
# IMPORTS SECTION
# ============================================================================
# Standard library imports
import os
import sys
import json
import yaml
import logging
import argparse
import pickle
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
warnings.filterwarnings('ignore')

# Data manipulation
import numpy as np
import pandas as pd

# Machine Learning
from sklearn.model_selection import (
    train_test_split, 
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    cross_validate
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve
)

# Classical ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Advanced ML Models
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import torch.nn.functional as F

# Imbalanced Learning
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.combine import SMOTETomek

# Feature selection
from sklearn.feature_selection import (
    SelectKBest,
    SelectFromModel,
    RFE,
    RFECV,
    mutual_info_classif,
    f_classif
)

# Hyperparameter optimization
import optuna
from optuna.samplers import TPESampler, RandomSampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner

# Model interpretability
import shap
import eli5
from lime import lime_tabular

# Experiment tracking
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.pytorch

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Project imports
from src.models.classical_ml import (
    XGBoostModel,
    LightGBMModel,
    RandomForestModel,
    EnsembleMethods
)
from src.models.deep_learning import (
    NeuralNetwork,
    AutoencoderModel,
    LSTMModel,
    TransformerModel
)
from src.models.training import (
    CrossValidator,
    HyperparameterTuner,
    EarlyStopping,
    ModelCheckpoint
)
from src.models.evaluation import (
    MetricsCalculator,
    ThresholdOptimizer,
    ModelInterpreter,
    BusinessMetrics
)
from src.utils import (
    Logger,
    ConfigManager,
    ModelSerializer,
    ExperimentTracker
)


# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

class TrainingConfig:
    """
    Configuration manager for model training.
    
    Handles:
    - Model architecture configuration
    - Training hyperparameters
    - Data configuration
    - Experiment tracking settings
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize training configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_default_config()
        
        if config_path and os.path.exists(config_path):
            user_config = self._load_config_file(config_path)
            self._deep_update(self.config, user_config)
        
        self._validate_config()
        self._setup_directories()
    
    def _load_default_config(self) -> Dict:
        """
        Load default training configuration.
        
        Returns:
            Dictionary with default configuration
        """
        return {
            'experiment': {
                'name': 'fraud_detection_experiment',
                'tracking_uri': './mlruns',
                'artifact_location': './artifacts',
                'tags': {
                    'project': 'fraud_detection',
                    'environment': 'development',
                    'team': 'data_science'
                }
            },
            'data': {
                'features_path': 'data/features/',
                'target_column': 'is_fraud',
                'test_size': 0.2,
                'validation_size': 0.15,
                'random_state': 42,
                'stratify': True,
                'sampling_strategy': None  # None, 'smote', 'adasyn', 'undersample'
            },
            'models': {
                'classical_ml': {
                    'enabled': True,
                    'algorithms': ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'catboost'],
                    'ensemble': True,
                    'stacking': True
                },
                'deep_learning': {
                    'enabled': True,
                    'architectures': ['neural_network', 'autoencoder', 'lstm', 'transformer'],
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                    'batch_size': 256,
                    'epochs': 100,
                    'learning_rate': 0.001
                }
            },
            'training': {
                'cv_folds': 5,
                'cv_strategy': 'stratified_kfold',
                'scoring_metrics': ['f1', 'roc_auc', 'average_precision'],
                'early_stopping_patience': 10,
                'early_stopping_metric': 'val_loss',
                'class_weight': 'balanced'
            },
            'hyperparameter_tuning': {
                'enabled': True,
                'method': 'optuna',  # grid, random, optuna, bayesian
                'n_trials': 100,
                'n_jobs': -1,
                'timeout': 3600,  # 1 hour
                'direction': 'maximize',  # maximize or minimize
                'metric': 'f1'
            },
            'evaluation': {
                'threshold_optimization': True,
                'threshold_metric': 'f1',  # f1, precision, recall, business_cost
                'business_constants': {
                    'fraud_cost': 100,
                    'false_positive_cost': 1,
                    'investigation_cost': 5
                },
                'generate_reports': True,
                'visualizations': True
            },
            'model_registry': {
                'save_best_only': True,
                'max_models': 10,
                'versioning': True,
                'model_card': True
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/training.log',
                'mlflow_tracking': True
            }
        }
    
    def _load_config_file(self, config_path: str) -> Dict:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary with configuration
        """
        try:
            ext = Path(config_path).suffix.lower()
            with open(config_path, 'r') as f:
                if ext in ['.yaml', '.yml']:
                    return yaml.safe_load(f)
                elif ext == '.json':
                    return json.load(f)
                else:
                    self.logger.warning(f"Unsupported config format: {ext}")
                    return {}
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return {}
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> Dict:
        """
        Deep update nested dictionary.
        
        Args:
            base_dict: Base dictionary to update
            update_dict: Dictionary with updates
            
        Returns:
            Updated dictionary
        """
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
        return base_dict
    
    def _validate_config(self):
        """Validate configuration values."""
        required_keys = ['experiment', 'data', 'models', 'training']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
    
    def _setup_directories(self):
        """Create necessary directories."""
        directories = [
            self.config['experiment']['artifact_location'],
            'logs',
            'models',
            'reports/training',
            'reports/evaluation'
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default=None):
        """Get configuration value by dot notation."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value


# ============================================================================
# DATA LOADER CLASS
# ============================================================================

class DataLoader:
    """
    Handles data loading and preparation for model training.
    
    Features:
    - Load preprocessed data
    - Create train/validation/test splits
    - Handle class imbalance
    - Create PyTorch DataLoaders for deep learning
    - Data validation and profiling
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize data loader.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data = {}
        self.metadata = {}
    
    def load_data(self, data_path: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load features and target data.
        
        Args:
            data_path: Path to data file (if None, uses config)
            
        Returns:
            Tuple of (X, y)
        """
        self.logger.info("Loading data for training...")
        
        if data_path is None:
            data_path = self.config.get('data.features_path')
        
        # Try to load from various possible file formats
        possible_files = [
            Path(data_path) / 'features_data.parquet',
            Path(data_path) / 'features_data.csv',
            Path(data_path) / 'data_splits.pkl'
        ]
        
        for file_path in possible_files:
            if file_path.exists():
                self.logger.info(f"Loading data from: {file_path}")
                
                if file_path.suffix == '.parquet':
                    df = pd.read_parquet(file_path)
                elif file_path.suffix == '.csv':
                    df = pd.read_csv(file_path)
                elif file_path.suffix == '.pkl':
                    with open(file_path, 'rb') as f:
                        splits = pickle.load(f)
                        # If it's already split, return the training data
                        if all(k in splits for k in ['X_train', 'y_train']):
                            self.data = splits
                            return splits['X_train'], splits['y_train']
                        else:
                            df = splits  # Assume it's a DataFrame
        
        # If we have a DataFrame, split features and target
        target_col = self.config.get('data.target_column', 'is_fraud')
        
        if target_col in df.columns:
            X = df.drop(columns=[target_col])
            y = df[target_col]
        else:
            # Assume last column is target
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
        
        self.logger.info(f"Data loaded: X shape {X.shape}, y shape {y.shape}")
        self.logger.info(f"Class distribution:\n{y.value_counts()}")
        
        # Store metadata
        self.metadata = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'feature_names': X.columns.tolist(),
            'class_distribution': y.value_counts().to_dict(),
            'data_types': X.dtypes.astype(str).to_dict()
        }
        
        return X, y
    
    def create_splits(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Create train/validation/test splits.
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Dictionary with data splits
        """
        self.logger.info("Creating data splits...")
        
        test_size = self.config.get('data.test_size', 0.2)
        val_size = self.config.get('data.validation_size', 0.15)
        random_state = self.config.get('data.random_state', 42)
        stratify = y if self.config.get('data.stratify', True) else None
        
        # First split: train + temp (validation + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=test_size + val_size,
            random_state=random_state,
            stratify=stratify
        )
        
        # Second split: validation and test from temp
        # Adjust test size to be relative to temp
        relative_test_size = test_size / (test_size + val_size)
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=relative_test_size,
            random_state=random_state,
            stratify=y_temp
        )
        
        self.logger.info(f"Train size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
        self.logger.info(f"Validation size: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
        self.logger.info(f"Test size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
        
        splits = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'metadata': {
                'train_distribution': y_train.value_counts().to_dict(),
                'val_distribution': y_val.value_counts().to_dict(),
                'test_distribution': y_test.value_counts().to_dict()
            }
        }
        
        self.data = splits
        return splits
    
    def apply_sampling(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply sampling strategy for imbalanced data.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Resampled X_train, y_train
        """
        strategy = self.config.get('data.sampling_strategy')
        
        if strategy is None:
            self.logger.info("No sampling applied")
            return X_train, y_train
        
        self.logger.info(f"Applying sampling strategy: {strategy}")
        
        # Check if sampling is needed
        class_counts = y_train.value_counts()
        imbalance_ratio = class_counts.max() / class_counts.min()
        
        if imbalance_ratio < 2:
            self.logger.info(f"Imbalance ratio {imbalance_ratio:.2f}:1 is acceptable, skipping sampling")
            return X_train, y_train
        
        if strategy == 'smote':
            sampler = SMOTE(random_state=42)
        elif strategy == 'adasyn':
            sampler = ADASYN(random_state=42)
        elif strategy == 'undersample':
            sampler = RandomUnderSampler(random_state=42)
        elif strategy == 'smote_tomek':
            sampler = SMOTETomek(random_state=42)
        else:
            self.logger.warning(f"Unknown sampling strategy: {strategy}")
            return X_train, y_train
        
        # Apply sampling
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        
        self.logger.info(f"After sampling - X shape: {X_resampled.shape}")
        self.logger.info(f"Class distribution:\n{y_resampled.value_counts()}")
        
        return X_resampled, y_resampled
    
    def create_dataloaders(self, X_train, y_train, X_val, y_val, X_test, y_test) -> Dict[str, DataLoader]:
        """
        Create PyTorch DataLoaders for deep learning.
        
        Args:
            X_train, y_train, X_val, y_val, X_test, y_test: Data splits
            
        Returns:
            Dictionary of DataLoaders
        """
        self.logger.info("Creating PyTorch DataLoaders...")
        
        batch_size = self.config.get('models.deep_learning.batch_size', 256)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train.values)
        y_train_tensor = torch.LongTensor(y_train.values)
        X_val_tensor = torch.FloatTensor(X_val.values)
        y_val_tensor = torch.LongTensor(y_val.values)
        X_test_tensor = torch.FloatTensor(X_test.values)
        y_test_tensor = torch.LongTensor(y_test.values)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # Create weighted sampler for imbalanced data
        class_counts = y_train.value_counts()
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_train].values
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        # Create dataloaders
        dataloaders = {
            'train': DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=4,
                pin_memory=True
            ),
            'val': DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            ),
            'test': DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        }
        
        self.logger.info(f"DataLoaders created with batch size {batch_size}")
        
        return dataloaders


# ============================================================================
# MODEL FACTORY CLASS
# ============================================================================

class ModelFactory:
    """
    Factory class for creating different model types.
    
    Creates:
    - Classical ML models (sklearn, xgboost, lightgbm)
    - Deep learning models (PyTorch)
    - Ensemble models
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize model factory.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
    
    def create_model(self, model_type: str, **kwargs) -> Any:
        """
        Create a model instance.
        
        Args:
            model_type: Type of model to create
            **kwargs: Additional arguments for model
            
        Returns:
            Model instance
        """
        self.logger.info(f"Creating model: {model_type}")
        
        # Get number of features for neural networks
        n_features = kwargs.get('n_features', 100)
        
        if model_type == 'logistic_regression':
            model = LogisticRegression(
                class_weight=self.config.get('training.class_weight', 'balanced'),
                random_state=42,
                max_iter=1000,
                n_jobs=-1,
                **kwargs
            )
        
        elif model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight=self.config.get('training.class_weight', 'balanced'),
                random_state=42,
                n_jobs=-1,
                **kwargs
            )
        
        elif model_type == 'xgboost':
            # Calculate scale_pos_weight for imbalance
            class_counts = kwargs.get('class_counts', {0: 1000, 1: 10})
            scale_pos_weight = class_counts.get(0, 1) / class_counts.get(1, 1)
            
            model = xgb.XGBClassifier(
                objective='binary:logistic',
                scale_pos_weight=scale_pos_weight,
                eval_metric='aucpr',
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1,
                **kwargs
            )
        
        elif model_type == 'lightgbm':
            model = lgb.LGBMClassifier(
                objective='binary',
                class_weight=self.config.get('training.class_weight', 'balanced'),
                random_state=42,
                n_jobs=-1,
                verbose=-1,
                **kwargs
            )
        
        elif model_type == 'catboost':
            model = cb.CatBoostClassifier(
                loss_function='Logloss',
                class_weights=self.config.get('training.class_weight', 'balanced'),
                random_seed=42,
                verbose=0,
                **kwargs
            )
        
        elif model_type == 'neural_network':
            model = NeuralNetwork(
                input_size=n_features,
                hidden_sizes=[256, 128, 64],
                dropout_rate=0.3,
                **kwargs
            )
        
        elif model_type == 'autoencoder':
            model = AutoencoderModel(
                input_size=n_features,
                encoding_dim=32,
                **kwargs
            )
        
        elif model_type == 'lstm':
            model = LSTMModel(
                input_size=n_features,
                hidden_size=128,
                num_layers=2,
                **kwargs
            )
        
        elif model_type == 'transformer':
            model = TransformerModel(
                input_size=n_features,
                d_model=128,
                nhead=8,
                num_layers=4,
                **kwargs
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.models[model_type] = model
        return model
    
    def create_ensemble(self, base_models: List[Any], ensemble_type: str = 'voting') -> Any:
        """
        Create ensemble model.
        
        Args:
            base_models: List of base models
            ensemble_type: Type of ensemble ('voting', 'stacking')
            
        Returns:
            Ensemble model
        """
        self.logger.info(f"Creating ensemble: {ensemble_type}")
        
        if ensemble_type == 'voting':
            model = VotingClassifier(
                estimators=[(f"model_{i}", model) for i, model in enumerate(base_models)],
                voting='soft',
                weights=[1] * len(base_models)
            )
        
        elif ensemble_type == 'stacking':
            model = StackingClassifier(
                estimators=[(f"model_{i}", model) for i, model in enumerate(base_models)],
                final_estimator=LogisticRegression(),
                cv=5
            )
        
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")
        
        return model


# ============================================================================
# HYPERPARAMETER TUNER CLASS
# ============================================================================

class HyperparameterTuner:
    """
    Advanced hyperparameter tuning using Optuna.
    
    Features:
    - Bayesian optimization with Optuna
    - Cross-validation
    - Early stopping
    - Multiple objective metrics
    - Parallel execution
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize hyperparameter tuner.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.study = None
        self.best_params = {}
        self.best_score = 0.0
    
    def tune_xgboost(self, X_train, y_train, X_val=None, y_val=None) -> Dict:
        """
        Tune XGBoost hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val, y_val: Validation data (optional)
            
        Returns:
            Best hyperparameters
        """
        self.logger.info("Tuning XGBoost hyperparameters...")
        
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
            }
            
            # Calculate class weights
            scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
            
            # Create model
            model = xgb.XGBClassifier(
                objective='binary:logistic',
                scale_pos_weight=scale_pos_weight,
                eval_metric='aucpr',
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1,
                **params
            )
            
            # Cross-validation
            if X_val is None:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
                return scores.mean()
            else:
                # Use validation set
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
                y_pred = model.predict(X_val)
                return f1_score(y_val, y_pred)
        
        # Create study
        self.study = optuna.create_study(
            direction=self.config.get('hyperparameter_tuning.direction', 'maximize'),
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=20)
        )
        
        # Run optimization
        self.study.optimize(
            objective,
            n_trials=self.config.get('hyperparameter_tuning.n_trials', 100),
            timeout=self.config.get('hyperparameter_tuning.timeout', 3600),
            n_jobs=self.config.get('hyperparameter_tuning.n_jobs', -1),
            show_progress_bar=True
        )
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        self.logger.info(f"Best F1 score: {self.best_score:.4f}")
        self.logger.info(f"Best params: {self.best_params}")
        
        return self.best_params
    
    def tune_lightgbm(self, X_train, y_train, X_val=None, y_val=None) -> Dict:
        """
        Tune LightGBM hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val, y_val: Validation data (optional)
            
        Returns:
            Best hyperparameters
        """
        self.logger.info("Tuning LightGBM hyperparameters...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300, step=20),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
            }
            
            model = lgb.LGBMClassifier(
                objective='binary',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                verbose=-1,
                **params
            )
            
            if X_val is None:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
                return scores.mean()
            else:
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc')
                y_pred = model.predict(X_val)
                return f1_score(y_val, y_pred)
        
        self.study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        self.study.optimize(objective, n_trials=self.config.get('hyperparameter_tuning.n_trials', 100))
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        return self.best_params
    
    def tune_random_forest(self, X_train, y_train) -> Dict:
        """
        Tune Random Forest hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Best hyperparameters
        """
        self.logger.info("Tuning Random Forest hyperparameters...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }
            
            model = RandomForestClassifier(
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                **params
            )
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
            return scores.mean()
        
        self.study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        self.study.optimize(objective, n_trials=self.config.get('hyperparameter_tuning.n_trials', 50))
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        return self.best_params
    
    def tune_neural_network(self, train_loader, val_loader, input_size) -> Dict:
        """
        Tune neural network hyperparameters.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            input_size: Input feature size
            
        Returns:
            Best hyperparameters
        """
        self.logger.info("Tuning Neural Network hyperparameters...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        def objective(trial):
            # Define hyperparameters
            n_layers = trial.suggest_int('n_layers', 2, 5)
            hidden_sizes = []
            for i in range(n_layers):
                hidden_sizes.append(trial.suggest_int(f'hidden_size_{i}', 32, 512, step=32))
            
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])
            
            # Create model
            model = NeuralNetwork(
                input_size=input_size,
                hidden_sizes=hidden_sizes,
                dropout_rate=dropout_rate
            ).to(device)
            
            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Training loop
            model.train()
            for epoch in range(10):  # Quick training for tuning
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            # Validation
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    _, predicted = torch.max(outputs.data, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(batch_y.cpu().numpy())
            
            f1 = f1_score(all_labels, all_preds)
            return f1
        
        self.study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        self.study.optimize(objective, n_trials=self.config.get('hyperparameter_tuning.n_trials', 50))
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        return self.best_params


# ============================================================================
# MODEL TRAINER CLASS
# ============================================================================

class ModelTrainer:
    """
    Main model training orchestrator.
    
    Handles:
    - Training multiple model types
    - Cross-validation
    - Early stopping
    - Model checkpointing
    - Experiment tracking with MLflow
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize model trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_loader = DataLoader(config)
        self.model_factory = ModelFactory(config)
        self.tuner = HyperparameterTuner(config)
        
        # Initialize MLflow
        self._init_mlflow()
        
        # Tracking variables
        self.trained_models = {}
        self.best_model = None
        self.best_score = 0.0
        self.training_history = {}
    
    def _init_mlflow(self):
        """Initialize MLflow for experiment tracking."""
        try:
            mlflow.set_tracking_uri(self.config.get('experiment.tracking_uri', './mlruns'))
            mlflow.set_experiment(self.config.get('experiment.name', 'fraud_detection'))
            
            # Set tags
            for key, value in self.config.get('experiment.tags', {}).items():
                mlflow.set_tag(key, value)
            
            self.logger.info("MLflow initialized")
        except Exception as e:
            self.logger.warning(f"Could not initialize MLflow: {e}")
    
    def train(self, data_path: str = None) -> Dict[str, Any]:
        """
        Run complete training pipeline.
        
        Args:
            data_path: Path to data file
            
        Returns:
            Dictionary with training results
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING MODEL TRAINING PIPELINE")
        self.logger.info("=" * 80)
        
        with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_params({
                'data_path': data_path,
                'config': json.dumps(self.config.config)
            })
            
            # Step 1: Load data
            self.logger.info("\n1. LOADING DATA")
            X, y = self.data_loader.load_data(data_path)
            mlflow.log_metrics({
                'n_samples': len(X),
                'n_features': X.shape[1],
                'fraud_rate': (y.sum() / len(y)) * 100
            })
            
            # Step 2: Create splits
            self.logger.info("\n2. CREATING DATA SPLITS")
            splits = self.data_loader.create_splits(X, y)
            
            # Step 3: Apply sampling if configured
            self.logger.info("\n3. APPLYING SAMPLING")
            X_train, y_train = self.data_loader.apply_sampling(
                splits['X_train'], splits['y_train']
            )
            
            # Step 4: Train classical ML models
            if self.config.get('models.classical_ml.enabled', True):
                self.logger.info("\n4. TRAINING CLASSICAL ML MODELS")
                self._train_classical_ml_models(
                    X_train, y_train,
                    splits['X_val'], splits['y_val']
                )
            
            # Step 5: Train deep learning models
            if self.config.get('models.deep_learning.enabled', True):
                self.logger.info("\n5. TRAINING DEEP LEARNING MODELS")
                # Create DataLoaders for deep learning
                dataloaders = self.data_loader.create_dataloaders(
                    X_train, y_train,
                    splits['X_val'], splits['y_val'],
                    splits['X_test'], splits['y_test']
                )
                self._train_deep_learning_models(dataloaders, X.shape[1])
            
            # Step 6: Create ensemble
            if self.config.get('models.classical_ml.ensemble', True) and len(self.trained_models) >= 2:
                self.logger.info("\n6. CREATING ENSEMBLE MODEL")
                self._create_ensemble(splits)
            
            # Step 7: Evaluate all models
            self.logger.info("\n7. EVALUATING MODELS")
            evaluation_results = self._evaluate_all_models(splits)
            
            # Step 8: Select best model
            self.logger.info("\n8. SELECTING BEST MODEL")
            self._select_best_model(evaluation_results)
            
            # Step 9: Save models
            self.logger.info("\n9. SAVING MODELS")
            self._save_models()
            
            # Log final metrics
            mlflow.log_metrics({
                'best_model_score': self.best_score,
                'best_model_name': self.best_model['name'] if self.best_model else None,
                'n_models_trained': len(self.trained_models)
            })
            
            self.logger.info("=" * 80)
            self.logger.info("TRAINING PIPELINE COMPLETE")
            self.logger.info("=" * 80)
            
            return {
                'trained_models': self.trained_models,
                'best_model': self.best_model,
                'evaluation_results': evaluation_results,
                'training_history': self.training_history
            }
    
    def _train_classical_ml_models(self, X_train, y_train, X_val, y_val):
        """
        Train classical machine learning models.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
        """
        algorithms = self.config.get('models.classical_ml.algorithms', [])
        
        for algo_name in algorithms:
            self.logger.info(f"  Training {algo_name}...")
            
            try:
                with mlflow.start_run(run_name=algo_name, nested=True):
                    # Log algorithm info
                    mlflow.set_tag('model_type', 'classical_ml')
                    mlflow.set_tag('algorithm', algo_name)
                    
                    # Hyperparameter tuning if enabled
                    if self.config.get('hyperparameter_tuning.enabled', True):
                        self.logger.info(f"    Tuning hyperparameters for {algo_name}...")
                        
                        if algo_name == 'xgboost':
                            best_params = self.tuner.tune_xgboost(X_train, y_train, X_val, y_val)
                        elif algo_name == 'lightgbm':
                            best_params = self.tuner.tune_lightgbm(X_train, y_train, X_val, y_val)
                        elif algo_name == 'random_forest':
                            best_params = self.tuner.tune_random_forest(X_train, y_train)
                        else:
                            best_params = {}
                        
                        mlflow.log_params(best_params)
                    else:
                        best_params = {}
                    
                    # Get class counts for scale_pos_weight
                    class_counts = y_train.value_counts().to_dict()
                    
                    # Create and train model
                    model = self.model_factory.create_model(
                        algo_name,
                        n_features=X_train.shape[1],
                        class_counts=class_counts,
                        **best_params
                    )
                    
                    # Train with early stopping if supported
                    if algo_name in ['xgboost', 'lightgbm']:
                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            early_stopping_rounds=self.config.get('training.early_stopping_patience', 50),
                            verbose=False
                        )
                    else:
                        model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_val)
                    y_pred_proba = model.predict_proba(X_val)[:, 1]
                    
                    # Calculate metrics
                    metrics = {
                        'f1': f1_score(y_val, y_pred),
                        'precision': precision_score(y_val, y_pred),
                        'recall': recall_score(y_val, y_pred),
                        'roc_auc': roc_auc_score(y_val, y_pred_proba),
                        'pr_auc': average_precision_score(y_val, y_pred_proba)
                    }
                    
                    # Log metrics
                    mlflow.log_metrics(metrics)
                    
                    # Log model
                    if algo_name == 'xgboost':
                        mlflow.xgboost.log_model(model, algo_name)
                    else:
                        mlflow.sklearn.log_model(model, algo_name)
                    
                    # Store model
                    self.trained_models[algo_name] = {
                        'model': model,
                        'metrics': metrics,
                        'params': best_params,
                        'type': 'classical_ml'
                    }
                    
                    self.logger.info(f"    {algo_name} - F1: {metrics['f1']:.4f}, "
                                   f"ROC-AUC: {metrics['roc_auc']:.4f}")
                    
            except Exception as e:
                self.logger.error(f"    Error training {algo_name}: {e}")
                continue
    
    def _train_deep_learning_models(self, dataloaders: Dict, input_size: int):
        """
        Train deep learning models.
        
        Args:
            dataloaders: Dictionary of DataLoaders
            input_size: Input feature size
        """
        architectures = self.config.get('models.deep_learning.architectures', [])
        device = torch.device(self.config.get('models.deep_learning.device', 'cpu'))
        
        for arch_name in architectures:
            self.logger.info(f"  Training {arch_name}...")
            
            try:
                with mlflow.start_run(run_name=arch_name, nested=True):
                    mlflow.set_tag('model_type', 'deep_learning')
                    mlflow.set_tag('architecture', arch_name)
                    
                    # Hyperparameter tuning for neural networks
                    if arch_name == 'neural_network' and self.config.get('hyperparameter_tuning.enabled', True):
                        best_params = self.tuner.tune_neural_network(
                            dataloaders['train'],
                            dataloaders['val'],
                            input_size
                        )
                        mlflow.log_params(best_params)
                    else:
                        best_params = {}
                    
                    # Create model
                    model = self.model_factory.create_model(
                        arch_name,
                        n_features=input_size,
                        **best_params
                    ).to(device)
                    
                    # Loss and optimizer
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(
                        model.parameters(),
                        lr=self.config.get('models.deep_learning.learning_rate', 0.001)
                    )
                    
                    # Learning rate scheduler
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min', patience=5, factor=0.5
                    )
                    
                    # Training loop
                    epochs = self.config.get('models.deep_learning.epochs', 100)
                    patience = self.config.get('training.early_stopping_patience', 10)
                    
                    best_val_loss = float('inf')
                    best_model_state = None
                    patience_counter = 0
                    
                    train_losses = []
                    val_losses = []
                    
                    for epoch in range(epochs):
                        # Training
                        model.train()
                        train_loss = 0.0
                        for batch_X, batch_y in dataloaders['train']:
                            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                            
                            optimizer.zero_grad()
                            outputs = model(batch_X)
                            loss = criterion(outputs, batch_y)
                            loss.backward()
                            optimizer.step()
                            
                            train_loss += loss.item()
                        
                        avg_train_loss = train_loss / len(dataloaders['train'])
                        train_losses.append(avg_train_loss)
                        
                        # Validation
                        model.eval()
                        val_loss = 0.0
                        all_preds = []
                        all_labels = []
                        
                        with torch.no_grad():
                            for batch_X, batch_y in dataloaders['val']:
                                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                                outputs = model(batch_X)
                                loss = criterion(outputs, batch_y)
                                val_loss += loss.item()
                                
                                _, predicted = torch.max(outputs.data, 1)
                                all_preds.extend(predicted.cpu().numpy())
                                all_labels.extend(batch_y.cpu().numpy())
                        
                        avg_val_loss = val_loss / len(dataloaders['val'])
                        val_losses.append(avg_val_loss)
                        
                        # Update learning rate
                        scheduler.step(avg_val_loss)
                        
                        # Calculate validation metrics
                        val_f1 = f1_score(all_labels, all_preds)
                        
                        # Early stopping
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            best_model_state = model.state_dict().copy()
                            patience_counter = 0
                        else:
                            patience_counter += 1
                        
                        if patience_counter >= patience:
                            self.logger.info(f"    Early stopping at epoch {epoch+1}")
                            break
                        
                        if (epoch + 1) % 10 == 0:
                            self.logger.info(f"    Epoch {epoch+1}/{epochs} - "
                                           f"Train Loss: {avg_train_loss:.4f}, "
                                           f"Val Loss: {avg_val_loss:.4f}, "
                                           f"Val F1: {val_f1:.4f}")
                    
                    # Load best model
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    
                    # Final evaluation on test set
                    model.eval()
                    all_preds = []
                    all_probs = []
                    all_labels = []
                    
                    with torch.no_grad():
                        for batch_X, batch_y in dataloaders['test']:
                            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                            outputs = model(batch_X)
                            probs = torch.softmax(outputs, dim=1)
                            
                            _, predicted = torch.max(outputs.data, 1)
                            all_preds.extend(predicted.cpu().numpy())
                            all_probs.extend(probs.cpu().numpy()[:, 1])
                            all_labels.extend(batch_y.cpu().numpy())
                    
                    # Calculate metrics
                    metrics = {
                        'f1': f1_score(all_labels, all_preds),
                        'precision': precision_score(all_labels, all_preds),
                        'recall': recall_score(all_labels, all_preds),
                        'roc_auc': roc_auc_score(all_labels, all_probs),
                        'pr_auc': average_precision_score(all_labels, all_probs)
                    }
                    
                    # Log metrics
                    mlflow.log_metrics(metrics)
                    
                    # Log model
                    mlflow.pytorch.log_model(model, arch_name)
                    
                    # Store model
                    self.trained_models[arch_name] = {
                        'model': model,
                        'metrics': metrics,
                        'params': best_params,
                        'type': 'deep_learning',
                        'history': {
                            'train_loss': train_losses,
                            'val_loss': val_losses
                        }
                    }
                    
                    self.logger.info(f"    {arch_name} - F1: {metrics['f1']:.4f}, "
                                   f"ROC-AUC: {metrics['roc_auc']:.4f}")
                    
            except Exception as e:
                self.logger.error(f"    Error training {arch_name}: {e}")
                continue
    
    def _create_ensemble(self, splits: Dict):
        """
        Create ensemble model from base models.
        
        Args:
            splits: Data splits dictionary
        """
        try:
            # Get top performing models
            base_models = []
            for name, model_info in self.trained_models.items():
                if model_info['type'] == 'classical_ml':
                    base_models.append((name, model_info['model']))
            
            if len(base_models) < 2:
                self.logger.info("    Not enough models for ensemble")
                return
            
            with mlflow.start_run(run_name='ensemble', nested=True):
                mlflow.set_tag('model_type', 'ensemble')
                
                # Create voting ensemble
                estimators = [(name, model) for name, model in base_models]
                ensemble = VotingClassifier(
                    estimators=estimators,
                    voting='soft'
                )
                
                # Train ensemble
                ensemble.fit(splits['X_train'], splits['y_train'])
                
                # Evaluate
                y_pred = ensemble.predict(splits['X_val'])
                y_pred_proba = ensemble.predict_proba(splits['X_val'])[:, 1]
                
                metrics = {
                    'f1': f1_score(splits['y_val'], y_pred),
                    'precision': precision_score(splits['y_val'], y_pred),
                    'recall': recall_score(splits['y_val'], y_pred),
                    'roc_auc': roc_auc_score(splits['y_val'], y_pred_proba),
                    'pr_auc': average_precision_score(splits['y_val'], y_pred_proba)
                }
                
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(ensemble, 'ensemble')
                
                # Store ensemble
                self.trained_models['ensemble_voting'] = {
                    'model': ensemble,
                    'metrics': metrics,
                    'type': 'ensemble',
                    'base_models': [name for name, _ in base_models]
                }
                
                self.logger.info(f"    Ensemble - F1: {metrics['f1']:.4f}")
                
        except Exception as e:
            self.logger.error(f"    Error creating ensemble: {e}")
    
    def _evaluate_all_models(self, splits: Dict) -> Dict:
        """
        Evaluate all trained models on test set.
        
        Args:
            splits: Data splits dictionary
            
        Returns:
            Dictionary with evaluation results
        """
        evaluation_results = {}
        
        X_test = splits['X_test']
        y_test = splits['y_test']
        
        for name, model_info in self.trained_models.items():
            self.logger.info(f"  Evaluating {name}...")
            
            model = model_info['model']
            
            # Make predictions
            if model_info['type'] == 'deep_learning':
                # Deep learning models need special handling
                device = next(model.parameters()).device
                X_test_tensor = torch.FloatTensor(X_test.values).to(device)
                
                model.eval()
                with torch.no_grad():
                    outputs = model(X_test_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    y_pred = torch.max(outputs.data, 1)[1].cpu().numpy()
                    y_pred_proba = probs.cpu().numpy()[:, 1]
            else:
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate all metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'pr_auc': average_precision_score(y_test, y_pred_proba)
            }
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = cm.ravel()
            
            # Specificity and other derived metrics
            metrics['specificity'] = metrics['tn'] / (metrics['tn'] + metrics['fp']) if (metrics['tn'] + metrics['fp']) > 0 else 0
            metrics['fpr'] = metrics['fp'] / (metrics['fp'] + metrics['tn']) if (metrics['fp'] + metrics['tn']) > 0 else 0
            metrics['fnr'] = metrics['fn'] / (metrics['fn'] + metrics['tp']) if (metrics['fn'] + metrics['tp']) > 0 else 0
            
            # Business metrics
            fraud_cost = self.config.get('evaluation.business_constants.fraud_cost', 100)
            fp_cost = self.config.get('evaluation.business_constants.false_positive_cost', 1)
            
            metrics['fraud_savings'] = metrics['tp'] * fraud_cost
            metrics['fp_cost'] = metrics['fp'] * fp_cost
            metrics['net_savings'] = metrics['fraud_savings'] - metrics['fp_cost']
            
            # Store results
            evaluation_results[name] = {
                'metrics': metrics,
                'predictions': {
                    'y_true': y_test.tolist(),
                    'y_pred': y_pred.tolist(),
                    'y_pred_proba': y_pred_proba.tolist()
                }
            }
            
            self.logger.info(f"    F1: {metrics['f1']:.4f}, "
                           f"Precision: {metrics['precision']:.4f}, "
                           f"Recall: {metrics['recall']:.4f}")
        
        return evaluation_results
    
    def _select_best_model(self, evaluation_results: Dict):
        """
        Select the best performing model.
        
        Args:
            evaluation_results: Dictionary with evaluation results
        """
        primary_metric = self.config.get('hyperparameter_tuning.metric', 'f1')
        
        best_score = -1
        best_model_name = None
        
        for name, results in evaluation_results.items():
            score = results['metrics'][primary_metric]
            
            if score > best_score:
                best_score = score
                best_model_name = name
        
        if best_model_name:
            self.best_model = {
                'name': best_model_name,
                'model': self.trained_models[best_model_name]['model'],
                'metrics': evaluation_results[best_model_name]['metrics'],
                'type': self.trained_models[best_model_name]['type']
            }
            self.best_score = best_score
            
            self.logger.info(f"  Best model: {best_model_name}")
            self.logger.info(f"  Best {primary_metric}: {best_score:.4f}")
    
    def _save_models(self):
        """Save all trained models."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save best model separately
        if self.best_model:
            best_model_path = f"models/best_model_{timestamp}.pkl"
            with open(best_model_path, 'wb') as f:
                pickle.dump(self.best_model['model'], f)
            
            # Save metadata
            metadata = {
                'name': self.best_model['name'],
                'metrics': self.best_model['metrics'],
                'type': self.best_model['type'],
                'timestamp': timestamp
            }
            
            with open(f"models/best_model_{timestamp}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"  Best model saved to: {best_model_path}")
        
        # Save all models
        for name, model_info in self.trained_models.items():
            model_path = f"models/{name}_{timestamp}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model_info['model'], f)
            
            self.logger.info(f"  {name} saved to: {model_path}")


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """
    Main execution function for model training.
    """
    parser = argparse.ArgumentParser(
        description='VeritasFinancial - Model Training for Fraud Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/model_config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        help='Path to data file (overrides config)'
    )
    
    parser.add_argument(
        '--model-type', '-m',
        type=str,
        choices=['all', 'classical', 'deep', 'xgboost', 'lightgbm', 'random_forest', 'neural_network'],
        default='all',
        help='Type of model to train'
    )
    
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Enable hyperparameter tuning'
    )
    
    parser.add_argument(
        '--no-mlflow',
        action='store_true',
        help='Disable MLflow tracking'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("=" * 80)
        logger.info("VERITASFINANCIAL - MODEL TRAINING")
        logger.info("=" * 80)
        
        # Load configuration
        config = TrainingConfig(args.config)
        
        # Override config with command line arguments
        if args.model_type != 'all':
            # Filter models based on type
            if args.model_type == 'classical':
                config.config['models']['deep_learning']['enabled'] = False
            elif args.model_type == 'deep':
                config.config['models']['classical_ml']['enabled'] = False
            else:
                # Single model type
                config.config['models']['classical_ml']['algorithms'] = [args.model_type]
                config.config['models']['deep_learning']['enabled'] = False
        
        if args.tune:
            config.config['hyperparameter_tuning']['enabled'] = True
        
        if args.no_mlflow:
            # Disable MLflow by setting tracking URI to local file
            config.config['experiment']['tracking_uri'] = './mlruns'
        
        # Initialize and run trainer
        trainer = ModelTrainer(config)
        results = trainer.train(data_path=args.data)
        
        logger.info("-" * 80)
        logger.info("Training Summary:")
        logger.info(f"  Models trained: {len(results['trained_models'])}")
        if results['best_model']:
            logger.info(f"  Best model: {results['best_model']['name']}")
            logger.info(f"  Best F1 score: {results['best_model']['metrics']['f1']:.4f}")
            logger.info(f"  Best ROC-AUC: {results['best_model']['metrics']['roc_auc']:.4f}")
        
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())