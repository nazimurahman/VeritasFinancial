"""
Preprocessing Pipeline Module

This module provides an end-to-end preprocessing pipeline that combines
all data cleaning, transformation, and handling steps into a single
coherent pipeline for fraud detection.

The pipeline ensures:
1. Consistent preprocessing across training and inference
2. Proper ordering of operations
3. Tracking of preprocessing steps
4. Reproducibility
5. Easy deployment
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import joblib
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import yaml
import json
from datetime import datetime

# Import all preprocessing components
from ..cleaners.transaction_cleaner import TransactionCleaner
from ..cleaners.customer_cleaner import CustomerCleaner
from ..cleaners.device_cleaner import DeviceCleaner
from ..transformers.categorical_encoder import CategoricalEncoder
from ..transformers.numerical_scaler import NumericalScaler
from ..transformers.datetime_processor import DateTimeProcessor
from ..handlers.missing_values import MissingValueHandler
from ..handlers.outliers import OutlierHandler
from ..handlers.imbalance import ImbalanceHandler

logger = logging.getLogger(__name__)


class PreprocessingPipeline(BaseEstimator, TransformerMixin):
    """
    End-to-end preprocessing pipeline for fraud detection.
    
    This pipeline orchestrates all preprocessing steps in the correct order,
    handling different data types and ensuring consistency.
    
    Attributes:
        steps (list): List of preprocessing steps
        step_names (list): Names of steps
        fitted_steps (dict): Fitted transformers
        config (dict): Pipeline configuration
        target_column (str): Name of target column
        feature_names_ (list): Names of features after preprocessing
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        target_column: str = 'is_fraud',
        preserve_target: bool = True,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            config_path: Path to configuration file
            target_column: Name of target column
            preserve_target: Whether to preserve target column
            random_state: Random state for reproducibility
            **kwargs: Additional arguments
        """
        self.config_path = config_path
        self.target_column = target_column
        self.preserve_target = preserve_target
        self.random_state = random_state
        self.kwargs = kwargs
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize steps
        self.steps = []
        self.step_names = []
        self.fitted_steps = {}
        self.feature_names_ = []
        
        # Build pipeline
        self._build_pipeline()
        
        logger.info(f"PreprocessingPipeline initialized with {len(self.steps)} steps")
    
    def _load_config(self) -> Dict:
        """
        Load configuration from file or use defaults.
        
        Returns:
            Configuration dictionary
        """
        default_config = {
            'cleaners': {
                'transaction': {'enabled': True},
                'customer': {'enabled': True},
                'device': {'enabled': True}
            },
            'handlers': {
                'missing_values': {
                    'enabled': True,
                    'strategy': 'auto',
                    'missing_threshold': 0.5,
                    'add_missing_indicator': True
                },
                'outliers': {
                    'enabled': True,
                    'detection_method': 'iqr',
                    'handling_method': 'cap',
                    'add_outlier_flag': True
                },
                'imbalance': {
                    'enabled': False,  # Usually applied only to training
                    'method': 'smote',
                    'sampling_strategy': 'auto'
                }
            },
            'transformers': {
                'datetime': {
                    'enabled': True,
                    'add_cyclical_features': True,
                    'add_business_hours': True
                },
                'categorical': {
                    'enabled': True,
                    'encoding_method': 'auto'
                },
                'numerical': {
                    'enabled': True,
                    'scaling_method': 'auto',
                    'outlier_handling': 'clip'
                }
            }
        }
        
        if self.config_path:
            try:
                with open(self.config_path, 'r') as f:
                    if self.config_path.endswith('.yaml'):
                        config = yaml.safe_load(f)
                    elif self.config_path.endswith('.json'):
                        config = json.load(f)
                    else:
                        config = default_config
                
                # Merge with defaults
                config = self._merge_configs(default_config, config)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config
            except Exception as e:
                logger.warning(f"Could not load config: {e}. Using defaults.")
        
        return default_config
    
    def _merge_configs(self, default: Dict, custom: Dict) -> Dict:
        """
        Recursively merge custom config with defaults.
        
        Args:
            default: Default configuration
            custom: Custom configuration
            
        Returns:
            Merged configuration
        """
        merged = default.copy()
        
        for key, value in custom.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _build_pipeline(self):
        """Build the preprocessing pipeline by adding steps in order."""
        
        # Step 1: Data cleaners (applied first to clean raw data)
        if self.config['cleaners']['transaction']['enabled']:
            self._add_step(
                'transaction_cleaner',
                TransactionCleaner(config=self.config['cleaners']['transaction'])
            )
        
        if self.config['cleaners']['customer']['enabled']:
            self._add_step(
                'customer_cleaner',
                CustomerCleaner(config=self.config['cleaners']['customer'])
            )
        
        if self.config['cleaners']['device']['enabled']:
            self._add_step(
                'device_cleaner',
                DeviceCleaner(config=self.config['cleaners']['device'])
            )
        
        # Step 2: Missing value handler
        if self.config['handlers']['missing_values']['enabled']:
            self._add_step(
                'missing_values',
                MissingValueHandler(**self.config['handlers']['missing_values'])
            )
        
        # Step 3: Outlier handler
        if self.config['handlers']['outliers']['enabled']:
            self._add_step(
                'outliers',
                OutlierHandler(**self.config['handlers']['outliers'])
            )
        
        # Step 4: DateTime processor
        if self.config['transformers']['datetime']['enabled']:
            self._add_step(
                'datetime',
                DateTimeProcessor(**self.config['transformers']['datetime'])
            )
        
        # Step 5: Categorical encoder
        if self.config['transformers']['categorical']['enabled']:
            self._add_step(
                'categorical',
                CategoricalEncoder(**self.config['transformers']['categorical'])
            )
        
        # Step 6: Numerical scaler
        if self.config['transformers']['numerical']['enabled']:
            self._add_step(
                'numerical',
                NumericalScaler(**self.config['transformers']['numerical'])
            )
        
        # Note: Imbalance handler is applied separately during training
    
    def _add_step(self, name: str, transformer):
        """
        Add a step to the pipeline.
        
        Args:
            name: Step name
            transformer: Transformer object
        """
        self.steps.append((name, transformer))
        self.step_names.append(name)
        logger.debug(f"Added step: {name}")
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the preprocessing pipeline.
        
        Args:
            X: Input DataFrame
            y: Optional target values
            
        Returns:
            self
        """
        logger.info(f"Fitting preprocessing pipeline on {len(X)} samples")
        
        # Separate target if present and should be preserved
        if self.preserve_target and self.target_column in X.columns:
            X_data = X.drop(columns=[self.target_column])
            y_data = X[self.target_column] if y is None else y
        else:
            X_data = X
            y_data = y
        
        # Fit each step
        for name, transformer in self.steps:
            logger.info(f"Fitting step: {name}")
            
            if hasattr(transformer, 'fit'):
                transformer.fit(X_data, y_data)
            
            self.fitted_steps[name] = transformer
            
            # Transform data for next step (if needed for fitting)
            if hasattr(transformer, 'transform'):
                X_data = transformer.transform(X_data)
        
        # Generate final feature names
        self._generate_feature_names(X_data)
        
        logger.info("Pipeline fitting complete")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted pipeline.
        
        Args:
            X: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        logger.info(f"Transforming {len(X)} samples")
        
        # Separate target if present
        if self.preserve_target and self.target_column in X.columns:
            X_data = X.drop(columns=[self.target_column])
            target_data = X[self.target_column]
        else:
            X_data = X
            target_data = None
        
        # Apply each step
        for name, transformer in self.steps:
            logger.debug(f"Applying step: {name}")
            
            if hasattr(transformer, 'transform'):
                X_data = transformer.transform(X_data)
        
        # Add target back if needed
        if target_data is not None:
            X_data[self.target_column] = target_data.values
        
        logger.info(f"Transformation complete. Output shape: {X_data.shape}")
        return X_data
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit to data, then transform it.
        
        Args:
            X: Input DataFrame
            y: Optional target values
            
        Returns:
            Transformed DataFrame
        """
        return self.fit(X, y).transform(X)
    
    def _generate_feature_names(self, X: pd.DataFrame):
        """
        Generate names of features after preprocessing.
        
        Args:
            X: Transformed DataFrame
        """
        self.feature_names_ = X.columns.tolist()
        logger.info(f"Final feature count: {len(self.feature_names_)}")
    
    def get_feature_names_out(self, input_features=None) -> List[str]:
        """
        Get output feature names.
        
        Returns:
            List of feature names
        """
        return self.feature_names_
    
    def get_step(self, step_name: str):
        """
        Get a specific step from the pipeline.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Transformer object
        """
        if step_name in self.fitted_steps:
            return self.fitted_steps[step_name]
        else:
            raise ValueError(f"Step '{step_name}' not found")
    
    def apply_imbalance_handling(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'smote',
        **kwargs
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply imbalance handling to training data.
        
        This method should be called separately during training only.
        
        Args:
            X: Features
            y: Target
            method: Resampling method
            **kwargs: Additional arguments for imbalance handler
            
        Returns:
            Tuple of (resampled_features, resampled_target)
        """
        logger.info(f"Applying imbalance handling with method: {method}")
        
        # Get imbalance handler config
        imbalance_config = self.config['handlers']['imbalance'].copy()
        imbalance_config.update(kwargs)
        imbalance_config['method'] = method
        
        # Create and apply handler
        handler = ImbalanceHandler(**imbalance_config)
        X_resampled, y_resampled = handler.fit_resample(X, y)
        
        logger.info(f"Imbalance handling complete. New shape: {X_resampled.shape}")
        return X_resampled, y_resampled
    
    def save(self, path: str):
        """
        Save the fitted pipeline to disk.
        
        Args:
            path: Path to save the pipeline
        """
        # Save pipeline
        pipeline_data = {
            'steps': self.steps,
            'step_names': self.step_names,
            'fitted_steps': self.fitted_steps,
            'config': self.config,
            'target_column': self.target_column,
            'feature_names_': self.feature_names_,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(pipeline_data, path)
        logger.info(f"Pipeline saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        """
        Load a fitted pipeline from disk.
        
        Args:
            path: Path to the saved pipeline
            
        Returns:
            Loaded pipeline
        """
        pipeline_data = joblib.load(path)
        
        # Create new instance
        instance = cls(
            config_path=None,
            target_column=pipeline_data['target_column']
        )
        
        # Restore state
        instance.steps = pipeline_data['steps']
        instance.step_names = pipeline_data['step_names']
        instance.fitted_steps = pipeline_data['fitted_steps']
        instance.config = pipeline_data['config']
        instance.feature_names_ = pipeline_data['feature_names_']
        
        logger.info(f"Pipeline loaded from {path}")
        return instance
    
    def get_pipeline_report(self) -> Dict:
        """
        Get detailed report on pipeline steps and configuration.
        
        Returns:
            Dictionary with pipeline report
        """
        report = {
            'steps': self.step_names,
            'config': self.config,
            'target_column': self.target_column,
            'feature_count': len(self.feature_names_),
            'features': self.feature_names_[:10] + ['...'] if len(self.feature_names_) > 10 else self.feature_names_,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add step-specific information
        step_info = {}
        for name, transformer in self.fitted_steps.items():
            if hasattr(transformer, 'get_cleaning_stats'):
                step_info[name] = transformer.get_cleaning_stats()
            elif hasattr(transformer, 'get_scaling_info'):
                step_info[name] = transformer.get_scaling_info()
            elif hasattr(transformer, 'get_processor_info'):
                step_info[name] = transformer.get_processor_info()
            elif hasattr(transformer, 'get_missing_report'):
                step_info[name] = transformer.get_missing_report()
            elif hasattr(transformer, 'get_outlier_report'):
                step_info[name] = transformer.get_outlier_report()
        
        report['step_details'] = step_info
        
        return report