#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VeritasFinancial - Banking Fraud Detection System
Data Pipeline Execution Script
==================================================
This script orchestrates the complete data pipeline for fraud detection:
1. Data acquisition from multiple sources (databases, APIs, files)
2. Data validation and quality checks
3. Data preprocessing and cleaning
4. Exploratory Data Analysis (EDA)
5. Feature engineering and transformation
6. Data versioning and storage

Author: VeritasFinancial Data Science Team
Version: 1.0.0
"""

# ============================================================================
# IMPORTS SECTION
# ============================================================================
# Standard library imports - Python's built-in modules
import os                       # Operating system interfaces for file/directory operations
import sys                      # System-specific parameters and functions
import json                     # JSON encoding/decoding for configuration files
import yaml                     # YAML parsing for configuration files (more readable than JSON)
import logging                  # Logging facility for Python
import argparse                 # Command-line argument parsing
from datetime import datetime, timedelta  # Date and time handling
from pathlib import Path        # Object-oriented filesystem paths
import pickle                   # Python object serialization
import hashlib                  # Secure hashes for data versioning
from typing import Dict, List, Optional, Union, Any  # Type hints for better code documentation

# Third-party imports - Data manipulation and analysis
import numpy as np              # Numerical computing (arrays, matrices)
import pandas as pd             # Data manipulation and analysis
pd.set_option('display.max_columns', None)  # Show all columns in pandas display
pd.set_option('display.width', None)        # Auto-detect display width

# Data preprocessing and feature engineering
from sklearn.model_selection import train_test_split  # Split data into train/test sets
from sklearn.preprocessing import (  # Various preprocessing techniques
    StandardScaler,     # Standardize features by removing mean and scaling to unit variance
    RobustScaler,       # Scale features using statistics that are robust to outliers
    MinMaxScaler,       # Transform features by scaling each feature to a given range
    LabelEncoder,       # Encode target labels with value between 0 and n_classes-1
    OneHotEncoder       # Encode categorical features as a one-hot numeric array
)
from sklearn.impute import SimpleImputer  # Imputation transformer for completing missing values
from sklearn.compose import ColumnTransformer  # Applies transformers to columns of an array
from sklearn.pipeline import Pipeline        # Chain multiple estimators into one

# Imbalanced learning (crucial for fraud detection where fraud is rare)
from imblearn.over_sampling import (  # Oversampling techniques
    SMOTE,              # Synthetic Minority Over-sampling Technique
    ADASYN,             # Adaptive Synthetic Sampling
    RandomOverSampler   # Random oversampling
)
from imblearn.under_sampling import (  # Undersampling techniques
    RandomUnderSampler,  # Random undersampling
    NearMiss,           # NearMiss undersampling
    TomekLinks          # Tomek links undersampling
)
from imblearn.combine import SMOTETomek  # Combined over and under sampling

# Feature engineering
from feature_engine import (
    creation,           # Feature creation tools
    selection,          # Feature selection tools
    encoding,           # Categorical encoding tools
    transformation,     # Feature transformation tools
    datetime_features   # DateTime feature extraction
)

# Database connectors
import sqlalchemy as sa  # SQL toolkit and Object Relational Mapper
from sqlalchemy import create_engine, text  # Database engine creation
import psycopg2          # PostgreSQL database adapter
import pymongo           # MongoDB driver

# Cloud storage
import boto3              # AWS SDK for Python
from azure.storage.blob import BlobServiceClient  # Azure Blob Storage
import google.cloud.storage as gcs  # Google Cloud Storage

# Data quality and validation
import great_expectations as ge  # Data validation and profiling

# Visualization (for EDA)
import matplotlib.pyplot as plt  # Plotting library
import seaborn as sns            # Statistical data visualization
plt.style.use('seaborn-v0_8-darkgrid')  # Set plot style

# Progress tracking
from tqdm import tqdm  # Progress bars for loops

# Suppress warnings (optional - use carefully in production)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PROJECT-SPECIFIC IMPORTS
# ============================================================================
# Add the project root to Python path for local imports
sys.path.append(str(Path(__file__).parent.parent))

# Import project modules
from src.data_acquisition import (  # Data acquisition modules
    BankingDataClient,
    APIClient,
    DatabaseConnector,
    StreamConsumer
)
from src.data_preprocessing.pipelines import (  # Preprocessing pipelines
    FraudPreprocessingPipeline,
    DataCleaner,
    OutlierHandler,
    MissingValueHandler
)
from src.data_preprocessing.transformers import (  # Data transformers
    DateTimeProcessor,
    CategoricalEncoder,
    NumericalScaler
)
from src.exploratory_analysis import (  # EDA tools
    FraudStatisticalAnalysis,
    VisualizationGenerator,
    CorrelationAnalysis,
    TemporalAnalysis
)
from src.feature_engineering import (  # Feature engineering
    TransactionFeatureEngineer,
    CustomerFeatureEngineer,
    BehavioralFeatureEngineer,
    TemporalFeatureEngineer,
    GraphFeatureEngineer
)
from src.utils import (  # Utility functions
    Logger,
    ConfigManager,
    DataSerializer,
    ParallelProcessor,
    SecurityManager
)

# ============================================================================
# CONFIGURATION AND LOGGING SETUP
# ============================================================================

class DataPipelineConfig:
    """
    Configuration manager for the data pipeline.
    
    This class handles all configuration aspects including:
    - Loading config files (YAML/JSON)
    - Environment variable substitution
    - Configuration validation
    - Default values management
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
                        If None, uses default configuration
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_default_config()
        
        if config_path and os.path.exists(config_path):
            self.config.update(self._load_config_file(config_path))
        
        # Validate configuration
        self._validate_config()
        
        # Set up environment variable substitution
        self._substitute_env_vars()
        
    def _load_default_config(self) -> Dict:
        """
        Load default configuration values.
        
        Returns:
            Dictionary with default configuration
        """
        return {
            'pipeline': {
                'name': 'fraud_detection_pipeline',
                'version': '1.0.0',
                'mode': 'full',  # full, incremental, realtime
                'parallel_workers': 4,
                'batch_size': 10000,
                'memory_limit': '8GB'
            },
            'data_sources': {
                'transactions': {
                    'type': 'csv',
                    'path': 'data/raw/transactions.csv',
                    'format': 'csv',
                    'compression': None
                },
                'customers': {
                    'type': 'parquet',
                    'path': 'data/raw/customers.parquet'
                },
                'devices': {
                    'type': 'json',
                    'path': 'data/raw/devices.json'
                }
            },
            'data_quality': {
                'missing_threshold': 0.3,  # Max allowed missing values
                'duplicate_threshold': 0.1,  # Max allowed duplicates
                'outlier_method': 'iqr',  # iqr, zscore, isolation_forest
                'outlier_threshold': 1.5  # IQR multiplier
            },
            'preprocessing': {
                'handle_missing': 'advanced',  # simple, advanced, auto
                'handle_outliers': 'winsorize',  # remove, winsorize, cap
                'scaling_method': 'robust',  # standard, robust, minmax
                'encoding_method': 'target'  # onehot, label, target, frequency
            },
            'feature_engineering': {
                'create_temporal_features': True,
                'create_behavioral_features': True,
                'create_interaction_features': True,
                'max_features': 500,
                'feature_selection': 'mutual_info'  # mutual_info, chi2, f_classif
            },
            'output': {
                'processed_data_path': 'data/processed/',
                'features_path': 'data/features/',
                'cache_path': 'data/cache/',
                'save_format': 'parquet',  # parquet, csv, feather
                'compression': 'snappy'
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/data_pipeline.log',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
    
    def _load_config_file(self, config_path: str) -> Dict:
        """
        Load configuration from file (YAML or JSON).
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary with configuration
        """
        try:
            file_ext = Path(config_path).suffix.lower()
            
            if file_ext in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            elif file_ext == '.json':
                with open(config_path, 'r') as f:
                    return json.load(f)
            else:
                self.logger.warning(f"Unsupported config file format: {file_ext}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error loading config file: {e}")
            return {}
    
    def _validate_config(self):
        """
        Validate configuration values.
        
        Raises:
            ValueError: If configuration is invalid
        """
        required_keys = ['pipeline', 'data_sources', 'output']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        # Validate data sources
        if not self.config['data_sources']:
            raise ValueError("No data sources configured")
        
        # Validate paths
        for source_name, source_config in self.config['data_sources'].items():
            if 'path' in source_config:
                path = Path(source_config['path'])
                # Create directory if it doesn't exist
                path.parent.mkdir(parents=True, exist_ok=True)
    
    def _substitute_env_vars(self):
        """
        Substitute environment variables in configuration.
        Format: ${ENV_VAR_NAME}
        """
        def substitute_value(value):
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                env_var = value[2:-1]
                return os.getenv(env_var, value)
            return value
        
        # Recursively substitute in dictionary
        def recursive_substitute(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    recursive_substitute(value)
                elif isinstance(value, list):
                    d[key] = [substitute_value(v) for v in value]
                else:
                    d[key] = substitute_value(value)
        
        recursive_substitute(self.config)
    
    def get(self, key: str, default=None):
        """
        Get configuration value by dot notation key.
        
        Args:
            key: Dot notation key (e.g., 'pipeline.batch_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        
        return value


class DataPipeline:
    """
    Main data pipeline orchestrator for fraud detection.
    
    This class coordinates the entire data pipeline:
    1. Data acquisition from multiple sources
    2. Data validation and quality checks
    3. Data preprocessing and cleaning
    4. Exploratory Data Analysis
    5. Feature engineering
    6. Data versioning and storage
    
    The pipeline is designed to be:
    - Modular: Each step is independent and configurable
    - Reproducible: All steps are versioned and logged
    - Scalable: Can handle large datasets with parallel processing
    - Production-ready: Includes error handling, logging, and monitoring
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the data pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Setup logging first
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        self.logger.info("=" * 80)
        self.logger.info("VERITASFINANCIAL - DATA PIPELINE INITIALIZATION")
        self.logger.info("=" * 80)
        
        # Load configuration
        self.config = DataPipelineConfig(config_path)
        self.logger.info(f"Pipeline: {self.config.get('pipeline.name')} v{self.config.get('pipeline.version')}")
        self.logger.info(f"Mode: {self.config.get('pipeline.mode')}")
        
        # Initialize components
        self.data_sources = {}
        self.data_frames = {}
        self.preprocessing_pipeline = None
        self.feature_pipeline = None
        self.metadata = {
            'pipeline_id': hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8],
            'start_time': datetime.now(),
            'steps_completed': [],
            'statistics': {},
            'warnings': [],
            'errors': []
        }
        
        # Create output directories
        self._create_directories()
        
        # Initialize data quality expectations
        self._init_data_quality()
        
        self.logger.info("Pipeline initialization complete")
        self.logger.info("-" * 80)
    
    def _setup_logging(self):
        """
        Setup logging configuration.
        Creates both file and console handlers with appropriate formatting.
        """
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_file = log_config.get('file', 'logs/data_pipeline.log')
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Create logs directory if it doesn't exist
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def _create_directories(self):
        """
        Create necessary directories for data storage.
        """
        directories = [
            self.config.get('output.processed_data_path'),
            self.config.get('output.features_path'),
            self.config.get('output.cache_path'),
            'data/raw',
            'data/external',
            'logs',
            'reports/eda',
            'reports/quality',
            'artifacts/pipelines'
        ]
        
        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Created directory: {directory}")
    
    def _init_data_quality(self):
        """
        Initialize data quality expectations using Great Expectations.
        This helps maintain data quality throughout the pipeline.
        """
        try:
            # Create Great Expectations context
            self.ge_context = ge.data_context.DataContext()
            
            # Define expectations for different data types
            self.data_expectations = {
                'transactions': {
                    'transaction_id': {'expect_column_values_to_not_be_null': True},
                    'amount': {
                        'expect_column_values_to_be_between': {'min_value': 0.01, 'max_value': 1000000},
                        'expect_column_values_to_not_be_null': True
                    },
                    'customer_id': {'expect_column_values_to_not_be_null': True},
                    'transaction_time': {'expect_column_values_to_not_be_null': True}
                },
                'customers': {
                    'customer_id': {'expect_column_values_to_not_be_null': True},
                    'account_age_days': {'expect_column_values_to_be_between': {'min_value': 0, 'max_value': 36500}}
                },
                'devices': {
                    'device_id': {'expect_column_values_to_not_be_null': True},
                    'risk_score': {'expect_column_values_to_be_between': {'min_value': 0, 'max_value': 100}}
                }
            }
            
            self.logger.info("Data quality expectations initialized")
            
        except Exception as e:
            self.logger.warning(f"Could not initialize Great Expectations: {e}")
            self.ge_context = None
    
    def run(self, steps: List[str] = None):
        """
        Execute the data pipeline.
        
        Args:
            steps: List of steps to execute. If None, runs all steps.
                  Available steps: ['acquire', 'validate', 'preprocess', 'eda', 'engineer', 'save']
        
        Returns:
            Dictionary with pipeline results and metadata
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING DATA PIPELINE EXECUTION")
        self.logger.info("=" * 80)
        
        # Define available steps
        available_steps = {
            'acquire': self._acquire_data,
            'validate': self._validate_data,
            'preprocess': self._preprocess_data,
            'eda': self._run_eda,
            'engineer': self._engineer_features,
            'save': self._save_results
        }
        
        # Determine which steps to run
        if steps is None:
            steps_to_run = list(available_steps.keys())
        else:
            steps_to_run = [step for step in steps if step in available_steps]
        
        self.logger.info(f"Steps to execute: {steps_to_run}")
        
        # Execute each step
        for step_name in steps_to_run:
            try:
                self.logger.info("-" * 60)
                self.logger.info(f"Executing step: {step_name.upper()}")
                
                # Execute step with timing
                start_time = datetime.now()
                result = available_steps[step_name]()
                elapsed_time = (datetime.now() - start_time).total_seconds()
                
                # Record step completion
                self.metadata['steps_completed'].append({
                    'step': step_name,
                    'timestamp': datetime.now().isoformat(),
                    'elapsed_time': elapsed_time,
                    'status': 'success'
                })
                
                self.logger.info(f"Step {step_name} completed in {elapsed_time:.2f} seconds")
                
            except Exception as e:
                self.logger.error(f"Error in step {step_name}: {str(e)}", exc_info=True)
                self.metadata['errors'].append({
                    'step': step_name,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                
                # Decide whether to continue or stop
                if self.config.get('pipeline.stop_on_error', True):
                    self.logger.error("Stopping pipeline due to error")
                    break
                else:
                    self.logger.warning("Continuing despite error")
        
        # Finalize pipeline
        self._finalize_pipeline()
        
        return {
            'status': 'completed' if not self.metadata['errors'] else 'completed_with_errors',
            'metadata': self.metadata,
            'data': self.data_frames,
            'features': self.feature_pipeline if hasattr(self, 'feature_pipeline') else None
        }
    
    def _acquire_data(self) -> Dict[str, pd.DataFrame]:
        """
        Step 1: Acquire data from configured sources.
        
        This method handles:
        - Multiple data source types (CSV, Parquet, Database, API)
        - Incremental or full data loading
        - Data validation and basic cleaning
        - Progress tracking
        
        Returns:
            Dictionary of DataFrames with acquired data
        """
        self.logger.info("Acquiring data from sources...")
        
        data_sources = self.config.get('data_sources', {})
        mode = self.config.get('pipeline.mode', 'full')
        batch_size = self.config.get('pipeline.batch_size', 10000)
        
        for source_name, source_config in data_sources.items():
            self.logger.info(f"Loading data source: {source_name}")
            
            try:
                source_type = source_config.get('type', 'csv')
                source_path = source_config.get('path')
                
                # Progress tracking
                pbar = tqdm(total=100, desc=f"Loading {source_name}", unit='%')
                
                # Load based on source type
                if source_type == 'csv':
                    # CSV file loading with optimizations
                    df = pd.read_csv(
                        source_path,
                        **source_config.get('read_params', {}),
                        nrows=source_config.get('sample_rows') if mode == 'sample' else None
                    )
                    
                elif source_type == 'parquet':
                    # Parquet file loading (faster, compressed)
                    df = pd.read_parquet(
                        source_path,
                        **source_config.get('read_params', {})
                    )
                    
                elif source_type == 'json':
                    # JSON file loading
                    df = pd.read_json(
                        source_path,
                        **source_config.get('read_params', {})
                    )
                    
                elif source_type == 'database':
                    # Database connection
                    connection_string = source_config.get('connection_string')
                    query = source_config.get('query')
                    
                    # Create database engine
                    engine = create_engine(connection_string)
                    
                    # Load in batches for large tables
                    if mode == 'incremental' and 'incremental_column' in source_config:
                        # Incremental loading based on timestamp
                        last_load = self._get_last_load_time(source_name)
                        query = query.replace('{{last_load}}', f"'{last_load}'")
                    
                    df = pd.read_sql_query(
                        query,
                        engine,
                        chunksize=batch_size
                    )
                    
                    # Concatenate chunks
                    df = pd.concat(df, ignore_index=True)
                    
                elif source_type == 'api':
                    # API data fetching
                    client = BankingDataClient(source_config)
                    df = client.fetch_data(
                        start_date=source_config.get('start_date'),
                        end_date=source_config.get('end_date'),
                        batch_size=batch_size
                    )
                    
                else:
                    self.logger.warning(f"Unsupported source type: {source_type}")
                    continue
                
                pbar.update(80)
                
                # Basic validation after loading
                if df is not None and not df.empty:
                    # Add source metadata
                    df.attrs['source'] = source_name
                    df.attrs['load_time'] = datetime.now()
                    df.attrs['record_count'] = len(df)
                    
                    # Store in data_frames dictionary
                    self.data_frames[source_name] = df
                    
                    self.logger.info(f"Loaded {len(df):,} records from {source_name}")
                    self.logger.info(f"Columns: {list(df.columns)}")
                    
                    # Quick data quality stats
                    self.logger.debug(f"Missing values: {df.isnull().sum().sum()}")
                    self.logger.debug(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                    
                else:
                    self.logger.warning(f"No data loaded from {source_name}")
                
                pbar.update(20)
                pbar.close()
                
            except Exception as e:
                self.logger.error(f"Error loading {source_name}: {str(e)}")
                self.metadata['warnings'].append({
                    'source': source_name,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                continue
        
        self.logger.info(f"Data acquisition complete. Loaded {len(self.data_frames)} sources")
        
        # Log summary
        for name, df in self.data_frames.items():
            self.logger.info(f"  - {name}: {len(df):,} records, {df.shape[1]} features")
        
        return self.data_frames
    
    def _validate_data(self) -> bool:
        """
        Step 2: Validate data quality and consistency.
        
        Performs:
        - Data quality checks using Great Expectations
        - Schema validation
        - Data type consistency
        - Missing value analysis
        - Duplicate detection
        - Statistical summary generation
        
        Returns:
            Boolean indicating if validation passed
        """
        self.logger.info("Validating data quality...")
        
        validation_results = {}
        all_valid = True
        
        for source_name, df in self.data_frames.items():
            self.logger.info(f"Validating: {source_name}")
            
            validation_result = {
                'source': source_name,
                'timestamp': datetime.now().isoformat(),
                'checks': {}
            }
            
            # 1. Basic validation
            validation_result['checks']['not_empty'] = len(df) > 0
            validation_result['checks']['has_columns'] = len(df.columns) > 0
            
            # 2. Missing value analysis
            missing_stats = df.isnull().sum()
            missing_percentages = (missing_stats / len(df)) * 100
            
            validation_result['missing_values'] = {
                'total_missing': int(missing_stats.sum()),
                'missing_percentage': float((missing_stats.sum() / (len(df) * len(df.columns))) * 100),
                'columns_with_missing': missing_stats[missing_stats > 0].to_dict()
            }
            
            # Check missing value threshold
            missing_threshold = self.config.get('data_quality.missing_threshold', 0.3)
            high_missing_cols = missing_percentages[missing_percentages > missing_threshold * 100]
            
            if not high_missing_cols.empty:
                self.logger.warning(f"Columns with >{missing_threshold*100}% missing: {list(high_missing_cols.index)}")
                validation_result['warnings'] = {
                    'high_missing_columns': high_missing_cols.to_dict()
                }
            
            # 3. Duplicate detection
            duplicate_count = df.duplicated().sum()
            duplicate_percentage = (duplicate_count / len(df)) * 100
            
            validation_result['duplicates'] = {
                'count': int(duplicate_count),
                'percentage': float(duplicate_percentage)
            }
            
            if duplicate_percentage > self.config.get('data_quality.duplicate_threshold', 10):
                self.logger.warning(f"High duplicate percentage: {duplicate_percentage:.2f}%")
            
            # 4. Data type validation
            dtypes = df.dtypes.astype(str).to_dict()
            validation_result['data_types'] = dtypes
            
            # 5. Statistical summary
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                stats_summary = df[numeric_cols].describe(percentiles=[.25, .5, .75, .95, .99]).to_dict()
                validation_result['statistics'] = stats_summary
                
                # Check for outliers using IQR method
                for col in numeric_cols[:10]:  # Limit to first 10 numeric columns
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outlier_threshold = self.config.get('data_quality.outlier_threshold', 1.5)
                    
                    lower_bound = Q1 - outlier_threshold * IQR
                    upper_bound = Q3 + outlier_threshold * IQR
                    
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                    
                    if len(outliers) > 0:
                        outlier_pct = (len(outliers) / len(df)) * 100
                        self.logger.debug(f"Column {col}: {len(outliers)} outliers ({outlier_pct:.2f}%)")
            
            # 6. Use Great Expectations for advanced validation
            if self.ge_context:
                try:
                    # Create expectation suite
                    suite = self.ge_context.create_expectation_suite(
                        expectation_suite_name=f"{source_name}_validation",
                        overwrite_existing=True
                    )
                    
                    # Add expectations based on configuration
                    if source_name in self.data_expectations:
                        for column, expectations in self.data_expectations[source_name].items():
                            if column in df.columns:
                                for expectation, params in expectations.items():
                                    # Add expectation to suite
                                    pass  # Simplified for example
                    
                    # Validate
                    # batch = self.ge_context.get_batch(df)
                    # results = batch.validate(expectation_suite_name=f"{source_name}_validation")
                    # validation_result['ge_results'] = results
                    
                except Exception as e:
                    self.logger.warning(f"Great Expectations validation failed: {e}")
            
            # Store validation results
            validation_results[source_name] = validation_result
            
            # Determine overall validity
            if not validation_result['checks']['not_empty']:
                all_valid = False
                self.metadata['errors'].append({
                    'source': source_name,
                    'error': 'Empty dataset',
                    'timestamp': datetime.now().isoformat()
                })
        
        # Store validation results in metadata
        self.metadata['validation_results'] = validation_results
        self.metadata['data_valid'] = all_valid
        
        self.logger.info(f"Data validation complete. All valid: {all_valid}")
        
        return all_valid
    
    def _preprocess_data(self) -> pd.DataFrame:
        """
        Step 3: Preprocess and clean the data.
        
        This comprehensive preprocessing includes:
        - Missing value imputation (multiple strategies)
        - Outlier detection and treatment
        - Data type conversion
        - Duplicate removal
        - Data standardization
        - Handling imbalanced classes
        
        Returns:
            Preprocessed DataFrame
        """
        self.logger.info("Preprocessing data...")
        
        # Combine all data sources if multiple
        if len(self.data_frames) > 1:
            self.logger.info("Merging multiple data sources...")
            merged_df = self._merge_data_sources()
        else:
            # Single data source
            source_name = list(self.data_frames.keys())[0]
            merged_df = self.data_frames[source_name].copy()
        
        self.logger.info(f"Initial data shape: {merged_df.shape}")
        
        # Initialize preprocessing pipeline
        preprocessing_pipeline = FraudPreprocessingPipeline(
            config=self.config.config
        )
        
        # 1. Handle missing values
        self.logger.info("Handling missing values...")
        missing_handler = MissingValueHandler(
            strategy=self.config.get('preprocessing.handle_missing', 'advanced'),
            missing_threshold=self.config.get('data_quality.missing_threshold', 0.3)
        )
        merged_df = missing_handler.fit_transform(merged_df)
        
        # Log missing value handling results
        missing_stats = missing_handler.get_statistics()
        self.logger.info(f"Missing values after handling: {missing_stats.get('remaining_missing', 0)}")
        
        # 2. Handle outliers
        self.logger.info("Detecting and handling outliers...")
        outlier_handler = OutlierHandler(
            method=self.config.get('preprocessing.handle_outliers', 'winsorize'),
            threshold=self.config.get('data_quality.outlier_threshold', 1.5)
        )
        merged_df = outlier_handler.fit_transform(merged_df)
        
        # Log outlier statistics
        outlier_stats = outlier_handler.get_statistics()
        self.logger.info(f"Outliers detected: {outlier_stats.get('outliers_detected', 0)}")
        self.logger.info(f"Outliers treated: {outlier_stats.get('outliers_treated', 0)}")
        
        # 3. Remove duplicates
        self.logger.info("Removing duplicates...")
        before_count = len(merged_df)
        merged_df = merged_df.drop_duplicates(
            subset=self._get_key_columns(merged_df),
            keep='first'
        )
        after_count = len(merged_df)
        self.logger.info(f"Removed {before_count - after_count} duplicate rows")
        
        # 4. Data type conversion
        self.logger.info("Converting data types...")
        
        # Identify column types
        date_columns = [col for col in merged_df.columns if 'time' in col.lower() or 'date' in col.lower()]
        categorical_columns = merged_df.select_dtypes(include=['object']).columns.tolist()
        numerical_columns = merged_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Convert date columns
        for col in date_columns:
            try:
                merged_df[col] = pd.to_datetime(merged_df[col])
                self.logger.debug(f"Converted {col} to datetime")
            except:
                self.logger.debug(f"Could not convert {col} to datetime")
        
        # Convert categorical columns
        for col in categorical_columns:
            merged_df[col] = merged_df[col].astype('category')
        
        # 5. Handle class imbalance if target column exists
        target_column = self._identify_target_column(merged_df)
        if target_column:
            self.logger.info(f"Target column identified: {target_column}")
            
            # Check class distribution
            class_dist = merged_df[target_column].value_counts()
            self.logger.info(f"Class distribution:\n{class_dist}")
            
            # Calculate imbalance ratio
            if len(class_dist) == 2:
                majority_count = class_dist.max()
                minority_count = class_dist.min()
                imbalance_ratio = majority_count / minority_count
                
                self.logger.info(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
                
                # Apply SMOTE if imbalance is severe
                if imbalance_ratio > 10:
                    self.logger.info("Severe imbalance detected. Applying SMOTE...")
                    
                    # Separate features and target
                    X = merged_df.drop(columns=[target_column])
                    y = merged_df[target_column]
                    
                    # Select only numeric columns for SMOTE
                    numeric_X = X.select_dtypes(include=[np.number])
                    
                    if len(numeric_X.columns) > 0:
                        # Apply SMOTE
                        smote = SMOTE(random_state=42)
                        X_resampled, y_resampled = smote.fit_resample(numeric_X, y)
                        
                        # Reconstruct DataFrame
                        merged_df = pd.concat([
                            pd.DataFrame(X_resampled, columns=numeric_X.columns),
                            pd.Series(y_resampled, name=target_column)
                        ], axis=1)
                        
                        self.logger.info(f"After SMOTE - Shape: {merged_df.shape}")
                        self.logger.info(f"New class distribution:\n{merged_df[target_column].value_counts()}")
        
        # 6. Feature scaling
        self.logger.info("Scaling numerical features...")
        scaler = NumericalScaler(
            method=self.config.get('preprocessing.scaling_method', 'robust')
        )
        
        numerical_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column and target_column in numerical_cols:
            numerical_cols.remove(target_column)
        
        if numerical_cols:
            merged_df[numerical_cols] = scaler.fit_transform(merged_df[numerical_cols])
        
        # Store preprocessing pipeline for later use
        self.preprocessing_pipeline = preprocessing_pipeline
        
        # Update data_frames with preprocessed data
        self.data_frames['preprocessed'] = merged_df
        
        self.logger.info(f"Preprocessing complete. Final shape: {merged_df.shape}")
        self.logger.info(f"Memory usage: {merged_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return merged_df
    
    def _merge_data_sources(self) -> pd.DataFrame:
        """
        Merge multiple data sources intelligently.
        
        Handles different merge strategies:
        - Left merge for primary-secondary relationships
        - Inner merge for strict matching
        - Outer merge for complete data
        
        Returns:
            Merged DataFrame
        """
        self.logger.info("Merging data sources...")
        
        # Determine merge strategy based on data sources
        source_names = list(self.data_frames.keys())
        
        # Assume first source is primary (usually transactions)
        primary_source = source_names[0]
        merged_df = self.data_frames[primary_source].copy()
        
        self.logger.info(f"Primary source: {primary_source} ({len(merged_df)} records)")
        
        # Merge other sources
        for source_name in source_names[1:]:
            source_df = self.data_frames[source_name]
            
            # Find common key columns
            common_keys = set(merged_df.columns).intersection(set(source_df.columns))
            
            # Filter to likely key columns (id fields)
            key_candidates = [col for col in common_keys if 'id' in col.lower()]
            
            if key_candidates:
                # Use first key candidate for merging
                merge_key = key_candidates[0]
                self.logger.info(f"Merging {source_name} on key: {merge_key}")
                
                # Determine merge type based on data
                # Check if source is likely a dimension table (unique keys)
                if source_df[merge_key].is_unique:
                    merge_type = 'left'  # Dimension table
                else:
                    merge_type = 'inner'  # Fact table
                
                # Perform merge
                before_merge = len(merged_df)
                merged_df = merged_df.merge(
                    source_df,
                    on=merge_key,
                    how=merge_type,
                    suffixes=(f'_{primary_source}', f'_{source_name}')
                )
                
                self.logger.info(f"After merge: {len(merged_df)} records ({len(merged_df) - before_merge} change)")
            else:
                # No common key - append if same structure
                if set(merged_df.columns) == set(source_df.columns):
                    merged_df = pd.concat([merged_df, source_df], ignore_index=True)
                    self.logger.info(f"Appended {source_name} (same structure)")
                else:
                    self.logger.warning(f"Cannot merge {source_name} - no common keys or structure")
        
        return merged_df
    
    def _get_key_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Identify key columns for deduplication.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of key column names
        """
        key_columns = []
        
        # Look for id columns
        id_columns = [col for col in df.columns if 'id' in col.lower()]
        
        if id_columns:
            # Check if any id column uniquely identifies rows
            for col in id_columns:
                if df[col].is_unique:
                    return [col]
            
            # Otherwise, use combination of id columns
            key_columns = id_columns
        else:
            # Use all columns for deduplication
            key_columns = df.columns.tolist()
        
        return key_columns
    
    def _identify_target_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Identify the target column for fraud detection.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Name of target column or None
        """
        # Common names for fraud indicator
        fraud_indicators = ['is_fraud', 'fraud', 'label', 'class', 'target']
        
        for indicator in fraud_indicators:
            if indicator in df.columns:
                return indicator
            
            # Check for columns containing these words
            for col in df.columns:
                if indicator in col.lower():
                    return col
        
        return None
    
    def _run_eda(self) -> Dict:
        """
        Step 4: Run Exploratory Data Analysis.
        
        Comprehensive EDA including:
        - Statistical summaries
        - Distribution analysis
        - Correlation analysis
        - Temporal patterns
        - Segmentation analysis
        - Visualization generation
        
        Returns:
            Dictionary with EDA results
        """
        self.logger.info("Running Exploratory Data Analysis...")
        
        # Get preprocessed data
        df = self.data_frames.get('preprocessed')
        if df is None:
            # Use first available data source
            df = list(self.data_frames.values())[0]
        
        self.logger.info(f"EDA on dataset: {df.shape}")
        
        # Initialize EDA components
        eda_results = {}
        
        # 1. Basic statistical analysis
        self.logger.info("1. Computing statistical summaries...")
        stats_analyzer = FraudStatisticalAnalysis(df)
        eda_results['statistics'] = stats_analyzer.perform_full_analysis()
        
        # 2. Correlation analysis
        self.logger.info("2. Analyzing correlations...")
        corr_analyzer = CorrelationAnalysis(df)
        eda_results['correlations'] = corr_analyzer.analyze()
        
        # 3. Temporal analysis (if time columns exist)
        time_columns = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
        if time_columns:
            self.logger.info("3. Analyzing temporal patterns...")
            temporal_analyzer = TemporalAnalysis(df, time_column=time_columns[0])
            eda_results['temporal'] = temporal_analyzer.analyze()
        
        # 4. Identify target for segmentation
        target_column = self._identify_target_column(df)
        
        if target_column:
            self.logger.info(f"4. Analyzing by target: {target_column}")
            
            # Compare fraud vs non-fraud
            fraud_df = df[df[target_column] == 1]
            non_fraud_df = df[df[target_column] == 0]
            
            eda_results['segmentation'] = {
                'fraud_count': len(fraud_df),
                'fraud_percentage': (len(fraud_df) / len(df)) * 100,
                'non_fraud_count': len(non_fraud_df),
                'non_fraud_percentage': (len(non_fraud_df) / len(df)) * 100,
                'imbalance_ratio': len(non_fraud_df) / len(fraud_df) if len(fraud_df) > 0 else float('inf')
            }
            
            # Compare statistics between groups
            if len(fraud_df) > 0 and len(non_fraud_df) > 0:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                comparison = {}
                for col in numeric_cols[:10]:  # Limit to first 10 numeric columns
                    if col != target_column:
                        comparison[col] = {
                            'fraud_mean': fraud_df[col].mean(),
                            'fraud_std': fraud_df[col].std(),
                            'non_fraud_mean': non_fraud_df[col].mean(),
                            'non_fraud_std': non_fraud_df[col].std(),
                            'difference': fraud_df[col].mean() - non_fraud_df[col].mean()
                        }
                
                eda_results['group_comparison'] = comparison
        
        # 5. Generate visualizations
        self.logger.info("5. Generating visualizations...")
        viz_generator = VisualizationGenerator(
            save_path='reports/eda/',
            style='seaborn'
        )
        
        visualization_files = viz_generator.generate_all(
            df=df,
            target_column=target_column
        )
        
        eda_results['visualizations'] = visualization_files
        
        # 6. Feature analysis
        self.logger.info("6. Analyzing features...")
        
        # Identify feature types
        feature_analysis = {
            'numerical_features': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_features': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'datetime_features': df.select_dtypes(include=['datetime64']).columns.tolist(),
            'feature_count': len(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        eda_results['features'] = feature_analysis
        
        # 7. Generate HTML report
        self.logger.info("7. Generating HTML report...")
        report_path = self._generate_eda_report(eda_results, df)
        eda_results['report_path'] = report_path
        
        # Store EDA results
        self.metadata['eda'] = eda_results
        
        self.logger.info(f"EDA complete. Report saved to: {report_path}")
        
        return eda_results
    
    def _generate_eda_report(self, eda_results: Dict, df: pd.DataFrame) -> str:
        """
        Generate comprehensive HTML EDA report.
        
        Args:
            eda_results: Dictionary with EDA results
            df: DataFrame used for analysis
            
        Returns:
            Path to generated report
        """
        from jinja2 import Template
        
        # HTML template for report
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>VeritasFinancial - EDA Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #2c3e50; border-bottom: 2px solid #3498db; }
                h2 { color: #34495e; margin-top: 30px; }
                .summary { background: #ecf0f1; padding: 20px; border-radius: 5px; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background: white; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                .metric-value { font-size: 24px; font-weight: bold; color: #3498db; }
                .metric-label { font-size: 12px; color: #7f8c8d; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #3498db; color: white; }
                .warning { background-color: #f39c12; color: white; padding: 10px; border-radius: 5px; }
                .error { background-color: #e74c3c; color: white; padding: 10px; border-radius: 5px; }
                .success { background-color: #27ae60; color: white; padding: 10px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>VeritasFinancial - Fraud Detection EDA Report</h1>
            <p>Generated: {{ generation_time }}</p>
            
            <div class="summary">
                <h2>Dataset Overview</h2>
                <div class="metric">
                    <div class="metric-value">{{ dataset_stats.rows }}</div>
                    <div class="metric-label">Rows</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{{ dataset_stats.columns }}</div>
                    <div class="metric-label">Columns</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{{ "%.2f"|format(dataset_stats.memory) }}</div>
                    <div class="metric-label">Memory (MB)</div>
                </div>
            </div>
            
            <h2>Class Distribution</h2>
            {% if segmentation %}
            <table>
                <tr>
                    <th>Class</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
                <tr>
                    <td>Fraud</td>
                    <td>{{ segmentation.fraud_count }}</td>
                    <td>{{ "%.2f"|format(segmentation.fraud_percentage) }}%</td>
                </tr>
                <tr>
                    <td>Non-Fraud</td>
                    <td>{{ segmentation.non_fraud_count }}</td>
                    <td>{{ "%.2f"|format(segmentation.non_fraud_percentage) }}%</td>
                </tr>
                <tr>
                    <td colspan="3">Imbalance Ratio: {{ "%.2f"|format(segmentation.imbalance_ratio) }}:1</td>
                </tr>
            </table>
            {% endif %}
            
            <h2>Feature Types</h2>
            <table>
                <tr>
                    <th>Feature Type</th>
                    <th>Count</th>
                </tr>
                <tr>
                    <td>Numerical</td>
                    <td>{{ features.numerical_features|length }}</td>
                </tr>
                <tr>
                    <td>Categorical</td>
                    <td>{{ features.categorical_features|length }}</td>
                </tr>
                <tr>
                    <td>Datetime</td>
                    <td>{{ features.datetime_features|length }}</td>
                </tr>
            </table>
            
            <h2>Data Quality Issues</h2>
            {% if warnings %}
            <div class="warning">
                <h3>Warnings</h3>
                <ul>
                {% for warning in warnings %}
                    <li>{{ warning }}</li>
                {% endfor %}
                </ul>
            </div>
            {% endif %}
            
            <h2>Recommendations</h2>
            <ul>
                {% if segmentation.fraud_percentage < 1 %}
                <li>Severe class imbalance detected. Use SMOTE or class weights.</li>
                {% endif %}
                {% if dataset_stats.columns > 100 %}
                <li>High dimensionality. Consider feature selection.</li>
                {% endif %}
                {% if warnings|length > 0 %}
                <li>Address data quality issues before modeling.</li>
                {% endif %}
            </ul>
            
            <h2>Visualizations</h2>
            {% for viz in visualizations %}
            <div>
                <img src="{{ viz }}" style="max-width: 100%; margin: 10px 0;">
            </div>
            {% endfor %}
        </body>
        </html>
        """
        
        # Prepare template data
        template_data = {
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_stats': {
                'rows': len(df),
                'columns': len(df.columns),
                'memory': df.memory_usage(deep=True).sum() / 1024**2
            },
            'segmentation': eda_results.get('segmentation', {}),
            'features': eda_results.get('features', {}),
            'warnings': self.metadata.get('warnings', []),
            'visualizations': eda_results.get('visualizations', [])
        }
        
        # Render template
        template = Template(html_template)
        html_content = template.render(**template_data)
        
        # Save report
        report_path = f"reports/eda/eda_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return report_path
    
    def _engineer_features(self) -> pd.DataFrame:
        """
        Step 5: Engineer features for fraud detection.
        
        Creates comprehensive feature set including:
        - Transaction-based features
        - Customer behavior features
        - Temporal features
        - Interaction features
        - Aggregated features
        - Risk scores
        
        Returns:
            DataFrame with engineered features
        """
        self.logger.info("Engineering features for fraud detection...")
        
        # Get preprocessed data
        df = self.data_frames.get('preprocessed')
        if df is None:
            df = list(self.data_frames.values())[0]
        
        self.logger.info(f"Starting feature engineering on {df.shape}")
        
        # Initialize feature engineering pipeline
        feature_pipeline = {}
        
        # 1. Transaction-based features
        self.logger.info("1. Creating transaction-based features...")
        transaction_engineer = TransactionFeatureEngineer()
        df = transaction_engineer.create_all_features(df)
        feature_pipeline['transaction_features'] = transaction_engineer.get_feature_names()
        self.logger.info(f"   Added {len(transaction_engineer.get_feature_names())} transaction features")
        
        # 2. Customer behavior features
        self.logger.info("2. Creating customer behavior features...")
        if 'customer_id' in df.columns:
            behavior_engineer = BehavioralFeatureEngineer()
            df = behavior_engineer.create_all_features(df)
            feature_pipeline['behavioral_features'] = behavior_engineer.get_feature_names()
            self.logger.info(f"   Added {len(behavior_engineer.get_feature_names())} behavioral features")
        
        # 3. Temporal features
        self.logger.info("3. Creating temporal features...")
        temporal_engineer = TemporalFeatureEngineer()
        df = temporal_engineer.create_all_features(df)
        feature_pipeline['temporal_features'] = temporal_engineer.get_feature_names()
        self.logger.info(f"   Added {len(temporal_engineer.get_feature_names())} temporal features")
        
        # 4. Aggregated features
        self.logger.info("4. Creating aggregated features...")
        # Group by customer if available
        if 'customer_id' in df.columns:
            # Customer-level aggregates
            customer_agg = df.groupby('customer_id').agg({
                'amount': ['mean', 'std', 'min', 'max', 'count'],
                'transaction_time': lambda x: (x.max() - x.min()).total_seconds() if len(x) > 1 else 0
            }).reset_index()
            
            # Flatten column names
            customer_agg.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in customer_agg.columns.values]
            
            # Merge back
            df = df.merge(customer_agg, on='customer_id', how='left')
            feature_pipeline['aggregated_features'] = customer_agg.columns.tolist()
            self.logger.info(f"   Added {len(customer_agg.columns)} aggregated features")
        
        # 5. Interaction features
        self.logger.info("5. Creating interaction features...")
        # Create interactions between important features
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]  # Limit to first 10
        
        if len(numeric_cols) >= 2:
            interaction_features = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    col1 = numeric_cols[i]
                    col2 = numeric_cols[j]
                    
                    # Multiplication interaction
                    interaction_name = f"{col1}_x_{col2}"
                    df[interaction_name] = df[col1] * df[col2]
                    interaction_features.append(interaction_name)
            
            feature_pipeline['interaction_features'] = interaction_features
            self.logger.info(f"   Added {len(interaction_features)} interaction features")
        
        # 6. Risk scores
        self.logger.info("6. Creating risk scores...")
        
        # Transaction amount risk
        if 'amount' in df.columns:
            # Amount percentile risk
            df['amount_risk_score'] = pd.qcut(df['amount'].rank(method='first'), 
                                              q=10, labels=False, duplicates='drop') / 10
            
            # Amount deviation risk
            if 'avg_transaction_amount' in df.columns:
                df['amount_deviation_risk'] = np.abs(df['amount'] - df['avg_transaction_amount']) / (df['avg_transaction_amount'] + 1)
        
        # Time risk (late night transactions)
        if 'hour_of_day' in df.columns:
            df['time_risk_score'] = np.where(
                (df['hour_of_day'] < 6) | (df['hour_of_day'] > 22),
                0.8,  # High risk for late night
                0.2   # Low risk for daytime
            )
        
        # Velocity risk
        if 'transactions_last_hour' in df.columns:
            df['velocity_risk_score'] = np.where(
                df['transactions_last_hour'] > 5,
                0.9,  # High risk for high velocity
                np.where(df['transactions_last_hour'] > 2, 0.5, 0.1)
            )
        
        feature_pipeline['risk_scores'] = ['amount_risk_score', 'time_risk_score', 'velocity_risk_score']
        
        # 7. Feature selection (if too many features)
        max_features = self.config.get('feature_engineering.max_features', 500)
        if df.shape[1] > max_features:
            self.logger.info(f"7. Selecting top {max_features} features...")
            
            # Identify target for feature selection
            target_column = self._identify_target_column(df)
            
            if target_column:
                # Use mutual information for feature selection
                X = df.drop(columns=[target_column])
                y = df[target_column]
                
                # Select only numeric columns
                numeric_X = X.select_dtypes(include=[np.number])
                
                if numeric_X.shape[1] > max_features:
                    from sklearn.feature_selection import SelectKBest, mutual_info_classif
                    
                    selector = SelectKBest(score_func=mutual_info_classif, k=min(max_features, numeric_X.shape[1]))
                    selector.fit(numeric_X, y)
                    
                    # Get selected features
                    selected_mask = selector.get_support()
                    selected_features = numeric_X.columns[selected_mask].tolist()
                    
                    # Keep only selected features plus target
                    columns_to_keep = selected_features + [target_column]
                    df = df[columns_to_keep]
                    
                    feature_pipeline['selected_features'] = selected_features
                    self.logger.info(f"   Selected {len(selected_features)} features using mutual information")
        
        # Store feature pipeline
        self.feature_pipeline = feature_pipeline
        
        # Update data_frames with engineered features
        self.data_frames['features'] = df
        
        self.logger.info(f"Feature engineering complete. Final shape: {df.shape}")
        self.logger.info(f"Total features created: {df.shape[1]}")
        
        # Save feature list for documentation
        self._save_feature_list(df.columns.tolist())
        
        return df
    
    def _save_feature_list(self, features: List[str]):
        """
        Save feature list for documentation.
        
        Args:
            features: List of feature names
        """
        feature_doc = {
            'generated_at': datetime.now().isoformat(),
            'total_features': len(features),
            'features': features
        }
        
        # Save as JSON
        with open('artifacts/feature_list.json', 'w') as f:
            json.dump(feature_doc, f, indent=2)
        
        # Save as CSV for easy viewing
        pd.DataFrame({'feature_name': features}).to_csv('artifacts/feature_list.csv', index=False)
        
        self.logger.info(f"Feature list saved to artifacts/feature_list.csv")
    
    def _save_results(self) -> Dict:
        """
        Step 6: Save all pipeline results.
        
        Saves:
        - Processed data in multiple formats
        - Feature-engineered data
        - Pipeline artifacts and metadata
        - EDA reports
        - Configuration used
        
        Returns:
            Dictionary with saved file paths
        """
        self.logger.info("Saving pipeline results...")
        
        saved_files = {}
        
        # Get output configuration
        processed_path = self.config.get('output.processed_data_path', 'data/processed/')
        features_path = self.config.get('output.features_path', 'data/features/')
        save_format = self.config.get('output.save_format', 'parquet')
        compression = self.config.get('output.compression', 'snappy')
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Save preprocessed data
        if 'preprocessed' in self.data_frames:
            df = self.data_frames['preprocessed']
            
            # Save in specified format
            if save_format == 'parquet':
                file_path = f"{processed_path}preprocessed_data_{timestamp}.parquet"
                df.to_parquet(file_path, compression=compression, index=False)
            elif save_format == 'csv':
                file_path = f"{processed_path}preprocessed_data_{timestamp}.csv"
                df.to_csv(file_path, index=False)
            elif save_format == 'feather':
                file_path = f"{processed_path}preprocessed_data_{timestamp}.feather"
                df.to_feather(file_path)
            
            saved_files['preprocessed'] = file_path
            self.logger.info(f"Preprocessed data saved to: {file_path}")
            
            # Save sample for quick viewing
            sample_path = f"{processed_path}preprocessed_data_sample_{timestamp}.csv"
            df.head(1000).to_csv(sample_path, index=False)
            saved_files['preprocessed_sample'] = sample_path
        
        # 2. Save feature-engineered data
        if 'features' in self.data_frames:
            df = self.data_frames['features']
            
            # Save in specified format
            if save_format == 'parquet':
                file_path = f"{features_path}features_data_{timestamp}.parquet"
                df.to_parquet(file_path, compression=compression, index=False)
            elif save_format == 'csv':
                file_path = f"{features_path}features_data_{timestamp}.csv"
                df.to_csv(file_path, index=False)
            
            saved_files['features'] = file_path
            self.logger.info(f"Features data saved to: {file_path}")
            
            # Save feature statistics
            stats_path = f"{features_path}feature_statistics_{timestamp}.csv"
            df.describe().to_csv(stats_path)
            saved_files['feature_stats'] = stats_path
        
        # 3. Save pipeline metadata
        self.metadata['end_time'] = datetime.now().isoformat()
        self.metadata['elapsed_time'] = (self.metadata['end_time'] - self.metadata['start_time']).total_seconds()
        
        metadata_path = f"artifacts/pipeline_metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            # Convert datetime objects to strings
            metadata_copy = self.metadata.copy()
            if 'start_time' in metadata_copy:
                metadata_copy['start_time'] = metadata_copy['start_time'].isoformat()
            if 'end_time' in metadata_copy:
                metadata_copy['end_time'] = metadata_copy['end_time'].isoformat()
            
            json.dump(metadata_copy, f, indent=2, default=str)
        
        saved_files['metadata'] = metadata_path
        self.logger.info(f"Pipeline metadata saved to: {metadata_path}")
        
        # 4. Save configuration used
        config_path = f"artifacts/pipeline_config_{timestamp}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config.config, f, default_flow_style=False)
        
        saved_files['config'] = config_path
        self.logger.info(f"Configuration saved to: {config_path}")
        
        # 5. Save data splits for modeling
        if 'features' in self.data_frames:
            df = self.data_frames['features']
            target_column = self._identify_target_column(df)
            
            if target_column:
                # Create train/validation/test splits
                X = df.drop(columns=[target_column])
                y = df[target_column]
                
                # First split: 70% train, 30% temp
                X_train, X_temp, y_train, y_temp = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )
                
                # Second split: 15% validation, 15% test from temp
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
                )
                
                # Save splits
                splits = {
                    'X_train': X_train,
                    'X_val': X_val,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_val': y_val,
                    'y_test': y_test
                }
                
                splits_path = f"{features_path}data_splits_{timestamp}.pkl"
                with open(splits_path, 'wb') as f:
                    pickle.dump(splits, f)
                
                saved_files['splits'] = splits_path
                self.logger.info(f"Data splits saved to: {splits_path}")
                
                # Log split sizes
                self.logger.info(f"Train size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
                self.logger.info(f"Validation size: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
                self.logger.info(f"Test size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
        
        self.logger.info("All results saved successfully")
        
        return saved_files
    
    def _finalize_pipeline(self):
        """
        Finalize pipeline execution.
        Logs summary and cleans up temporary files.
        """
        self.logger.info("=" * 80)
        self.logger.info("PIPELINE EXECUTION COMPLETE")
        self.logger.info("=" * 80)
        
        # Calculate total time
        end_time = datetime.now()
        total_time = (end_time - self.metadata['start_time']).total_seconds()
        
        # Log summary
        self.logger.info(f"Pipeline ID: {self.metadata['pipeline_id']}")
        self.logger.info(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        self.logger.info(f"Steps completed: {len(self.metadata['steps_completed'])}")
        self.logger.info(f"Warnings: {len(self.metadata['warnings'])}")
        self.logger.info(f"Errors: {len(self.metadata['errors'])}")
        
        if self.metadata['errors']:
            self.logger.warning("Pipeline completed with errors. Check logs for details.")
        else:
            self.logger.info("Pipeline completed successfully!")
        
        # Log data statistics
        if 'features' in self.data_frames:
            df = self.data_frames['features']
            self.logger.info(f"Final feature dataset: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
        
        self.logger.info("-" * 80)
    
    def _get_last_load_time(self, source_name: str) -> str:
        """
        Get last load time for incremental loading.
        
        Args:
            source_name: Name of data source
            
        Returns:
            ISO format timestamp of last load
        """
        # In production, this would read from a metadata store
        # For now, return default
        return (datetime.now() - timedelta(days=1)).isoformat()


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """
    Main execution function for the data pipeline.
    
    Handles:
    - Command line argument parsing
    - Pipeline initialization and execution
    - Error handling and logging
    - Return code management
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='VeritasFinancial - Data Pipeline for Fraud Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_data_pipeline.py --config configs/data_config.yaml
  python run_data_pipeline.py --steps acquire validate --mode sample
  python run_data_pipeline.py --help
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/data_config.yaml',
        help='Path to configuration file (default: configs/data_config.yaml)'
    )
    
    parser.add_argument(
        '--steps', '-s',
        type=str,
        nargs='+',
        choices=['acquire', 'validate', 'preprocess', 'eda', 'engineer', 'save'],
        help='Specific steps to run (if not provided, runs all steps)'
    )
    
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['full', 'sample', 'incremental'],
        default='full',
        help='Pipeline execution mode (default: full)'
    )
    
    parser.add_argument(
        '--sample-size', '-n',
        type=int,
        default=10000,
        help='Sample size for sample mode (default: 10000)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Set up basic logging before pipeline initialization
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("=" * 80)
        logger.info("VERITASFINANCIAL - DATA PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Arguments: {args}")
        
        # Initialize and run pipeline
        pipeline = DataPipeline(config_path=args.config)
        
        # Override mode from command line
        if args.mode:
            pipeline.config.config['pipeline']['mode'] = args.mode
        
        # Override sample size if in sample mode
        if args.mode == 'sample':
            for source_name in pipeline.config.config['data_sources']:
                pipeline.config.config['data_sources'][source_name]['sample_rows'] = args.sample_size
        
        # Run pipeline
        results = pipeline.run(steps=args.steps)
        
        logger.info("-" * 80)
        logger.info("Pipeline execution summary:")
        logger.info(f"  Status: {results['status']}")
        
        if 'metadata' in results:
            metadata = results['metadata']
            logger.info(f"  Steps completed: {len(metadata.get('steps_completed', []))}")
            logger.info(f"  Warnings: {len(metadata.get('warnings', []))}")
            logger.info(f"  Errors: {len(metadata.get('errors', []))}")
        
        logger.info("=" * 80)
        
        # Return appropriate exit code
        if results['status'] == 'completed':
            return 0
        else:
            return 1
            
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        return 1
    finally:
        logger.info(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    """
    Script entry point.
    Executes main function and exits with appropriate return code.
    """
    sys.exit(main())