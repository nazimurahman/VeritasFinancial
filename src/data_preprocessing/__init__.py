"""
VeritasFinancial - Banking Fraud Detection System
Data Preprocessing Module

This module handles all data preprocessing operations for banking fraud detection including:
- Data cleaning for transactions, customers, and devices
- Feature transformations and encoding
- Handling missing values, outliers, and class imbalance
- End-to-end preprocessing pipelines

Author: VeritasFinancial Team
Version: 1.0.0
"""

# Import main classes for easy access
from .cleaners.transaction_cleaner import TransactionCleaner
from .cleaners.customer_cleaner import CustomerCleaner
from .cleaners.device_cleaner import DeviceCleaner

from .transformers.categorical_encoder import CategoricalEncoder
from .transformers.numerical_scaler import NumericalScaler
from .transformers.datetime_processor import DateTimeProcessor

from .handlers.missing_values import MissingValueHandler
from .handlers.outliers import OutlierHandler
from .handlers.imbalance import ImbalanceHandler

from .pipelines.preprocessing_pipeline import PreprocessingPipeline
from .pipelines.feature_pipeline import FeaturePipeline

# Define module version
__version__ = '1.0.0'

# Define what gets imported with "from data_preprocessing import *"
__all__ = [
    'TransactionCleaner',
    'CustomerCleaner',
    'DeviceCleaner',
    'CategoricalEncoder',
    'NumericalScaler',
    'DateTimeProcessor',
    'MissingValueHandler',
    'OutlierHandler',
    'ImbalanceHandler',
    'PreprocessingPipeline',
    'FeaturePipeline'
]