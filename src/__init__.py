"""
VeritasFinancial: Banking Fraud Detection System
================================================
A comprehensive machine learning system for detecting fraudulent transactions in banking systems.

This package provides end-to-end capabilities for:
- Data acquisition from various banking sources
- Advanced preprocessing and feature engineering
- Comprehensive exploratory data analysis
- Multiple modeling approaches (classical ML to Transformers)
- Production deployment with monitoring
"""

__version__ = "1.0.0"
__author__ = "VeritasFinancial Team"
__license__ = "Proprietary"

# Import main modules for easy access
from zipfile import Path

from src import (
    data_acquisition,
    data_preprocessing,
    exploratory_analysis,
    feature_engineering,
    modeling,
    deployment,
    utils
)

# Define what gets imported with "from src import *"
__all__ = [
    'data_acquisition',
    'data_preprocessing',
    'exploratory_analysis',
    'feature_engineering',
    'modeling',
    'deployment',
    'utils'
]

# Package-level configuration
PACKAGE_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PACKAGE_ROOT / 'configs'
DATA_PATH = PACKAGE_ROOT / 'data'
ARTIFACTS_PATH = PACKAGE_ROOT / 'artifacts'

# Initialize logging
from src.utils.logger import setup_logging
logger = setup_logging(__name__)
logger.info(f"VeritasFinancial v{__version__} initialized")
logger.info(f"Package root: {PACKAGE_ROOT}")