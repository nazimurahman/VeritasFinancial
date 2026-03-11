"""
Transaction Data Cleaner Module

This module handles cleaning and validation of banking transaction data.
Transaction data is the core of fraud detection and requires meticulous cleaning
to ensure data quality and consistency.

Key functionalities:
1. Transaction ID validation and deduplication
2. Amount validation and formatting
3. Currency code standardization
4. Timestamp cleaning and timezone handling
5. Merchant information validation
6. Geographic data validation (country, city)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Union
import re

# Configure logging for the cleaner
logger = logging.getLogger(__name__)


class TransactionCleaner:
    """
    Comprehensive transaction data cleaner for banking fraud detection.
    
    This class implements industry-standard cleaning procedures for banking
    transaction data, ensuring data quality and consistency before feature
    engineering and model training.
    
    Attributes:
        config (dict): Configuration parameters for cleaning operations
        cleaned_stats (dict): Statistics about cleaning operations performed
        valid_currencies (list): List of valid currency codes
        min_amount (float): Minimum valid transaction amount
        max_amount (float): Maximum valid transaction amount
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the TransactionCleaner with configuration.
        
        Args:
            config: Dictionary containing cleaning parameters
                   If None, default configuration will be used
        """
        # Default configuration for transaction cleaning
        self.config = config or {
            'min_amount': 0.01,  # Minimum transaction amount (avoid zero/negative)
            'max_amount': 1000000,  # Maximum transaction amount (reasonable upper bound)
            'valid_currencies': ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'CNY', 'HKD', 'SGD'],
            'max_future_days': 1,  # Allow transactions up to 1 day in future (timezone issues)
            'max_past_years': 5,  # Maximum age of historical transactions
            'remove_duplicates': True,
            'validate_merchant': True,
            'validate_location': True,
            'standardize_timestamps': True
        }
        
        # Initialize statistics tracking
        self.cleaned_stats = {
            'total_transactions_processed': 0,
            'duplicates_removed': 0,
            'invalid_amounts_removed': 0,
            'invalid_currencies_removed': 0,
            'invalid_dates_removed': 0,
            'invalid_merchants_removed': 0,
            'invalid_locations_removed': 0,
            'null_values_filled': 0
        }
        
        # Derived attributes from config
        self.valid_currencies = self.config['valid_currencies']
        self.min_amount = self.config['min_amount']
        self.max_amount = self.config['max_amount']
        
        logger.info("TransactionCleaner initialized with config: %s", self.config)
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main cleaning method that orchestrates all cleaning operations.
        
        Args:
            df: Raw transaction DataFrame
            
        Returns:
            Cleaned transaction DataFrame
        """
        logger.info(f"Starting transaction cleaning on {len(df)} records")
        
        # Make a copy to avoid modifying original data
        cleaned_df = df.copy()
        
        # Track initial row count
        initial_count = len(cleaned_df)
        
        # Apply cleaning steps in sequence
        cleaned_df = self._validate_transaction_id(cleaned_df)
        cleaned_df = self._clean_amount(cleaned_df)
        cleaned_df = self._clean_currency(cleaned_df)
        cleaned_df = self._clean_timestamp(cleaned_df)
        cleaned_df = self._clean_merchant_info(cleaned_df)
        cleaned_df = self._clean_location_info(cleaned_df)
        
        if self.config['remove_duplicates']:
            cleaned_df = self._remove_duplicates(cleaned_df)
        
        # Remove rows with critical missing data
        cleaned_df = self._remove_invalid_rows(cleaned_df)
        
        # Update statistics
        self.cleaned_stats['total_transactions_processed'] = initial_count
        final_count = len(cleaned_df)
        logger.info(f"Cleaning complete. Kept {final_count}/{initial_count} records")
        
        return cleaned_df
    
    def _validate_transaction_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean transaction IDs.
        
        Transaction IDs should be:
        - Non-null
        - Unique (handled later)
        - Proper format (if specified)
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with validated transaction IDs
        """
        if 'transaction_id' not in df.columns:
            logger.warning("No transaction_id column found. Generating IDs.")
            df['transaction_id'] = [f"TXN_{i:010d}" for i in range(len(df))]
            return df
        
        # Convert to string and strip whitespace
        df['transaction_id'] = df['transaction_id'].astype(str).str.strip()
        
        # Remove completely empty IDs
        empty_mask = (df['transaction_id'] == '') | (df['transaction_id'] == 'nan') | (df['transaction_id'].isna())
        if empty_mask.any():
            logger.warning(f"Found {empty_mask.sum()} empty transaction IDs")
            # Generate new IDs for empty ones
            df.loc[empty_mask, 'transaction_id'] = [f"TXN_{i:010d}" for i in range(empty_mask.sum())]
            self.cleaned_stats['null_values_filled'] += empty_mask.sum()
        
        return df
    
    def _clean_amount(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate transaction amounts.
        
        Operations performed:
        1. Convert to numeric
        2. Handle negative amounts (abs or flag)
        3. Remove outliers beyond reasonable bounds
        4. Create flags for unusual amounts
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with cleaned amount column
        """
        if 'amount' not in df.columns:
            logger.error("No amount column found in transaction data")
            raise ValueError("Transaction data must contain 'amount' column")
        
        # Convert to numeric, coercing errors to NaN
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Track original count
        original_count = len(df)
        
        # Handle negative amounts
        negative_mask = df['amount'] < 0
        if negative_mask.any():
            logger.warning(f"Found {negative_mask.sum()} negative amounts. Converting to absolute values.")
            df.loc[negative_mask, 'amount'] = df.loc[negative_mask, 'amount'].abs()
            # Add flag for reversed transactions (negative amounts might indicate reversals)
            df['was_negative'] = negative_mask.astype(int)
        
        # Flag zero amounts (potentially suspicious)
        df['is_zero_amount'] = (df['amount'] == 0).astype(int)
        
        # Remove amounts outside reasonable bounds
        invalid_amount_mask = (df['amount'] < self.min_amount) | (df['amount'] > self.max_amount)
        if invalid_amount_mask.any():
            logger.warning(f"Found {invalid_amount_mask.sum()} amounts outside valid range")
            # Instead of removing, we flag them for review
            df['amount_out_of_bounds'] = invalid_amount_mask.astype(int)
            
            # Clip extreme values to bounds (optional - depends on business rules)
            df.loc[df['amount'] < self.min_amount, 'amount'] = self.min_amount
            df.loc[df['amount'] > self.max_amount, 'amount'] = self.max_amount
            
            self.cleaned_stats['invalid_amounts_removed'] += invalid_amount_mask.sum()
        
        # Create amount categories for analysis
        df['amount_category'] = pd.cut(
            df['amount'],
            bins=[0, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, float('inf')],
            labels=['micro', 'very_small', 'small', 'medium_small', 'medium',
                   'medium_large', 'large', 'very_large', 'huge', 'extreme']
        )
        
        # Log transformation for skewed distribution
        df['amount_log'] = np.log1p(df['amount'])
        
        logger.info(f"Amount cleaning complete. Processed {len(df)} records")
        return df
    
    def _clean_currency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize currency codes.
        
        Operations:
        1. Standardize to uppercase
        2. Validate against allowed currencies
        3. Handle missing currencies (impute with most common)
        4. Create currency risk flags
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with cleaned currency column
        """
        if 'currency' not in df.columns:
            logger.warning("No currency column found. Assuming USD.")
            df['currency'] = 'USD'
            return df
        
        # Convert to uppercase and strip
        df['currency'] = df['currency'].astype(str).str.upper().str.strip()
        
        # Handle missing/invalid currencies
        invalid_currency_mask = ~df['currency'].isin(self.valid_currencies)
        
        if invalid_currency_mask.any():
            logger.warning(f"Found {invalid_currency_mask.sum()} invalid currency codes")
            
            # Get most common valid currency for imputation
            most_common_currency = df.loc[~invalid_currency_mask, 'currency'].mode()
            if len(most_common_currency) > 0:
                default_currency = most_common_currency[0]
            else:
                default_currency = 'USD'  # Fallback
            
            # Replace invalid currencies with default
            df.loc[invalid_currency_mask, 'currency'] = default_currency
            df['currency_was_invalid'] = invalid_currency_mask.astype(int)
            
            self.cleaned_stats['invalid_currencies_removed'] += invalid_currency_mask.sum()
        
        # Create currency risk score (based on historical fraud rates per currency)
        # This would typically come from a configuration or database
        currency_risk = {
            'USD': 1.0, 'EUR': 1.0, 'GBP': 1.0,  # Major currencies - baseline risk
            'JPY': 0.8, 'CAD': 0.8, 'AUD': 0.8,   # Lower risk
            'CHF': 0.7, 'SGD': 0.7,                # Even lower
            'CNY': 1.2, 'HKD': 1.1,                # Slightly higher risk
        }
        df['currency_risk_score'] = df['currency'].map(currency_risk).fillna(1.5)  # Unknown currencies get higher risk
        
        return df
    
    def _clean_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize transaction timestamps.
        
        Critical for temporal fraud detection features.
        
        Operations:
        1. Convert to datetime
        2. Handle timezone information
        3. Remove future/past transactions beyond bounds
        4. Extract temporal components
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with cleaned timestamp and derived features
        """
        if 'transaction_time' not in df.columns:
            logger.error("No transaction_time column found")
            raise ValueError("Transaction data must contain 'transaction_time' column")
        
        # Convert to datetime, handling various formats
        try:
            df['transaction_time'] = pd.to_datetime(df['transaction_time'])
        except Exception as e:
            logger.error(f"Error converting transaction_time to datetime: {e}")
            # Try with explicit format detection
            df['transaction_time'] = pd.to_datetime(df['transaction_time'], infer_datetime_format=True)
        
        # Remove timezone information (convert to UTC)
        if df['transaction_time'].dt.tz is not None:
            df['transaction_time'] = df['transaction_time'].dt.tz_convert(None)
        
        # Get current time for validation
        now = datetime.now()
        
        # Check for future dates (possibly due to timezone issues)
        future_mask = df['transaction_time'] > now + timedelta(days=self.config['max_future_days'])
        if future_mask.any():
            logger.warning(f"Found {future_mask.sum()} future transactions")
            # Adjust future dates to past (common fix for timezone issues)
            df.loc[future_mask, 'transaction_time'] = df.loc[future_mask, 'transaction_time'] - timedelta(days=1)
            self.cleaned_stats['invalid_dates_removed'] += future_mask.sum()
        
        # Check for very old transactions
        past_cutoff = now - timedelta(days=self.config['max_past_years'] * 365)
        old_mask = df['transaction_time'] < past_cutoff
        if old_mask.any():
            logger.warning(f"Found {old_mask.sum()} transactions older than {self.config['max_past_years']} years")
            # Flag old transactions but keep them (might be needed for historical analysis)
            df['is_historical'] = old_mask.astype(int)
        
        # Extract temporal features
        df['hour_of_day'] = df['transaction_time'].dt.hour
        df['day_of_week'] = df['transaction_time'].dt.dayofweek
        df['day_of_month'] = df['transaction_time'].dt.day
        df['month'] = df['transaction_time'].dt.month
        df['quarter'] = df['transaction_time'].dt.quarter
        df['year'] = df['transaction_time'].dt.year
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Create cyclical features for hour (for ML models)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        
        # Day of week cyclical features
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        logger.info(f"Timestamp cleaning complete. Date range: {df['transaction_time'].min()} to {df['transaction_time'].max()}")
        return df
    
    def _clean_merchant_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate merchant information.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with cleaned merchant information
        """
        # Clean merchant ID if present
        if 'merchant_id' in df.columns:
            df['merchant_id'] = df['merchant_id'].astype(str).str.strip()
            # Flag missing merchants
            missing_merchant = (df['merchant_id'] == '') | (df['merchant_id'] == 'nan') | (df['merchant_id'].isna())
            if missing_merchant.any():
                logger.warning(f"Found {missing_merchant.sum()} missing merchant IDs")
                df.loc[missing_merchant, 'merchant_id'] = 'UNKNOWN_MERCHANT'
                df['merchant_missing'] = missing_merchant.astype(int)
        
        # Clean merchant category
        if 'merchant_category' in df.columns:
            df['merchant_category'] = df['merchant_category'].astype(str).str.upper().str.strip()
            # Standardize common categories
            category_mapping = {
                'RESTAURANT': 'FOOD_AND_BEVERAGE',
                'RESTAURANTS': 'FOOD_AND_BEVERAGE',
                'CAFE': 'FOOD_AND_BEVERAGE',
                'COFFEE': 'FOOD_AND_BEVERAGE',
                'BAR': 'FOOD_AND_BEVERAGE',
                'GROCERY': 'RETAIL_GROCERY',
                'SUPERMARKET': 'RETAIL_GROCERY',
                'DEPARTMENT STORE': 'RETAIL_GENERAL',
                'CLOTHING': 'RETAIL_APPAREL',
                'ELECTRONICS': 'RETAIL_ELECTRONICS',
                'GAS': 'TRANSPORTATION_FUEL',
                'GAS STATION': 'TRANSPORTATION_FUEL',
                'HOTEL': 'TRAVEL_LODGING',
                'AIRLINE': 'TRAVEL_TRANSPORTATION',
                'ATM': 'FINANCIAL_SERVICES',
                'BANK': 'FINANCIAL_SERVICES'
            }
            df['merchant_category_std'] = df['merchant_category'].map(category_mapping).fillna('OTHER')
        
        return df
    
    def _clean_location_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate location information (country, city, etc.).
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with cleaned location information
        """
        # Clean country code
        if 'country' in df.columns:
            df['country'] = df['country'].astype(str).str.upper().str.strip()
            # Keep only first 2 characters for country code
            df['country'] = df['country'].str[:2]
            # Flag invalid countries (would need a valid country list)
            # This is a simplified version
        
        # Clean city
        if 'city' in df.columns:
            df['city'] = df['city'].astype(str).str.strip()
            # Remove special characters
            df['city'] = df['city'].str.replace(r'[^a-zA-Z\s]', '', regex=True)
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate transactions.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with duplicates removed
        """
        initial_count = len(df)
        
        # Define duplicate criteria
        subset_cols = ['transaction_id'] if 'transaction_id' in df.columns else None
        
        if subset_cols:
            df = df.drop_duplicates(subset=subset_cols, keep='first')
        else:
            # If no transaction_id, use combination of columns
            dup_cols = ['amount', 'transaction_time', 'customer_id'] if all(c in df.columns for c in ['amount', 'transaction_time', 'customer_id']) else None
            if dup_cols:
                df = df.drop_duplicates(subset=dup_cols, keep='first')
        
        duplicates_removed = initial_count - len(df)
        self.cleaned_stats['duplicates_removed'] = duplicates_removed
        
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate transactions")
        
        return df
    
    def _remove_invalid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with critical invalid data.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with invalid rows removed
        """
        initial_count = len(df)
        
        # Define critical columns that must be valid
        critical_cols = ['amount', 'transaction_time']
        for col in critical_cols:
            if col in df.columns:
                df = df[df[col].notna()]
        
        rows_removed = initial_count - len(df)
        if rows_removed > 0:
            logger.info(f"Removed {rows_removed} rows with critical missing data")
        
        return df
    
    def get_cleaning_stats(self) -> Dict:
        """
        Return statistics about cleaning operations performed.
        
        Returns:
            Dictionary with cleaning statistics
        """
        return self.cleaned_stats
    
    def save_cleaned_data(self, df: pd.DataFrame, output_path: str):
        """
        Save cleaned transaction data to file.
        
        Args:
            df: Cleaned DataFrame
            output_path: Path to save the cleaned data
        """
        df.to_parquet(output_path, index=False)
        logger.info(f"Cleaned transaction data saved to {output_path}")