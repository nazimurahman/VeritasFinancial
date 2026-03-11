"""
Customer Data Cleaner Module

This module handles cleaning and validation of customer profile data.
Customer information is crucial for establishing behavioral baselines
and detecting anomalies in transaction patterns.

Key functionalities:
1. Customer ID validation
2. Personal information anonymization (PII handling)
3. Age and demographic validation
4. Account tenure calculation
5. Credit score normalization
6. Income verification and categorization
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple, Union
import re
import hashlib

logger = logging.getLogger(__name__)


class CustomerCleaner:
    """
    Comprehensive customer data cleaner for banking fraud detection.
    
    This class implements privacy-preserving cleaning operations for
    customer profile data, ensuring data quality while protecting
    personally identifiable information (PII).
    
    Attributes:
        config (dict): Configuration parameters for cleaning operations
        cleaned_stats (dict): Statistics about cleaning operations performed
        pii_columns (list): Columns containing PII that need anonymization
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the CustomerCleaner with configuration.
        
        Args:
            config: Dictionary containing cleaning parameters
                   If None, default configuration will be used
        """
        self.config = config or {
            'min_age': 18,  # Minimum customer age
            'max_age': 120,  # Maximum reasonable age
            'min_credit_score': 300,  # Minimum FICO score
            'max_credit_score': 850,  # Maximum FICO score
            'min_income': 0,  # Minimum annual income
            'max_income': 10_000_000,  # Maximum reasonable income
            'anonymize_pii': True,  # Whether to anonymize PII
            'validate_email': True,
            'validate_phone': True,
            'validate_address': True
        }
        
        # Initialize statistics tracking
        self.cleaned_stats = {
            'total_customers_processed': 0,
            'invalid_ages_corrected': 0,
            'invalid_credit_scores_corrected': 0,
            'invalid_incomes_corrected': 0,
            'pii_anonymized': 0,
            'null_values_filled': 0
        }
        
        # Columns that contain PII and need anonymization
        self.pii_columns = [
            'name', 'email', 'phone', 'address', 'ssn', 
            'passport_number', 'drivers_license', 'dob'
        ]
        
        logger.info("CustomerCleaner initialized with config: %s", self.config)
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main cleaning method that orchestrates all customer cleaning operations.
        
        Args:
            df: Raw customer DataFrame
            
        Returns:
            Cleaned customer DataFrame
        """
        logger.info(f"Starting customer data cleaning on {len(df)} records")
        
        # Make a copy to avoid modifying original data
        cleaned_df = df.copy()
        
        # Track initial row count
        initial_count = len(cleaned_df)
        
        # Apply cleaning steps in sequence
        cleaned_df = self._validate_customer_id(cleaned_df)
        cleaned_df = self._clean_demographics(cleaned_df)
        cleaned_df = self._clean_account_info(cleaned_df)
        cleaned_df = self._clean_financial_info(cleaned_df)
        
        if self.config['anonymize_pii']:
            cleaned_df = self._anonymize_pii(cleaned_df)
        
        if self.config['validate_email']:
            cleaned_df = self._validate_email(cleaned_df)
        
        if self.config['validate_phone']:
            cleaned_df = self._validate_phone(cleaned_df)
        
        # Remove rows with critical missing data
        cleaned_df = self._remove_invalid_rows(cleaned_df)
        
        # Update statistics
        self.cleaned_stats['total_customers_processed'] = initial_count
        final_count = len(cleaned_df)
        logger.info(f"Customer cleaning complete. Kept {final_count}/{initial_count} records")
        
        return cleaned_df
    
    def _validate_customer_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean customer IDs.
        
        Args:
            df: DataFrame with customer data
            
        Returns:
            DataFrame with validated customer IDs
        """
        if 'customer_id' not in df.columns:
            logger.error("No customer_id column found")
            raise ValueError("Customer data must contain 'customer_id' column")
        
        # Convert to string and strip
        df['customer_id'] = df['customer_id'].astype(str).str.strip()
        
        # Check for duplicates
        duplicates = df['customer_id'].duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate customer IDs")
            # Keep first occurrence, mark others for review
            df['is_duplicate_customer'] = df['customer_id'].duplicated(keep='first').astype(int)
        
        # Handle empty IDs
        empty_mask = (df['customer_id'] == '') | (df['customer_id'] == 'nan') | (df['customer_id'].isna())
        if empty_mask.any():
            logger.warning(f"Found {empty_mask.sum()} empty customer IDs")
            # Generate temporary IDs
            df.loc[empty_mask, 'customer_id'] = [f"TEMP_CUST_{i:010d}" for i in range(empty_mask.sum())]
            self.cleaned_stats['null_values_filled'] += empty_mask.sum()
        
        return df
    
    def _clean_demographics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate demographic information.
        
        Args:
            df: DataFrame with customer data
            
        Returns:
            DataFrame with cleaned demographics
        """
        # Clean age
        if 'age' in df.columns:
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
            
            # Handle invalid ages
            invalid_age_mask = (df['age'] < self.config['min_age']) | (df['age'] > self.config['max_age'])
            if invalid_age_mask.any():
                logger.warning(f"Found {invalid_age_mask.sum()} invalid ages")
                # Set invalid ages to median age
                median_age = df.loc[~invalid_age_mask, 'age'].median()
                df.loc[invalid_age_mask, 'age'] = median_age
                df['age_was_invalid'] = invalid_age_mask.astype(int)
                self.cleaned_stats['invalid_ages_corrected'] += invalid_age_mask.sum()
            
            # Create age groups
            df['age_group'] = pd.cut(
                df['age'],
                bins=[0, 25, 35, 50, 65, 100],
                labels=['young_adult', 'adult', 'middle_age', 'senior', 'elderly']
            )
        
        # Clean gender
        if 'gender' in df.columns:
            df['gender'] = df['gender'].astype(str).str.upper().str.strip()
            # Standardize gender codes
            gender_mapping = {
                'M': 'MALE', 'MALE': 'MALE',
                'F': 'FEMALE', 'FEMALE': 'FEMALE',
                'O': 'OTHER', 'OTHER': 'OTHER', 'NON-BINARY': 'OTHER'
            }
            df['gender_std'] = df['gender'].map(gender_mapping).fillna('UNKNOWN')
        
        # Clean date of birth (if present)
        if 'date_of_birth' in df.columns:
            df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
            # Calculate age from DOB if age not provided
            if 'age' not in df.columns:
                today = datetime.now()
                df['age'] = (today - df['date_of_birth']).dt.days // 365
            # Flag suspicious DOB (future dates, too old)
            future_dob = df['date_of_birth'] > datetime.now()
            if future_dob.any():
                logger.warning(f"Found {future_dob.sum()} future dates of birth")
                df.loc[future_dob, 'date_of_birth'] = pd.NaT
        
        return df
    
    def _clean_account_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate account information.
        
        Args:
            df: DataFrame with customer data
            
        Returns:
            DataFrame with cleaned account information
        """
        # Clean account opening date
        if 'account_open_date' in df.columns:
            df['account_open_date'] = pd.to_datetime(df['account_open_date'], errors='coerce')
            
            # Calculate account tenure in days
            today = datetime.now()
            df['account_tenure_days'] = (today - df['account_open_date']).dt.days
            df['account_tenure_days'] = df['account_tenure_days'].clip(lower=0)  # No negative tenure
            
            # Create tenure categories
            df['account_tenure_category'] = pd.cut(
                df['account_tenure_days'],
                bins=[0, 30, 90, 365, 730, 1825, float('inf')],
                labels=['new', 'recent', 'established', 'loyal', 'very_loyal', 'long_term']
            )
        
        # Clean account type
        if 'account_type' in df.columns:
            df['account_type'] = df['account_type'].astype(str).str.upper().str.strip()
            # Standardize account types
            type_mapping = {
                'CHECKING': 'CHECKING', 'CHEQUING': 'CHECKING',
                'SAVINGS': 'SAVINGS', 'SAVING': 'SAVINGS',
                'CREDIT': 'CREDIT', 'CREDIT CARD': 'CREDIT',
                'LOAN': 'LOAN', 'MORTGAGE': 'MORTGAGE',
                'INVESTMENT': 'INVESTMENT', 'BROKERAGE': 'INVESTMENT'
            }
            df['account_type_std'] = df['account_type'].map(type_mapping).fillna('OTHER')
        
        return df
    
    def _clean_financial_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate financial information.
        
        Args:
            df: DataFrame with customer data
            
        Returns:
            DataFrame with cleaned financial information
        """
        # Clean credit score
        if 'credit_score' in df.columns:
            df['credit_score'] = pd.to_numeric(df['credit_score'], errors='coerce')
            
            # Handle invalid credit scores
            invalid_score_mask = (df['credit_score'] < self.config['min_credit_score']) | \
                                 (df['credit_score'] > self.config['max_credit_score'])
            if invalid_score_mask.any():
                logger.warning(f"Found {invalid_score_mask.sum()} invalid credit scores")
                # Set to median score
                median_score = df.loc[~invalid_score_mask, 'credit_score'].median()
                df.loc[invalid_score_mask, 'credit_score'] = median_score
                df['credit_score_was_invalid'] = invalid_score_mask.astype(int)
                self.cleaned_stats['invalid_credit_scores_corrected'] += invalid_score_mask.sum()
            
            # Create credit score categories
            df['credit_score_category'] = pd.cut(
                df['credit_score'],
                bins=[0, 580, 670, 740, 800, 850],
                labels=['poor', 'fair', 'good', 'very_good', 'excellent']
            )
        
        # Clean annual income
        if 'annual_income' in df.columns:
            df['annual_income'] = pd.to_numeric(df['annual_income'], errors='coerce')
            
            # Handle invalid incomes
            invalid_income_mask = (df['annual_income'] < self.config['min_income']) | \
                                  (df['annual_income'] > self.config['max_income'])
            if invalid_income_mask.any():
                logger.warning(f"Found {invalid_income_mask.sum()} invalid incomes")
                # Set to median income
                median_income = df.loc[~invalid_income_mask, 'annual_income'].median()
                df.loc[invalid_income_mask, 'annual_income'] = median_income
                df['income_was_invalid'] = invalid_income_mask.astype(int)
                self.cleaned_stats['invalid_incomes_corrected'] += invalid_income_mask.sum()
            
            # Log transform for modeling
            df['annual_income_log'] = np.log1p(df['annual_income'])
            
            # Create income brackets
            df['income_bracket'] = pd.cut(
                df['annual_income'],
                bins=[0, 30000, 50000, 75000, 100000, 150000, 250000, float('inf')],
                labels=['low', 'lower_middle', 'middle', 'upper_middle', 'high', 'very_high', 'ultra_high']
            )
        
        return df
    
    def _anonymize_pii(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Anonymize personally identifiable information (PII).
        
        This is crucial for data privacy compliance (GDPR, CCPA, etc.)
        
        Args:
            df: DataFrame with customer data
            
        Returns:
            DataFrame with anonymized PII
        """
        # Identify which PII columns exist
        existing_pii = [col for col in self.pii_columns if col in df.columns]
        
        if not existing_pii:
            return df
        
        logger.info(f"Anonymizing PII in columns: {existing_pii}")
        
        for col in existing_pii:
            # Create anonymized version using hashing
            # One-way hash ensures PII cannot be reversed
            df[f'{col}_hash'] = df[col].astype(str).apply(
                lambda x: hashlib.sha256(x.encode()).hexdigest()[:16] if pd.notna(x) else None
            )
            
            # Optionally remove original PII column
            # df = df.drop(columns=[col])
            
            self.cleaned_stats['pii_anonymized'] += 1
        
        return df
    
    def _validate_email(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate email addresses format.
        
        Args:
            df: DataFrame with customer data
            
        Returns:
            DataFrame with validated emails
        """
        if 'email' not in df.columns:
            return df
        
        # Simple email validation regex
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        valid_email_mask = df['email'].astype(str).str.match(email_pattern, na=False)
        df['email_valid'] = valid_email_mask.astype(int)
        
        invalid_emails = (~valid_email_mask).sum()
        if invalid_emails > 0:
            logger.warning(f"Found {invalid_emails} invalid email addresses")
        
        return df
    
    def _validate_phone(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate phone numbers format.
        
        Args:
            df: DataFrame with customer data
            
        Returns:
            DataFrame with validated phone numbers
        """
        if 'phone' not in df.columns:
            return df
        
        # Clean phone numbers (remove non-digits)
        df['phone_cleaned'] = df['phone'].astype(str).str.replace(r'\D', '', regex=True)
        
        # Simple validation: between 10-15 digits (international)
        df['phone_valid'] = df['phone_cleaned'].str.len().between(10, 15).astype(int)
        
        invalid_phones = (df['phone_valid'] == 0).sum()
        if invalid_phones > 0:
            logger.warning(f"Found {invalid_phones} invalid phone numbers")
        
        return df
    
    def _remove_invalid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with critical invalid data.
        
        Args:
            df: DataFrame with customer data
            
        Returns:
            DataFrame with invalid rows removed
        """
        initial_count = len(df)
        
        # Define critical columns that must be valid
        critical_cols = ['customer_id']
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