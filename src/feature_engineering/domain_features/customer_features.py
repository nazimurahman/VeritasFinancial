"""
Customer Features Module
=======================
This module creates features related to customer profiles and behavior.
Customer-level features capture long-term patterns and risk characteristics
that help distinguish between legitimate and fraudulent transactions.

Key Concepts:
- Demographics: Age, income, location, etc.
- Account history: Account age, balance patterns, etc.
- Credit profile: Credit score, risk ratings, etc.
- Behavioral patterns: Spending habits, transaction frequencies, etc.

Author: VeritasFinancial Data Science Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class CustomerFeatureEngineer:
    """
    Master class for creating all customer-level features.
    
    This class orchestrates the creation of features related to customer
    profiles, history, and behavior patterns.
    
    Attributes:
        config (dict): Configuration parameters
        demographics_extractor (DemographicsFeatureExtractor)
        account_extractor (AccountFeatureExtractor)
        risk_extractor (RiskProfileFeatureExtractor)
        fitted (bool): Whether the engineer has been fitted
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the CustomerFeatureEngineer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.fitted = False
        
        # Initialize specialized extractors
        self.demographics_extractor = DemographicsFeatureExtractor(self.config)
        self.account_extractor = AccountFeatureExtractor(self.config)
        self.risk_extractor = RiskProfileFeatureExtractor(self.config)
        
        self.feature_names = []
        
    def fit(self, df: pd.DataFrame) -> 'CustomerFeatureEngineer':
        """
        Fit the customer feature engineer to the data.
        
        Args:
            df: DataFrame containing customer data
            
        Returns:
            self: The fitted instance
        """
        # Fit each extractor
        self.demographics_extractor.fit(df)
        self.account_extractor.fit(df)
        self.risk_extractor.fit(df)
        
        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by creating all customer features.
        
        Args:
            df: DataFrame containing customer data
            
        Returns:
            DataFrame with customer features added
        """
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")
        
        result_df = df.copy()
        
        # Generate features from each extractor
        result_df = self.demographics_extractor.transform(result_df)
        result_df = self.account_extractor.transform(result_df)
        result_df = self.risk_extractor.transform(result_df)
        
        # Update feature names
        self.feature_names = [
            col for col in result_df.columns 
            if col not in df.columns or col.startswith(('customer_', 'age_', 'income_'))
        ]
        
        return result_df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: DataFrame containing customer data
            
        Returns:
            DataFrame with customer features added
        """
        return self.fit(df).transform(df)


class DemographicsFeatureExtractor:
    """
    Extract demographic features about customers.
    
    Demographics provide context about who the customer is and what
    their typical financial behavior might look like.
    
    Features created:
    - age: Customer age
    - age_group: Categorical age group
    - income_level: Income bracket
    - income_to_age_ratio: Income normalized by age
    - location_density: Urban/suburban/rural indicator
    - employment_status: Employment category
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the DemographicsFeatureExtractor.
        
        Args:
            config: Configuration containing:
                - age_bins: Age categories boundaries
                - income_bins: Income categories boundaries
                - location_density_map: Mapping of locations to density
        """
        self.config = config or {}
        self.fitted = False
        self.age_bins = self.config.get('age_bins', [0, 25, 35, 50, 65, 100])
        self.age_labels = self.config.get('age_labels', 
                                          ['young', 'young_adult', 'adult', 'middle_age', 'senior'])
        self.income_bins = self.config.get('income_bins', 
                                           [0, 30000, 50000, 75000, 100000, 200000, float('inf')])
        self.income_labels = self.config.get('income_labels',
                                             ['low', 'lower_middle', 'middle', 'upper_middle', 'high', 'very_high'])
        
    def fit(self, df: pd.DataFrame) -> 'DemographicsFeatureExtractor':
        """
        Fit the demographics extractor.
        
        Args:
            df: DataFrame with demographic columns
        """
        # Check required columns
        self.has_age = 'customer_age' in df.columns
        self.has_income = 'income' in df.columns or 'annual_income' in df.columns
        self.has_location = 'city' in df.columns or 'region' in df.columns
        
        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create demographic features.
        
        Args:
            df: DataFrame with demographic data
            
        Returns:
            DataFrame with demographic features added
        """
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")
            
        result_df = df.copy()
        
        # Age-based features
        if self.has_age:
            result_df = self._create_age_features(result_df)
        
        # Income-based features
        if self.has_income:
            result_df = self._create_income_features(result_df)
        
        # Location-based demographic features
        if self.has_location:
            result_df = self._create_location_demographics(result_df)
        
        return result_df
    
    def _create_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create age-related features.
        
        Args:
            df: DataFrame with age column
            
        Returns:
            DataFrame with age features added
        """
        # Get age column (might be named differently)
        age_col = 'customer_age' if 'customer_age' in df.columns else 'age'
        
        # Basic age
        df['customer_age'] = df[age_col]
        
        # Age group categorization
        df['age_group'] = pd.cut(
            df['customer_age'],
            bins=self.age_bins,
            labels=self.age_labels,
            right=False
        )
        
        # Age squared (for non-linear effects)
        df['age_squared'] = df['customer_age'] ** 2
        
        # Age normalized (z-score)
        if 'age_mean' not in self.__dict__:
            self.age_mean = df['customer_age'].mean()
            self.age_std = df['customer_age'].std()
        
        df['age_normalized'] = (df['customer_age'] - self.age_mean) / self.age_std
        
        # Is senior (high risk for certain fraud types)
        df['is_senior'] = (df['customer_age'] >= 65).astype(int)
        
        # Is young adult (high risk for other fraud types)
        df['is_young_adult'] = ((df['customer_age'] >= 18) & (df['customer_age'] <= 25)).astype(int)
        
        return df
    
    def _create_income_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create income-related features.
        
        Args:
            df: DataFrame with income column
            
        Returns:
            DataFrame with income features added
        """
        # Get income column
        income_col = 'income' if 'income' in df.columns else 'annual_income'
        
        # Basic income
        df['annual_income'] = df[income_col]
        
        # Income group
        df['income_group'] = pd.cut(
            df['annual_income'],
            bins=self.income_bins,
            labels=self.income_labels,
            right=False
        )
        
        # Log income (handles skew)
        df['income_log'] = np.log1p(df['annual_income'])
        
        # Income percentile
        if not hasattr(self, 'income_percentiles'):
            self.income_percentiles = df['annual_income'].rank(pct=True)
        df['income_percentile'] = self.income_percentiles
        
        # High income flag
        df['is_high_income'] = (df['income_percentile'] > 0.8).astype(int)
        
        # Low income flag
        df['is_low_income'] = (df['income_percentile'] < 0.2).astype(int)
        
        # Income to age ratio (affluence measure)
        if self.has_age:
            df['income_age_ratio'] = df['annual_income'] / (df['customer_age'] + 1)
        
        return df
    
    def _create_location_demographics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create location-based demographic features.
        
        Args:
            df: DataFrame with location information
            
        Returns:
            DataFrame with location demographics added
        """
        # Determine which location column we have
        if 'city' in df.columns:
            location_col = 'city'
        elif 'region' in df.columns:
            location_col = 'region'
        else:
            return df
        
        # Location density (urban/suburban/rural)
        # This would typically come from an external mapping
        location_density_map = self.config.get('location_density_map', {})
        df['location_density'] = df[location_col].map(location_density_map).fillna('suburban')
        
        # One-hot encode location density
        density_dummies = pd.get_dummies(
            df['location_density'], 
            prefix='location_density',
            dummy_na=False
        )
        df = pd.concat([df, density_dummies], axis=1)
        
        # Regional economic indicators (if available)
        if 'region_gdp' in self.config:
            df['region_economic_power'] = df[location_col].map(
                self.config['region_gdp']
            ).fillna(df['annual_income'].median())
        
        return df


class AccountFeatureExtractor:
    """
    Extract features related to customer accounts.
    
    Account features capture the history and status of the customer's
    relationship with the bank, which can indicate risk levels.
    
    Features created:
    - account_age_days: How long the account has been open
    - account_tenure_category: Categorized account age
    - total_balance: Current account balance
    - avg_balance: Average balance over time
    - balance_volatility: How much balance fluctuates
    - account_activity_score: Measure of account usage
    - num_accounts: Number of accounts held
    - account_types: Types of accounts (checking, savings, etc.)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the AccountFeatureExtractor.
        
        Args:
            config: Configuration containing:
                - tenure_bins: Account age categories
                - balance_thresholds: Balance categories
                - activity_window: Window for activity calculation
        """
        self.config = config or {}
        self.fitted = False
        self.tenure_bins = self.config.get('tenure_bins', [0, 30, 90, 365, 1095, 3650])
        self.tenure_labels = self.config.get('tenure_labels',
                                             ['new', 'recent', 'established', 'loyal', 'long_term'])
        self.activity_window = self.config.get('activity_window', 30)  # days
        
    def fit(self, df: pd.DataFrame) -> 'AccountFeatureExtractor':
        """
        Fit the account extractor.
        
        Args:
            df: DataFrame with account data
        """
        # Check required columns
        self.has_balance = 'balance' in df.columns or 'current_balance' in df.columns
        self.has_account_age = 'account_age_days' in df.columns or 'account_open_date' in df.columns
        self.has_account_type = 'account_type' in df.columns
        
        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create account-based features.
        
        Args:
            df: DataFrame with account data
            
        Returns:
            DataFrame with account features added
        """
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")
            
        result_df = df.copy()
        
        # Account age features
        if self.has_account_age:
            result_df = self._create_account_age_features(result_df)
        
        # Balance features
        if self.has_balance:
            result_df = self._create_balance_features(result_df)
        
        # Account type features
        if self.has_account_type:
            result_df = self._create_account_type_features(result_df)
        
        # Activity features (requires transaction history)
        result_df = self._create_activity_features(result_df)
        
        return result_df
    
    def _create_account_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on how long the account has been open.
        
        Args:
            df: DataFrame with account age information
            
        Returns:
            DataFrame with account age features added
        """
        # Get account age in days
        if 'account_age_days' in df.columns:
            df['account_age_days'] = df['account_age_days']
        elif 'account_open_date' in df.columns:
            # Calculate from open date
            current_date = datetime.now()
            df['account_age_days'] = (
                current_date - pd.to_datetime(df['account_open_date'])
            ).dt.days
        
        # Account age in years (for interpretability)
        df['account_age_years'] = df['account_age_days'] / 365.25
        
        # Account tenure category
        df['account_tenure'] = pd.cut(
            df['account_age_days'],
            bins=self.tenure_bins,
            labels=self.tenure_labels,
            right=False
        )
        
        # New account flag (high risk)
        df['is_new_account'] = (df['account_age_days'] < 30).astype(int)
        
        # Very old account flag (may indicate different risk)
        df['is_very_old_account'] = (df['account_age_days'] > 3650).astype(int)  # 10+ years
        
        # Log account age (handles skew)
        df['account_age_log'] = np.log1p(df['account_age_days'])
        
        return df
    
    def _create_balance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on account balances.
        
        Args:
            df: DataFrame with balance information
            
        Returns:
            DataFrame with balance features added
        """
        # Get balance column
        balance_col = 'balance' if 'balance' in df.columns else 'current_balance'
        df['current_balance'] = df[balance_col]
        
        # Log balance (handles skew and large values)
        df['balance_log'] = np.log1p(np.abs(df['current_balance']) + 1)
        
        # Balance sign (positive/negative/zero)
        df['balance_sign'] = np.sign(df['current_balance']).astype(int)
        
        # Balance categories
        balance_thresholds = self.config.get('balance_thresholds', 
                                            [0, 1000, 5000, 10000, 50000, 100000, float('inf')])
        balance_labels = self.config.get('balance_labels',
                                        ['zero', 'low', 'medium_low', 'medium', 'high', 'very_high'])
        
        df['balance_category'] = pd.cut(
            np.abs(df['current_balance']),
            bins=balance_thresholds,
            labels=balance_labels,
            right=False
        )
        
        # Flag for negative balance (overdrawn)
        df['is_overdrawn'] = (df['current_balance'] < 0).astype(int)
        
        # Flag for high balance (might indicate high-value target)
        df['is_high_balance'] = (df['current_balance'] > 50000).astype(int)
        
        # Balance to income ratio (if income available)
        if 'annual_income' in df.columns:
            df['balance_to_income_ratio'] = df['current_balance'] / (df['annual_income'] + 1)
            
        return df
    
    def _create_account_type_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on account types.
        
        Args:
            df: DataFrame with account type information
            
        Returns:
            DataFrame with account type features added
        """
        # One-hot encode account type
        if 'account_type' in df.columns:
            account_type_dummies = pd.get_dummies(
                df['account_type'], 
                prefix='account_type',
                dummy_na=False
            )
            df = pd.concat([df, account_type_dummies], axis=1)
            
            # Count number of account types (if customer has multiple)
            if 'account_types_list' in df.columns:
                df['num_account_types'] = df['account_types_list'].apply(len)
            else:
                df['num_account_types'] = 1
        
        return df
    
    def _create_activity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on account activity.
        
        Requires transaction history to calculate activity metrics.
        
        Args:
            df: DataFrame with transaction history
            
        Returns:
            DataFrame with activity features added
        """
        # These features require transaction data, so they might be
        # calculated elsewhere and joined in
        
        # If we have transaction count for the account
        if 'transaction_count_30d' in df.columns:
            df['tx_count_30d'] = df['transaction_count_30d']
            df['is_active'] = (df['tx_count_30d'] > 5).astype(int)
            df['is_inactive'] = (df['tx_count_30d'] == 0).astype(int)
        
        # If we have average transaction amount
        if 'avg_tx_amount_30d' in df.columns:
            df['avg_tx_amount_30d'] = df['avg_tx_amount_30d']
        
        # Activity score (composite measure)
        if all(col in df.columns for col in ['tx_count_30d', 'avg_tx_amount_30d']):
            # Normalize components
            max_count = df['tx_count_30d'].max()
            max_amount = df['avg_tx_amount_30d'].max()
            
            df['activity_score'] = (
                0.5 * (df['tx_count_30d'] / max_count) +
                0.5 * (df['avg_tx_amount_30d'] / max_amount)
            )
        
        return df


class RiskProfileFeatureExtractor:
    """
    Extract features related to customer risk profiles.
    
    Risk profile features combine various indicators to assess the
    overall risk level of a customer.
    
    Features created:
    - credit_score: Customer's credit score
    - credit_score_category: Categorized credit score
    - risk_rating: Internal risk rating
    - past_fraud_count: Number of past fraud incidents
    - past_chargeback_count: Number of past chargebacks
    - risk_tier: Overall risk tier
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the RiskProfileFeatureExtractor.
        
        Args:
            config: Configuration containing:
                - credit_score_bins: Credit score categories
                - risk_tier_thresholds: Risk tier boundaries
        """
        self.config = config or {}
        self.fitted = False
        
        # Credit score categories (standard FICO ranges)
        self.credit_score_bins = self.config.get(
            'credit_score_bins', 
            [300, 580, 670, 740, 800, 850]
        )
        self.credit_score_labels = self.config.get(
            'credit_score_labels',
            ['poor', 'fair', 'good', 'very_good', 'excellent']
        )
        
    def fit(self, df: pd.DataFrame) -> 'RiskProfileFeatureExtractor':
        """
        Fit the risk profile extractor.
        
        Args:
            df: DataFrame with risk-related columns
        """
        self.has_credit_score = 'credit_score' in df.columns
        self.has_risk_rating = 'risk_rating' in df.columns
        self.has_fraud_history = 'past_fraud_count' in df.columns
        
        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create risk profile features.
        
        Args:
            df: DataFrame with risk-related data
            
        Returns:
            DataFrame with risk profile features added
        """
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")
            
        result_df = df.copy()
        
        # Credit score features
        if self.has_credit_score:
            result_df = self._create_credit_score_features(result_df)
        
        # Risk rating features
        if self.has_risk_rating:
            result_df = self._create_risk_rating_features(result_df)
        
        # Fraud history features
        if self.has_fraud_history:
            result_df = self._create_fraud_history_features(result_df)
        
        # Composite risk score
        result_df = self._create_composite_risk_score(result_df)
        
        return result_df
    
    def _create_credit_score_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from credit scores.
        
        Args:
            df: DataFrame with credit score column
            
        Returns:
            DataFrame with credit score features added
        """
        df['credit_score'] = df['credit_score']
        
        # Credit score category
        df['credit_score_category'] = pd.cut(
            df['credit_score'],
            bins=self.credit_score_bins,
            labels=self.credit_score_labels,
            right=False
        )
        
        # Normalized credit score (0-1)
        min_score = self.credit_score_bins[0]
        max_score = self.credit_score_bins[-1]
        df['credit_score_normalized'] = (
            (df['credit_score'] - min_score) / (max_score - min_score)
        )
        
        # Poor credit flag
        df['is_poor_credit'] = (df['credit_score'] < 580).astype(int)
        
        # Excellent credit flag
        df['is_excellent_credit'] = (df['credit_score'] >= 740).astype(int)
        
        return df
    
    def _create_risk_rating_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from internal risk ratings.
        
        Args:
            df: DataFrame with risk rating column
            
        Returns:
            DataFrame with risk rating features added
        """
        if 'risk_rating' in df.columns:
            # Convert rating to numeric if it's categorical
            if df['risk_rating'].dtype == 'object':
                # Map common risk ratings to numbers
                risk_map = {
                    'low': 1,
                    'medium': 2,
                    'high': 3,
                    'very_high': 4
                }
                df['risk_rating_numeric'] = df['risk_rating'].map(risk_map).fillna(2)
            else:
                df['risk_rating_numeric'] = df['risk_rating']
            
            # High risk flag
            df['is_high_risk_rating'] = (df['risk_rating_numeric'] >= 3).astype(int)
        
        return df
    
    def _create_fraud_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from fraud history.
        
        Args:
            df: DataFrame with fraud history columns
            
        Returns:
            DataFrame with fraud history features added
        """
        # Past fraud count
        if 'past_fraud_count' in df.columns:
            df['past_fraud_count'] = df['past_fraud_count'].fillna(0)
            
            # Binary flags
            df['has_past_fraud'] = (df['past_fraud_count'] > 0).astype(int)
            df['has_multiple_frauds'] = (df['past_fraud_count'] > 1).astype(int)
            
            # Log of fraud count (handles skew)
            df['past_fraud_count_log'] = np.log1p(df['past_fraud_count'])
        
        # Past chargeback count
        if 'past_chargeback_count' in df.columns:
            df['past_chargeback_count'] = df['past_chargeback_count'].fillna(0)
            df['has_past_chargeback'] = (df['past_chargeback_count'] > 0).astype(int)
        
        # Days since last fraud (if available)
        if 'days_since_last_fraud' in df.columns:
            df['days_since_last_fraud'] = df['days_since_last_fraud'].fillna(9999)
            df['recent_fraud'] = (df['days_since_last_fraud'] < 90).astype(int)
        
        return df
    
    def _create_composite_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a composite risk score combining multiple factors.
        
        Args:
            df: DataFrame with risk factors
            
        Returns:
            DataFrame with composite risk score added
        """
        # Initialize score
        risk_score = 0
        num_factors = 0
        
        # Credit score component (inverse relationship)
        if 'credit_score_normalized' in df.columns:
            risk_score += (1 - df['credit_score_normalized']) * 40  # 40% weight
            num_factors += 40
        
        # Risk rating component
        if 'risk_rating_numeric' in df.columns:
            # Normalize to 0-1
            max_rating = df['risk_rating_numeric'].max()
            if max_rating > 0:
                risk_score += (df['risk_rating_numeric'] / max_rating) * 30  # 30% weight
                num_factors += 30
        
        # Fraud history component
        if 'has_past_fraud' in df.columns:
            risk_score += df['has_past_fraud'] * 20  # 20% weight
            num_factors += 20
            
            if 'has_multiple_frauds' in df.columns:
                risk_score += df['has_multiple_frauds'] * 10  # Additional 10%
                num_factors += 10
        
        # Account age component (inverse relationship)
        if 'account_age_days' in df.columns:
            # Newer accounts are higher risk
            max_age = df['account_age_days'].max()
            if max_age > 0:
                age_risk = 1 - (df['account_age_days'] / max_age)
                risk_score += age_risk * 15
                num_factors += 15
        
        # Normalize to 0-100 scale
        if num_factors > 0:
            df['composite_risk_score'] = (risk_score / num_factors) * 100
        else:
            df['composite_risk_score'] = 50  # Default medium risk
        
        # Risk tiers
        risk_tiers = self.config.get('risk_tiers', [0, 30, 60, 80, 100])
        tier_labels = self.config.get('tier_labels', ['low', 'medium', 'high', 'very_high'])
        
        df['risk_tier'] = pd.cut(
            df['composite_risk_score'],
            bins=risk_tiers,
            labels=tier_labels,
            right=False
        )
        
        return df