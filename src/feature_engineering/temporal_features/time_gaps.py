"""
Time Gap Features for Banking Fraud Detection
=============================================
This module implements features based on time intervals between transactions.
Time gaps are powerful indicators of behavioral changes and can reveal
automated fraud attempts, account takeovers, and unusual patterns.

Key Concepts:
- Inter-transaction intervals
- Gap patterns and distributions
- Irregular timing detection
- Session-based gap analysis
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import timedelta


class TimeGapFeatureEngineer:
    """
    Create features based on time gaps between transactions.
    
    Banking Context:
    - Regular users have characteristic time gaps between transactions
    - Bots/automated fraud have very consistent, short gaps
    - Account takeover often shows unusual gap patterns
    - Card testing shows many transactions with minimal gaps
    """
    
    def __init__(self):
        self.feature_columns = []
        
    def create_intertransaction_gap_features(self,
                                           df: pd.DataFrame,
                                           customer_id_col: str = 'customer_id',
                                           time_col: str = 'transaction_time',
                                           transaction_id_col: str = 'transaction_id') -> pd.DataFrame:
        """
        Create features based on gaps between consecutive transactions.
        
        Parameters:
        -----------
        df : DataFrame with transaction data
        customer_id_col : Customer identifier
        time_col : Timestamp column
        transaction_id_col : Transaction identifier
        
        Returns:
        --------
        DataFrame with gap-based features
        """
        
        result_df = df.copy()
        
        # Ensure sorted for proper gap calculation
        result_df = result_df.sort_values([customer_id_col, time_col])
        
        # Calculate time since last transaction (in seconds)
        result_df['time_since_last_tx_seconds'] = (
            result_df
            .groupby(customer_id_col)[time_col]
            .diff()
            .dt.total_seconds()
        )
        
        # Time since last transaction in minutes (more interpretable)
        result_df['time_since_last_tx_minutes'] = (
            result_df['time_since_last_tx_seconds'] / 60
        )
        
        # Time since last transaction in hours
        result_df['time_since_last_tx_hours'] = (
            result_df['time_since_last_tx_minutes'] / 60
        )
        
        # Time since last transaction in days
        result_df['time_since_last_tx_days'] = (
        result_df['time_since_last_tx_hours'] / 24
        )
        
        # Fill NaN for first transactions (no previous transaction)
        result_df['time_since_last_tx_seconds'] = result_df['time_since_last_tx_seconds'].fillna(-1)
        result_df['time_since_last_tx_minutes'] = result_df['time_since_last_tx_minutes'].fillna(-1)
        result_df['time_since_last_tx_hours'] = result_df['time_since_last_tx_hours'].fillna(-1)
        result_df['time_since_last_tx_days'] = result_df['time_since_last_tx_days'].fillna(-1)
        
        # Time to next transaction (look forward)
        result_df['time_to_next_tx_seconds'] = (
            result_df
            .groupby(customer_id_col)[time_col]
            .diff(-1)
            .dt.total_seconds()
            .abs()
        )
        
        result_df['time_to_next_tx_minutes'] = (
            result_df['time_to_next_tx_seconds'] / 60
        )
        
        result_df['time_to_next_tx_hours'] = (
            result_df['time_to_next_tx_minutes'] / 60
        )
        
        # Fill NaN for last transactions
        result_df['time_to_next_tx_seconds'] = result_df['time_to_next_tx_seconds'].fillna(-1)
        result_df['time_to_next_tx_minutes'] = result_df['time_to_next_tx_minutes'].fillna(-1)
        result_df['time_to_next_tx_hours'] = result_df['time_to_next_tx_hours'].fillna(-1)
        
        self.feature_columns.extend([
            'time_since_last_tx_seconds',
            'time_since_last_tx_minutes',
            'time_since_last_tx_hours',
            'time_since_last_tx_days',
            'time_to_next_tx_seconds',
            'time_to_next_tx_minutes',
            'time_to_next_tx_hours'
        ])
        
        return result_df
    
    def create_gap_statistics_features(self,
                                      df: pd.DataFrame,
                                      customer_id_col: str = 'customer_id',
                                      time_since_col: str = 'time_since_last_tx_minutes') -> pd.DataFrame:
        """
        Create statistical features based on gap history.
        
        This captures how the current gap compares to the customer's
        historical gap patterns.
        """
        
        result_df = df.copy()
        
        # Rolling statistics of time gaps
        windows = [5, 10, 20]  # Number of transactions to look back
        
        for window in windows:
            # Average gap over last N transactions
            avg_gap_name = f'avg_gap_last_{window}_tx'
            result_df[avg_gap_name] = (
                result_df
                .groupby(customer_id_col)[time_since_col]
                .transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
            )
            
            # Standard deviation of gaps
            std_gap_name = f'std_gap_last_{window}_tx'
            result_df[std_gap_name] = (
                result_df
                .groupby(customer_id_col)[time_since_col]
                .transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
            )
            result_df[std_gap_name] = result_df[std_gap_name].fillna(0)
            
            # Coefficient of variation (std/mean) - captures regularity
            cv_gap_name = f'cv_gap_last_{window}_tx'
            result_df[cv_gap_name] = (
                result_df[std_gap_name] / (result_df[avg_gap_name] + 1e-8)
            )
            result_df[cv_gap_name] = result_df[cv_gap_name].clip(0, 10)
            
            # Min and max gaps
            min_gap_name = f'min_gap_last_{window}_tx'
            result_df[min_gap_name] = (
                result_df
                .groupby(customer_id_col)[time_since_col]
                .transform(
                    lambda x: x.rolling(window=window, min_periods=1).min()
                )
            )
            
            max_gap_name = f'max_gap_last_{window}_tx'
            result_df[max_gap_name] = (
                result_df
                .groupby(customer_id_col)[time_since_col]
                .transform(
                    lambda x: x.rolling(window=window, min_periods=1).max()
                )
            )
            
            # Gap range
            range_gap_name = f'range_gap_last_{window}_tx'
            result_df[range_gap_name] = (
                result_df[max_gap_name] - result_df[min_gap_name]
            )
            
            # Z-score of current gap
            zscore_gap_name = f'gap_zscore_last_{window}_tx'
            result_df[zscore_gap_name] = (
                (result_df[time_since_col] - result_df[avg_gap_name]) /
                (result_df[std_gap_name] + 1e-8)
            )
            result_df[zscore_gap_name] = result_df[zscore_gap_name].clip(-5, 5)
            
            self.feature_columns.extend([
                avg_gap_name, std_gap_name, cv_gap_name,
                min_gap_name, max_gap_name, range_gap_name,
                zscore_gap_name
            ])
        
        return result_df
    
    def create_irregular_timing_features(self,
                                        df: pd.DataFrame,
                                        customer_id_col: str = 'customer_id',
                                        time_since_col: str = 'time_since_last_tx_minutes') -> pd.DataFrame:
        """
        Detect irregular timing patterns that may indicate fraud.
        
        Irregular patterns include:
        - Extremely regular gaps (bot-like behavior)
        - Extremely irregular gaps (chaotic behavior)
        - Gaps that don't match typical human patterns
        """
        
        result_df = df.copy()
        
        # Flag for extremely short gaps (< 1 minute) - potential automated fraud
        result_df['is_very_short_gap'] = (
            (result_df[time_since_col] >= 0) & 
            (result_df[time_since_col] < 1)
        ).astype(int)
        
        # Flag for extremely long gaps (> 7 days) - account dormancy
        result_df['is_very_long_gap'] = (
            result_df[time_since_col] > 7 * 24 * 60  # 7 days in minutes
        ).astype(int)
        
        # Flag for gaps during sleeping hours (assuming 12 AM - 5 AM)
        if 'hour' in result_df.columns:
            result_df['is_gap_during_sleep'] = (
                (result_df['hour'] >= 0) & 
                (result_df['hour'] <= 5) & 
                (result_df[time_since_col] > 0)
            ).astype(int)
        else:
            result_df['is_gap_during_sleep'] = 0
        
        # Detect periodic patterns (multiple of certain intervals)
        # This catches potential scheduled fraudulent transactions
        
        # Check if gap is multiple of 1 hour (within tolerance)
        result_df['gap_multiple_of_1h'] = (
            (result_df[time_since_col] > 0) & 
            (np.abs(result_df[time_since_col] % 60) < 5)
        ).astype(int)
        
        # Check if gap is multiple of 24 hours
        result_df['gap_multiple_of_24h'] = (
            (result_df[time_since_col] > 0) & 
            (np.abs(result_df[time_since_col] % (24*60)) < 30)
        ).astype(int)
        
        # Check if gap is multiple of 7 days
        result_df['gap_multiple_of_7d'] = (
            (result_df[time_since_col] > 0) & 
            (np.abs(result_df[time_since_col] % (7*24*60)) < 60)
        ).astype(int)
        
        # Regularity score (0-1, higher means more regular)
        # Uses coefficient of variation (lower CV = more regular)
        if 'cv_gap_last_10_tx' in result_df.columns:
            # CV < 0.5 is very regular, CV > 2 is very irregular
            result_df['regularity_score'] = 1 / (1 + result_df['cv_gap_last_10_tx'])
            result_df['regularity_score'] = result_df['regularity_score'].clip(0, 1)
            
            # Flag for highly regular patterns (potential bots)
            result_df['is_highly_regular'] = (
                result_df['cv_gap_last_10_tx'] < 0.3
            ).astype(int)
            
            # Flag for highly irregular patterns (potential chaos)
            result_df['is_highly_irregular'] = (
                result_df['cv_gap_last_10_tx'] > 2
            ).astype(int)
        
        self.feature_columns.extend([
            'is_very_short_gap',
            'is_very_long_gap',
            'is_gap_during_sleep',
            'gap_multiple_of_1h',
            'gap_multiple_of_24h',
            'gap_multiple_of_7d'
        ])
        
        if 'regularity_score' in result_df.columns:
            self.feature_columns.extend([
                'regularity_score',
                'is_highly_regular',
                'is_highly_irregular'
            ])
        
        return result_df
    
    def create_amount_gap_interaction_features(self,
                                              df: pd.DataFrame,
                                              amount_col: str = 'amount',
                                              time_since_col: str = 'time_since_last_tx_minutes') -> pd.DataFrame:
        """
        Create features that combine amount and time gap information.
        
        These interactions are powerful fraud indicators:
        - Large amount after short gap (unusual)
        - Small amount after long gap (reactivation)
        - Amount increasing with gap (accumulated spending)
        """
        
        result_df = df.copy()
        
        # Amount per unit time (spending rate)
        result_df['amount_per_minute'] = (
            result_df[amount_col] / (result_df[time_since_col] + 1e-8)
        )
        result_df['amount_per_minute'] = result_df['amount_per_minute'].clip(0, 1e6)
        
        result_df['amount_per_hour'] = result_df['amount_per_minute'] * 60
        result_df['amount_per_day'] = result_df['amount_per_minute'] * 60 * 24
        
        # Amount normalized by gap (for comparability)
        # This tells us if the amount is proportional to the waiting time
        avg_spending_rate = result_df.groupby('customer_id')['amount_per_minute'].transform('median')
        result_df['amount_rate_ratio'] = (
            result_df['amount_per_minute'] / (avg_spending_rate + 1e-8)
        )
        result_df['amount_rate_ratio'] = result_df['amount_rate_ratio'].clip(0, 10)
        
        # Is amount unusually high given the short gap?
        result_df['high_amount_short_gap_flag'] = (
            (result_df[amount_col] > result_df.groupby('customer_id')[amount_col].transform(
                lambda x: x.quantile(0.95)
            )) &
            (result_df[time_since_col] < 60)  # Less than 1 hour
        ).astype(int)
        
        # Is amount unusually low given the long gap?
        result_df['low_amount_long_gap_flag'] = (
            (result_df[amount_col] < result_df.groupby('customer_id')[amount_col].transform(
                lambda x: x.quantile(0.25)
            )) &
            (result_df[time_since_col] > 7*24*60)  # More than 7 days
        ).astype(int)
        
        # Gap categories with amount statistics
        gap_bins = [0, 5, 30, 60, 180, 720, 1440, 10080, float('inf')]  # in minutes
        gap_labels = ['<5m', '5-30m', '30-60m', '1-3h', '3-12h', '12-24h', '1-7d', '>7d']
        
        result_df['gap_category'] = pd.cut(
            result_df[time_since_col],
            bins=gap_bins,
            labels=gap_labels
        )
        
        # For each gap category, what's the typical amount?
        for category in gap_labels:
            category_mask = result_df['gap_category'] == category
            if category_mask.any():
                typical_amount = result_df.loc[category_mask, amount_col].median()
                result_df[f'amount_vs_typical_{category}'] = (
                    result_df[amount_col] / (typical_amount + 1e-8)
                )
                result_df[f'amount_vs_typical_{category}'] = (
                    result_df[f'amount_vs_typical_{category}'].fillna(1).clip(0, 10)
                )
                self.feature_columns.append(f'amount_vs_typical_{category}')
        
        self.feature_columns.extend([
            'amount_per_minute',
            'amount_per_hour',
            'amount_per_day',
            'amount_rate_ratio',
            'high_amount_short_gap_flag',
            'low_amount_long_gap_flag'
        ])
        
        return result_df
    
    def create_cumulative_gap_features(self,
                                      df: pd.DataFrame,
                                      customer_id_col: str = 'customer_id',
                                      time_col: str = 'transaction_time') -> pd.DataFrame:
        """
        Create features based on cumulative time patterns.
        
        These capture long-term behavior changes:
        - Account age
        - Active periods vs dormant periods
        - Cumulative active time
        """
        
        result_df = df.copy()
        
        # Account age at transaction time
        first_tx_time = result_df.groupby(customer_id_col)[time_col].transform('min')
        result_df['account_age_days'] = (
            (result_df[time_col] - first_tx_time).dt.total_seconds() / (24 * 3600)
        )
        
        # Days since first transaction
        result_df['days_since_first_tx'] = result_df['account_age_days']
        
        # Number of active days
        result_df['date'] = result_df[time_col].dt.date
        active_days = (
            result_df
            .groupby(customer_id_col)['date']
            .transform('nunique')
        )
        result_df['total_active_days'] = active_days
        
        # Active ratio (active days / account age)
        result_df['active_ratio'] = (
            result_df['total_active_days'] / (result_df['account_age_days'] + 1)
        )
        
        # Longest dormant period
        # First create a series of dates with transactions
        def get_max_dormant_period(group):
            dates = sorted(group['date'].unique())
            if len(dates) < 2:
                return 0
            
            gaps = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
            return max(gaps) if gaps else 0
        
        max_dormant = (
            result_df
            .groupby(customer_id_col)
            .apply(get_max_dormant_period)
            .rename('max_dormant_days')
        )
        result_df = result_df.merge(max_dormant, on=customer_id_col, how='left')
        
        # Current dormant period
        last_tx = result_df.groupby(customer_id_col)[time_col].transform('max')
        result_df['current_dormant_days'] = (
            (last_tx - result_df[time_col]).dt.total_seconds() / (24 * 3600)
        )
        
        self.feature_columns.extend([
            'account_age_days',
            'days_since_first_tx',
            'total_active_days',
            'active_ratio',
            'max_dormant_days',
            'current_dormant_days'
        ])
        
        return result_df
    
    def get_feature_names(self) -> List[str]:
        """Return list of created feature names."""
        return self.feature_columns