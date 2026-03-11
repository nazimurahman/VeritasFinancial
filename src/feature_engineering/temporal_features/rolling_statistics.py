"""
Rolling Statistics for Banking Fraud Detection
=============================================
This module implements time-window based statistical features that capture
transaction patterns over different time periods. These are crucial for
detecting unusual spending behaviors and velocity-based fraud.

Key Concepts:
- Rolling windows: Moving time windows (1h, 24h, 7d, 30d)
- Velocity metrics: Rate of change in transaction patterns
- Statistical moments: Mean, std, skewness of rolling distributions
- Decay factors: Exponential weighting for recent transactions

Banking Domain Context:
- Sudden increase in transaction frequency → potential account takeover
- Unusual spending patterns → card testing, bust-out fraud
- Velocity checks → money mule detection
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')


class RollingStatisticsFeatureEngineer:
    """
    Engineer rolling window statistics for fraud detection.
    
    This class creates temporal features that capture transaction patterns
    over various time windows. It's designed to handle both batch processing
    and real-time streaming scenarios.
    
    Parameters:
    -----------
    windows : List[str]
        List of time windows for rolling calculations (e.g., ['1H', '24H', '7D'])
    min_periods : int
        Minimum number of observations required for calculation
    feature_namespace : str
        Prefix for feature names to avoid collisions
    """
    
    def __init__(self, 
                 windows: List[str] = None,
                 min_periods: int = 1,
                 feature_namespace: str = 'rolling'):
        
        # Default windows if none provided - these are standard in banking fraud
        if windows is None:
            windows = ['1H', '3H', '6H', '12H', '24H', '48H', '7D', '30D']
        
        self.windows = windows
        self.min_periods = min_periods
        self.feature_namespace = feature_namespace
        
        # Validate windows (ensure they're pandas-compatible frequency strings)
        self._validate_windows()
        
        # Store feature metadata
        self.feature_columns = []
        self.statistics_cache = {}  # For real-time incremental updates
        
    def _validate_windows(self):
        """Validate that window strings are properly formatted."""
        valid_frequencies = ['H', 'D', 'W', 'M', 'T', 'S']
        for window in self.windows:
            # Check if window ends with valid frequency
            freq = ''.join([c for c in window if not c.isdigit()])
            if freq not in valid_frequencies:
                raise ValueError(f"Invalid window frequency: {freq}. Must be one of {valid_frequencies}")
    
    def create_transaction_velocity_features(self, 
                                             df: pd.DataFrame,
                                             transaction_time_col: str = 'transaction_time',
                                             customer_id_col: str = 'customer_id',
                                             amount_col: str = 'amount',
                                             transaction_id_col: str = 'transaction_id') -> pd.DataFrame:
        """
        Create velocity-based features (transaction counts and amounts over time).
        
        Velocity features are critical for detecting:
        - Card testing: Many small transactions in short time
        - Account takeover: Sudden spike in transaction frequency
        - Money mule activity: Unusual transaction velocity patterns
        
        Parameters:
        -----------
        df : DataFrame with transaction data
        transaction_time_col : Timestamp column
        customer_id_col : Customer identifier
        amount_col : Transaction amount column
        transaction_id_col : Unique transaction identifier
        
        Returns:
        --------
        DataFrame with original columns plus new velocity features
        """
        
        # Make a copy to avoid modifying original
        result_df = df.copy()
        
        # Ensure transaction_time is datetime type
        if not pd.api.types.is_datetime64_any_dtype(result_df[transaction_time_col]):
            result_df[transaction_time_col] = pd.to_datetime(result_df[transaction_time_col])
        
        # Sort by customer and time for proper rolling calculations
        result_df = result_df.sort_values([customer_id_col, transaction_time_col])
        
        print("Creating transaction velocity features...")
        
        for window in self.windows:
            print(f"  Processing window: {window}")
            
            # Convert window string to pandas frequency
            # e.g., '24H' -> '24H' for pandas rolling
            
            # Count of transactions in window (frequency velocity)
            count_feature_name = f'{self.feature_namespace}_tx_count_{window}'
            result_df[count_feature_name] = (
                result_df
                .groupby(customer_id_col)[transaction_time_col]
                .transform(lambda x: x.rolling(window, min_periods=self.min_periods).count())
            )
            self.feature_columns.append(count_feature_name)
            
            # Sum of amounts in window (monetary velocity)
            sum_feature_name = f'{self.feature_namespace}_amount_sum_{window}'
            result_df[sum_feature_name] = (
                result_df
                .groupby(customer_id_col)[amount_col]
                .transform(lambda x: x.rolling(window, min_periods=self.min_periods).sum())
            )
            self.feature_columns.append(sum_feature_name)
            
            # Mean transaction amount in window
            mean_feature_name = f'{self.feature_namespace}_amount_mean_{window}'
            result_df[mean_feature_name] = (
                result_df
                .groupby(customer_id_col)[amount_col]
                .transform(lambda x: x.rolling(window, min_periods=self.min_periods).mean())
            )
            self.feature_columns.append(mean_feature_name)
            
            # Standard deviation of amounts (volatility indicator)
            std_feature_name = f'{self.feature_namespace}_amount_std_{window}'
            result_df[std_feature_name] = (
                result_df
                .groupby(customer_id_col)[amount_col]
                .transform(lambda x: x.rolling(window, min_periods=self.min_periods).std())
            )
            # Fill NaN with 0 for first transactions (no history)
            result_df[std_feature_name] = result_df[std_feature_name].fillna(0)
            self.feature_columns.append(std_feature_name)
            
            # Max transaction amount in window (peak spending)
            max_feature_name = f'{self.feature_namespace}_amount_max_{window}'
            result_df[max_feature_name] = (
                result_df
                .groupby(customer_id_col)[amount_col]
                .transform(lambda x: x.rolling(window, min_periods=self.min_periods).max())
            )
            self.feature_columns.append(max_feature_name)
            
            # Min transaction amount in window (floor spending)
            min_feature_name = f'{self.feature_namespace}_amount_min_{window}'
            result_df[min_feature_name] = (
                result_df
                .groupby(customer_id_col)[amount_col]
                .transform(lambda x: x.rolling(window, min_periods=self.min_periods).min())
            )
            self.feature_columns.append(min_feature_name)
            
            # Ratio of current amount to rolling mean (deviation from normal)
            # Values >> 1 indicate unusually large transactions
            ratio_feature_name = f'{self.feature_namespace}_amount_to_mean_ratio_{window}'
            result_df[ratio_feature_name] = result_df[amount_col] / (result_df[mean_feature_name] + 1e-8)
            # Clip extreme values to avoid outliers
            result_df[ratio_feature_name] = result_df[ratio_feature_name].clip(0, 10)
            self.feature_columns.append(ratio_feature_name)
            
            # Z-score of amount relative to rolling window (standardized deviation)
            # Values > 3 indicate significant outliers
            zscore_feature_name = f'{self.feature_namespace}_amount_zscore_{window}'
            result_df[zscore_feature_name] = (
                (result_df[amount_col] - result_df[mean_feature_name]) / 
                (result_df[std_feature_name] + 1e-8)
            )
            result_df[zscore_feature_name] = result_df[zscore_feature_name].clip(-10, 10)
            self.feature_columns.append(zscore_feature_name)
            
        return result_df
    
    def create_exponential_weighted_features(self,
                                             df: pd.DataFrame,
                                             customer_id_col: str = 'customer_id',
                                             amount_col: str = 'amount',
                                             transaction_time_col: str = 'transaction_time',
                                             alpha: float = 0.3) -> pd.DataFrame:
        """
        Create exponentially weighted moving averages (EWMA).
        
        EWMA gives more weight to recent transactions, capturing
        recency effects better than simple rolling windows.
        
        Parameters:
        -----------
        alpha : Smoothing factor (0 < alpha <= 1)
                Higher alpha = more weight to recent observations
        """
        
        result_df = df.copy()
        
        # Sort for proper EWMA calculation
        result_df = result_df.sort_values([customer_id_col, transaction_time_col])
        
        # EWMA of amounts
        ewma_feature_name = f'{self.feature_namespace}_amount_ewma_{alpha}'
        result_df[ewma_feature_name] = (
            result_df
            .groupby(customer_id_col)[amount_col]
            .transform(lambda x: x.ewm(alpha=alpha, min_periods=self.min_periods).mean())
        )
        self.feature_columns.append(ewma_feature_name)
        
        # EWMA of transaction counts (can't directly do counts, so create time-based)
        # First create a time-weighted feature
        result_df['time_weight'] = np.exp(-alpha * 
                                          (result_df[transaction_time_col].max() - 
                                           result_df[transaction_time_col]).dt.total_seconds() / 3600)
        
        # Calculate weighted sum and count
        weighted_sum = (
            result_df
            .groupby(customer_id_col)
            .apply(lambda x: (x[amount_col] * x['time_weight']).sum())
        )
        
        # Map back to original rows
        result_df[f'{self.feature_namespace}_weighted_avg'] = (
            result_df[customer_id_col].map(weighted_sum) / 
            (result_df.groupby(customer_id_col)['time_weight'].transform('sum') + 1e-8)
        )
        
        return result_df
    
    def create_velocity_change_features(self,
                                        df: pd.DataFrame,
                                        short_window: str = '1H',
                                        long_window: str = '24H') -> pd.DataFrame:
        """
        Detect sudden changes in transaction velocity.
        
        Ratio of short-term to long-term velocity helps identify
        abrupt behavioral changes that often indicate fraud.
        
        Returns:
        --------
        Features:
        - velocity_change_ratio: short_window_count / long_window_count
        - velocity_acceleration: rate of change in velocity
        """
        
        result_df = df.copy()
        
        # Get counts for both windows
        short_count = f'{self.feature_namespace}_tx_count_{short_window}'
        long_count = f'{self.feature_namespace}_tx_count_{long_window}'
        
        # Ensure these features exist
        if short_count not in result_df.columns:
            result_df = self.create_transaction_velocity_features(
                result_df, windows=[short_window]
            )
        
        if long_count not in result_df.columns:
            result_df = self.create_transaction_velocity_features(
                result_df, windows=[long_window]
            )
        
        # Velocity change ratio (normalized by time window lengths)
        # Convert windows to hours for fair comparison
        short_hours = self._window_to_hours(short_window)
        long_hours = self._window_to_hours(long_window)
        
        # Normalize rates per hour
        short_rate = result_df[short_count] / short_hours
        long_rate = result_df[long_count] / long_hours
        
        # Velocity change ratio (> 1 means recent acceleration)
        result_df[f'{self.feature_namespace}_velocity_change_ratio'] = (
            short_rate / (long_rate + 1e-8)
        )
        result_df[f'{self.feature_namespace}_velocity_change_ratio'] = (
            result_df[f'{self.feature_namespace}_velocity_change_ratio'].clip(0, 10)
        )
        
        # Absolute change in velocity
        result_df[f'{self.feature_namespace}_velocity_abs_change'] = short_rate - long_rate
        
        # Binary flag for sudden velocity spike (> 3x normal)
        result_df[f'{self.feature_namespace}_velocity_spike_flag'] = (
            (result_df[f'{self.feature_namespace}_velocity_change_ratio'] > 3).astype(int)
        )
        
        return result_df
    
    def _window_to_hours(self, window: str) -> float:
        """Convert pandas window string to hours."""
        import re
        # Extract number and unit
        match = re.match(r'(\d+)([HDWM])', window)
        if not match:
            return 1.0
        
        value, unit = int(match.group(1)), match.group(2)
        
        # Convert to hours
        if unit == 'H':
            return value
        elif unit == 'D':
            return value * 24
        elif unit == 'W':
            return value * 24 * 7
        elif unit == 'M':
            return value * 24 * 30  # Approximate
        else:
            return value
    
    def create_session_based_features(self,
                                      df: pd.DataFrame,
                                      session_timeout_minutes: int = 30) -> pd.DataFrame:
        """
        Create features based on transaction sessions.
        
        A "session" is defined as a sequence of transactions with
        gaps less than session_timeout_minutes.
        
        Banking Context:
        - Multiple transactions in quick succession may indicate
          automated fraud attempts
        - Session-based features capture this pattern
        """
        
        result_df = df.copy()
        result_df = result_df.sort_values(['customer_id', 'transaction_time'])
        
        # Calculate time difference from previous transaction
        result_df['time_since_prev'] = (
            result_df
            .groupby('customer_id')['transaction_time']
            .diff()
            .dt.total_seconds() / 60  # Convert to minutes
        )
        
        # Identify session boundaries (gaps > timeout)
        result_df['new_session_flag'] = (
            (result_df['time_since_prev'] > session_timeout_minutes) | 
            (result_df['time_since_prev'].isna())
        ).astype(int)
        
        # Create session IDs
        result_df['session_id'] = result_df.groupby('customer_id')['new_session_flag'].cumsum()
        
        # Session-level features
        session_features = result_df.groupby(['customer_id', 'session_id']).agg({
            'amount': ['count', 'sum', 'mean', 'std'],
            'transaction_id': 'count'
        }).round(2)
        
        session_features.columns = ['_'.join(col).strip() for col in session_features.columns.values]
        session_features = session_features.rename(columns={
            'amount_count': 'session_tx_count',
            'amount_sum': 'session_total_amount',
            'amount_mean': 'session_avg_amount',
            'amount_std': 'session_std_amount',
            'transaction_id_count': 'session_tx_count_alt'
        })
        session_features = session_features.reset_index()
        
        # Merge back to original
        result_df = result_df.merge(
            session_features, 
            on=['customer_id', 'session_id'], 
            how='left'
        )
        
        # Session position features
        result_df['tx_position_in_session'] = (
            result_df
            .groupby(['customer_id', 'session_id'])
            .cumcount() + 1
        )
        
        result_df['is_last_in_session'] = (
            result_df
            .groupby(['customer_id', 'session_id'])['transaction_time']
            .transform('max') == result_df['transaction_time']
        ).astype(int)
        
        # Session velocity
        result_df['session_tx_per_minute'] = (
            result_df['session_tx_count'] / 
            (result_df['session_id'].map(result_df.groupby('session_id')['transaction_time']
                                         .transform(lambda x: (x.max() - x.min()).total_seconds() / 60)) + 1)
        )
        
        return result_df
    
    def get_feature_names(self) -> List[str]:
        """Return list of created feature names."""
        return self.feature_columns
    
    def get_feature_importance_hints(self) -> Dict[str, str]:
        """
        Return dictionary of feature importance hints for domain knowledge.
        
        This helps data scientists understand which features are most
        important for fraud detection in the banking context.
        """
        return {
            'rolling_tx_count_1H': 'High importance - Velocity of transactions in last hour',
            'rolling_tx_count_24H': 'High importance - Daily transaction frequency',
            'rolling_amount_zscore_24H': 'High importance - Unusual transaction amounts',
            'rolling_velocity_change_ratio': 'High importance - Sudden behavioral change',
            'rolling_amount_to_mean_ratio_24H': 'Medium importance - Deviation from normal',
            'rolling_amount_std_7D': 'Low importance - Long-term volatility baseline'
        }