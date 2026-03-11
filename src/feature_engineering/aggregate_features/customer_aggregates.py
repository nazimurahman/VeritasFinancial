"""
Customer Aggregate Features for Banking Fraud Detection
=======================================================
This module implements features aggregated at the customer level.
These features capture the customer's historical behavior patterns and
provide context for evaluating individual transactions.

Key Concepts:
- Customer profiling: Building behavioral baselines
- Historical statistics: Mean, median, percentiles of past behavior
- Risk indicators: Flags based on customer history
- Deviation scores: How current transaction differs from customer norms
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy import stats


class CustomerAggregateFeatureEngineer:
    """
    Create customer-level aggregate features.
    
    Banking Context:
    - Each customer has a unique spending fingerprint
    - Deviations from this fingerprint may indicate fraud
    - New customers have different risk profiles than established ones
    - Customer segments (student, premium, business) have different patterns
    """
    
    def __init__(self):
        self.feature_columns = []
        self.customer_profiles = {}  # Store customer profiles for real-time updates
        
    def create_customer_profile_features(self,
                                        df: pd.DataFrame,
                                        customer_id_col: str = 'customer_id',
                                        amount_col: str = 'amount',
                                        time_col: str = 'transaction_time') -> pd.DataFrame:
        """
        Create comprehensive customer profile features based on historical data.
        
        This creates a "fingerprint" of normal customer behavior that can be
        used to detect anomalies.
        
        Parameters:
        -----------
        df : DataFrame with transaction data
        customer_id_col : Customer identifier
        amount_col : Transaction amount column
        time_col : Timestamp column
        
        Returns:
        --------
        DataFrame with customer profile features
        """
        
        result_df = df.copy()
        
        # Basic customer statistics (aggregated across all history)
        customer_stats = result_df.groupby(customer_id_col).agg({
            amount_col: [
                ('customer_total_spent', 'sum'),
                ('customer_avg_amount', 'mean'),
                ('customer_std_amount', 'std'),
                ('customer_median_amount', 'median'),
                ('customer_min_amount', 'min'),
                ('customer_max_amount', 'max'),
                ('customer_amount_range', lambda x: x.max() - x.min()),
                ('customer_amount_skew', lambda x: stats.skew(x) if len(x) > 2 else 0),
                ('customer_amount_kurtosis', lambda x: stats.kurtosis(x) if len(x) > 3 else 0),
                ('customer_amount_q1', lambda x: x.quantile(0.25)),
                ('customer_amount_q3', lambda x: x.quantile(0.75)),
                ('customer_amount_iqr', lambda x: x.quantile(0.75) - x.quantile(0.25)),
                ('customer_tx_count', 'count'),
                ('customer_unique_days', lambda x: x.nunique() if hasattr(x, 'nunique') else 1)
            ],
            time_col: [
                ('customer_first_tx', 'min'),
                ('customer_last_tx', 'max'),
                ('customer_avg_daily_tx', lambda x: len(x) / max(1, (x.max() - x.min()).days + 1))
            ]
        })
        
        # Flatten multi-level columns
        customer_stats.columns = ['_'.join(col).strip() for col in customer_stats.columns.values]
        customer_stats = customer_stats.reset_index()
        
        # Calculate additional derived features
        customer_stats['customer_avg_daily_spend'] = (
            customer_stats['customer_total_spent'] / 
            (customer_stats['customer_tx_count'] / (customer_stats['customer_avg_daily_tx'] + 1e-8) + 1)
        )
        
        customer_stats['customer_spend_consistency'] = (
            customer_stats['customer_std_amount'] / (customer_stats['customer_avg_amount'] + 1e-8)
        )
        
        # Fill NaN values (for customers with only one transaction)
        fill_values = {
            'customer_std_amount': 0,
            'customer_amount_skew': 0,
            'customer_amount_kurtosis': 0,
            'customer_spend_consistency': 0
        }
        customer_stats = customer_stats.fillna(fill_values)
        
        # Merge back to original dataframe
        result_df = result_df.merge(customer_stats, on=customer_id_col, how='left')
        
        # Store profiles for later use
        self.customer_profiles = customer_stats.set_index(customer_id_col).to_dict(orient='index')
        
        # Add features that compare current transaction to customer profile
        result_df['amount_vs_customer_avg'] = (
            result_df[amount_col] / (result_df['customer_avg_amount'] + 1e-8)
        )
        result_df['amount_vs_customer_avg'] = result_df['amount_vs_customer_avg'].clip(0, 20)
        
        result_df['amount_vs_customer_median'] = (
            result_df[amount_col] / (result_df['customer_median_amount'] + 1e-8)
        )
        result_df['amount_vs_customer_median'] = result_df['amount_vs_customer_median'].clip(0, 20)
        
        result_df['amount_zscore_vs_customer'] = (
            (result_df[amount_col] - result_df['customer_avg_amount']) /
            (result_df['customer_std_amount'] + 1e-8)
        )
        result_df['amount_zscore_vs_customer'] = result_df['amount_zscore_vs_customer'].clip(-10, 10)
        
        result_df['amount_percentile_vs_customer'] = result_df.groupby(customer_id_col)[amount_col].transform(
            lambda x: (x.rank(pct=True) * 100).astype(int)
        )
        
        # Is amount outside normal range?
        result_df['is_amount_outside_iqr'] = (
            (result_df[amount_col] < result_df['customer_amount_q1'] - 1.5 * result_df['customer_amount_iqr']) |
            (result_df[amount_col] > result_df['customer_amount_q3'] + 1.5 * result_df['customer_amount_iqr'])
        ).astype(int)
        
        # Is this the largest transaction ever for this customer?
        result_df['is_new_max_amount'] = (
            result_df[amount_col] >= result_df['customer_max_amount']
        ).astype(int)
        
        # Customer tenure features
        result_df['customer_tenure_days'] = (
            pd.Timestamp.now() - result_df['customer_first_tx']
        ).dt.total_seconds() / (24 * 3600)
        
        result_df['customer_tenure_months'] = result_df['customer_tenure_days'] / 30
        
        result_df['days_since_last_customer_tx'] = (
            pd.Timestamp.now() - result_df['customer_last_tx']
        ).dt.total_seconds() / (24 * 3600)
        
        self.feature_columns.extend([
            'customer_total_spent',
            'customer_avg_amount',
            'customer_std_amount',
            'customer_median_amount',
            'customer_min_amount',
            'customer_max_amount',
            'customer_amount_range',
            'customer_amount_skew',
            'customer_amount_kurtosis',
            'customer_amount_q1',
            'customer_amount_q3',
            'customer_amount_iqr',
            'customer_tx_count',
            'customer_unique_days',
            'customer_first_tx',
            'customer_last_tx',
            'customer_avg_daily_tx',
            'customer_avg_daily_spend',
            'customer_spend_consistency',
            'amount_vs_customer_avg',
            'amount_vs_customer_median',
            'amount_zscore_vs_customer',
            'amount_percentile_vs_customer',
            'is_amount_outside_iqr',
            'is_new_max_amount',
            'customer_tenure_days',
            'customer_tenure_months',
            'days_since_last_customer_tx'
        ])
        
        return result_df
    
    def create_customer_segment_features(self,
                                        df: pd.DataFrame,
                                        customer_id_col: str = 'customer_id') -> pd.DataFrame:
        """
        Create features that segment customers based on their behavior.
        
        Banking Context:
        - Different customer segments have different risk profiles
        - High-value customers may be targeted for fraud
        - Inactive customers suddenly becoming active is suspicious
        """
        
        result_df = df.copy()
        
        # Ensure customer profile features exist
        required_cols = ['customer_tx_count', 'customer_avg_amount', 'customer_tenure_days']
        for col in required_cols:
            if col not in result_df.columns:
                result_df = self.create_customer_profile_features(result_df, customer_id_col)
                break
        
        # Customer value segments
        result_df['customer_value_segment'] = pd.cut(
            result_df['customer_avg_amount'],
            bins=[0, 50, 200, 1000, float('inf')],
            labels=['low_value', 'medium_value', 'high_value', 'premium']
        )
        
        # Customer activity segments
        result_df['customer_activity_segment'] = pd.cut(
            result_df['customer_avg_daily_tx'],
            bins=[0, 0.1, 0.5, 2, float('inf')],
            labels=['inactive', 'occasional', 'regular', 'frequent']
        )
        
        # Customer tenure segments
        result_df['customer_tenure_segment'] = pd.cut(
            result_df['customer_tenure_days'],
            bins=[0, 30, 180, 365, float('inf')],
            labels=['new', 'established', 'long_term', 'veteran']
        )
        
        # Combined risk segment (simplified rule-based)
        conditions = [
            # Premium customers with frequent activity
            (result_df['customer_value_segment'] == 'premium') & 
            (result_df['customer_activity_segment'] == 'frequent'),
            
            # New customers with high value
            (result_df['customer_tenure_segment'] == 'new') & 
            (result_df['customer_value_segment'].isin(['high_value', 'premium'])),
            
            # Long-term inactive customers
            (result_df['customer_activity_segment'] == 'inactive') & 
            (result_df['customer_tenure_segment'].isin(['long_term', 'veteran'])),
            
            # Regular established customers (baseline)
            (result_df['customer_tenure_segment'] == 'established')
        ]
        
        choices = ['high_target_risk', 'new_high_value_risk', 'reactivation_risk', 'normal']
        
        result_df['customer_risk_profile'] = np.select(
            conditions, 
            choices, 
            default='normal'
        )
        
        # Encode segments as numeric features for ML
        segment_mappings = {
            'customer_value_segment': {
                'low_value': 1, 'medium_value': 2, 'high_value': 3, 'premium': 4
            },
            'customer_activity_segment': {
                'inactive': 1, 'occasional': 2, 'regular': 3, 'frequent': 4
            },
            'customer_tenure_segment': {
                'new': 1, 'established': 2, 'long_term': 3, 'veteran': 4
            },
            'customer_risk_profile': {
                'normal': 0, 'reactivation_risk': 1, 'new_high_value_risk': 2, 'high_target_risk': 3
            }
        }
        
        for col, mapping in segment_mappings.items():
            result_df[f'{col}_encoded'] = result_df[col].map(mapping)
            self.feature_columns.append(f'{col}_encoded')
        
        return result_df
    
    def create_customer_velocity_features(self,
                                         df: pd.DataFrame,
                                         customer_id_col: str = 'customer_id',
                                         amount_col: str = 'amount',
                                         time_col: str = 'transaction_time') -> pd.DataFrame:
        """
        Create velocity features specific to customer behavior patterns.
        
        These capture how quickly the customer's behavior is changing,
        which can indicate gradual account takeover.
        """
        
        result_df = df.copy()
        result_df = result_df.sort_values([customer_id_col, time_col])
        
        # Calculate velocity of amount (how quickly spending is changing)
        result_df['amount_velocity'] = (
            result_df
            .groupby(customer_id_col)[amount_col]
            .diff()
            .abs()
        )
        
        # Rolling average of velocity
        for window in [5, 10]:
            result_df[f'avg_velocity_last_{window}'] = (
                result_df
                .groupby(customer_id_col)['amount_velocity']
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
            
            result_df[f'velocity_trend_{window}'] = (
                result_df
                .groupby(customer_id_col)['amount_velocity']
                .transform(lambda x: x.rolling(window, min_periods=1)
                          .apply(lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0))
            )
            
            self.feature_columns.extend([
                f'avg_velocity_last_{window}',
                f'velocity_trend_{window}'
            ])
        
        return result_df
    
    def create_customer_recency_features(self,
                                        df: pd.DataFrame,
                                        customer_id_col: str = 'customer_id',
                                        time_col: str = 'transaction_time') -> pd.DataFrame:
        """
        Create recency-based features (RFM analysis style).
        
        Recency, Frequency, Monetary (RFM) is a classic customer
        segmentation approach that works well for fraud detection.
        """
        
        result_df = df.copy()
        
        # Calculate recency score (how recent was last transaction)
        current_time = result_df[time_col].max()
        recency = result_df.groupby(customer_id_col)[time_col].max().reset_index()
        recency.columns = [customer_id_col, 'last_tx_time']
        recency['recency_days'] = (current_time - recency['last_tx_time']).dt.total_seconds() / (24 * 3600)
        
        # Calculate frequency score (how often they transact)
        frequency = result_df.groupby(customer_id_col).size().reset_index(name='frequency_count')
        
        # Calculate monetary score (how much they spend)
        monetary = result_df.groupby(customer_id_col)[amount_col].sum().reset_index(name='monetary_total')
        
        # Merge all RFM features
        rfm = recency.merge(frequency, on=customer_id_col).merge(monetary, on=customer_id_col)
        
        # Create RFM scores (1-5 scale)
        rfm['recency_score'] = pd.qcut(rfm['recency_days'], q=5, labels=[5,4,3,2,1], duplicates='drop')
        rfm['frequency_score'] = pd.qcut(rfm['frequency_count'], q=5, labels=[1,2,3,4,5], duplicates='drop')
        rfm['monetary_score'] = pd.qcut(rfm['monetary_total'], q=5, labels=[1,2,3,4,5], duplicates='drop')
        
        # Combine into RFM score
        rfm['rfm_score'] = (
            rfm['recency_score'].astype(int) + 
            rfm['frequency_score'].astype(int) + 
            rfm['monetary_score'].astype(int)
        )
        
        # RFM segments
        rfm['rfm_segment'] = pd.cut(
            rfm['rfm_score'],
            bins=[0, 6, 9, 12, 15],
            labels=['low_value', 'medium_value', 'high_value', 'top_value']
        )
        
        # Merge back to original dataframe
        result_df = result_df.merge(
            rfm[[customer_id_col, 'recency_days', 'recency_score', 'frequency_score', 
                 'monetary_score', 'rfm_score', 'rfm_segment']],
            on=customer_id_col,
            how='left'
        )
        
        self.feature_columns.extend([
            'recency_days',
            'recency_score',
            'frequency_score',
            'monetary_score',
            'rfm_score'
        ])
        
        return result_df
    
    def create_customer_anomaly_flags(self,
                                     df: pd.DataFrame,
                                     customer_id_col: str = 'customer_id',
                                     amount_col: str = 'amount') -> pd.DataFrame:
        """
        Create flags for anomalous customer behavior.
        
        These are binary indicators that directly signal potential fraud.
        """
        
        result_df = df.copy()
        
        # Flag: First transaction ever
        result_df['is_first_customer_tx'] = (
            result_df.groupby(customer_id_col).cumcount() == 0
        ).astype(int)
        
        # Flag: Transaction after long inactivity (> 30 days)
        if 'days_since_last_customer_tx' in result_df.columns:
            result_df['is_reactivation'] = (
                result_df['days_since_last_customer_tx'] > 30
            ).astype(int)
        
        # Flag: Unusual time for this customer
        if 'hour' in result_df.columns:
            # Calculate customer's typical hours
            customer_hour_mode = result_df.groupby(customer_id_col)['hour'].agg(
                lambda x: stats.mode(x)[0] if len(x) > 0 else 12
            )
            result_df['customer_typical_hour'] = result_df[customer_id_col].map(customer_hour_mode)
            result_df['is_unusual_hour'] = (
                abs(result_df['hour'] - result_df['customer_typical_hour']) > 6
            ).astype(int)
            self.feature_columns.append('is_unusual_hour')
        
        # Flag: Amount > 3x customer average
        if 'customer_avg_amount' in result_df.columns:
            result_df['is_very_high_amount'] = (
                result_df[amount_col] > 3 * result_df['customer_avg_amount']
            ).astype(int)
            self.feature_columns.append('is_very_high_amount')
        
        # Flag: Unusual merchant for this customer
        if 'merchant_id' in result_df.columns:
            # Get customer's most frequent merchants
            top_merchants = (
                result_df
                .groupby([customer_id_col, 'merchant_id'])
                .size()
                .reset_index(name='count')
                .sort_values([customer_id_col, 'count'], ascending=[True, False])
                .groupby(customer_id_col)
                .head(3)
                .groupby(customer_id_col)['merchant_id']
                .apply(list)
            )
            
            result_df['customer_top_merchants'] = result_df[customer_id_col].map(top_merchants)
            
            # Check if current merchant is in top 3
            result_df['is_usual_merchant'] = result_df.apply(
                lambda row: row['merchant_id'] in row['customer_top_merchants'] 
                if isinstance(row['customer_top_merchants'], list) 
                else False,
                axis=1
            ).astype(int)
            
            result_df['is_unusual_merchant'] = 1 - result_df['is_usual_merchant']
            self.feature_columns.extend(['is_usual_merchant', 'is_unusual_merchant'])
        
        self.feature_columns.extend([
            'is_first_customer_tx'
        ])
        
        if 'is_reactivation' in result_df.columns:
            self.feature_columns.append('is_reactivation')
        
        return result_df
    
    def get_feature_names(self) -> List[str]:
        """Return list of created feature names."""
        return self.feature_columns
    
    def update_customer_profile(self, 
                               customer_id: str, 
                               new_transaction: Dict) -> Dict:
        """
        Update customer profile incrementally with new transaction.
        
        This is used for real-time streaming scenarios where we need
        to update profiles without recomputing all history.
        
        Parameters:
        -----------
        customer_id : Customer identifier
        new_transaction : Dictionary with transaction data
        
        Returns:
        --------
        Updated customer profile
        """
        
        if customer_id not in self.customer_profiles:
            # Initialize new profile
            self.customer_profiles[customer_id] = {
                'customer_tx_count': 0,
                'customer_total_spent': 0,
                'customer_avg_amount': 0,
                'customer_min_amount': float('inf'),
                'customer_max_amount': 0,
                'customer_first_tx': new_transaction['transaction_time'],
                'customer_last_tx': new_transaction['transaction_time']
            }
        
        profile = self.customer_profiles[customer_id]
        
        # Update statistics
        old_count = profile['customer_tx_count']
        new_amount = new_transaction['amount']
        
        # Update count
        profile['customer_tx_count'] = old_count + 1
        
        # Update total and average
        profile['customer_total_spent'] += new_amount
        profile['customer_avg_amount'] = (
            profile['customer_total_spent'] / profile['customer_tx_count']
        )
        
        # Update min/max
        profile['customer_min_amount'] = min(profile['customer_min_amount'], new_amount)
        profile['customer_max_amount'] = max(profile['customer_max_amount'], new_amount)
        
        # Update last transaction time
        profile['customer_last_tx'] = new_transaction['transaction_time']
        
        # Update standard deviation (using Welford's algorithm)
        if old_count == 0:
            profile['customer_m2'] = 0
        else:
            delta = new_amount - profile['customer_avg_amount']
            profile['customer_m2'] = profile.get('customer_m2', 0) + delta * (new_amount - profile['customer_avg_amount'])
        
        if profile['customer_tx_count'] > 1:
            profile['customer_std_amount'] = np.sqrt(profile['customer_m2'] / (profile['customer_tx_count'] - 1))
        
        self.customer_profiles[customer_id] = profile
        
        return profile