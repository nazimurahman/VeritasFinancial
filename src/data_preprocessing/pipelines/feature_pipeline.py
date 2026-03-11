"""
Feature Pipeline Module

This module provides a specialized pipeline for feature engineering
that builds upon the preprocessing pipeline. It focuses on creating
advanced features for fraud detection.

Features created:
1. Transaction velocity features
2. Customer behavioral features
3. Merchant risk features
4. Temporal pattern features
5. Graph-based features
6. Interaction features
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import timedelta
import warnings

logger = logging.getLogger(__name__)


class FeaturePipeline(BaseEstimator, TransformerMixin):
    """
    Advanced feature engineering pipeline for fraud detection.
    
    This pipeline creates sophisticated features that capture fraud patterns
    beyond basic preprocessing.
    
    Attributes:
        feature_groups (list): Groups of features to create
        window_sizes (list): Time windows for rolling features
        aggregation_functions (list): Functions for aggregation
        create_interactions (bool): Whether to create interaction features
        create_velocity_features (bool): Whether to create velocity features
        create_behavioral_features (bool): Whether to create behavioral features
    """
    
    def __init__(
        self,
        feature_groups: Optional[List[str]] = None,
        window_sizes: Optional[List[str]] = None,
        aggregation_functions: Optional[List[str]] = None,
        create_interactions: bool = True,
        create_velocity_features: bool = True,
        create_behavioral_features: bool = True,
        create_graph_features: bool = True,
        max_interaction_features: int = 100,
        customer_id_column: str = 'customer_id',
        transaction_time_column: str = 'transaction_time',
        amount_column: str = 'amount',
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize the FeaturePipeline.
        
        Args:
            feature_groups: List of feature groups to create
            window_sizes: Time windows for rolling features (e.g., ['1H', '24H', '7D'])
            aggregation_functions: Functions for aggregation
            create_interactions: Whether to create interaction features
            create_velocity_features: Whether to create velocity features
            create_behavioral_features: Whether to create behavioral features
            create_graph_features: Whether to create graph features
            max_interaction_features: Maximum number of interaction features
            customer_id_column: Name of customer ID column
            transaction_time_column: Name of transaction time column
            amount_column: Name of amount column
            random_state: Random state for reproducibility
            **kwargs: Additional arguments
        """
        self.feature_groups = feature_groups or [
            'velocity', 'behavioral', 'temporal', 'merchant', 'graph', 'interaction'
        ]
        self.window_sizes = window_sizes or ['1H', '24H', '7D', '30D']
        self.aggregation_functions = aggregation_functions or [
            'count', 'sum', 'mean', 'std', 'min', 'max', 'skew'
        ]
        self.create_interactions = create_interactions
        self.create_velocity_features = create_velocity_features
        self.create_behavioral_features = create_behavioral_features
        self.create_graph_features = create_graph_features
        self.max_interaction_features = max_interaction_features
        self.customer_id_column = customer_id_column
        self.transaction_time_column = transaction_time_column
        self.amount_column = amount_column
        self.random_state = random_state
        self.kwargs = kwargs
        
        # Storage for fitted data
        self.customer_profiles = {}
        self.merchant_profiles = {}
        self.feature_stats = {}
        self.feature_names_ = []
        
        logger.info(f"FeaturePipeline initialized with groups: {self.feature_groups}")
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the feature pipeline by building customer/merchant profiles.
        
        Args:
            X: Input DataFrame
            y: Optional target values
            
        Returns:
            self
        """
        logger.info(f"Fitting FeaturePipeline on {len(X)} samples")
        
        # Build customer profiles for behavioral features
        if self.create_behavioral_features and self.customer_id_column in X.columns:
            self.customer_profiles = self._build_customer_profiles(X, y)
        
        # Build merchant profiles if needed
        if 'merchant' in self.feature_groups and 'merchant_id' in X.columns:
            self.merchant_profiles = self._build_merchant_profiles(X, y)
        
        # Calculate feature statistics
        self.feature_stats = self._calculate_feature_stats(X)
        
        # Generate feature names
        self._generate_feature_names(X)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by creating advanced features.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with added features
        """
        logger.info(f"Transforming {len(X)} samples with feature engineering")
        
        result = X.copy()
        
        # Create velocity features
        if self.create_velocity_features and 'velocity' in self.feature_groups:
            result = self._create_velocity_features(result)
        
        # Create behavioral features
        if self.create_behavioral_features and 'behavioral' in self.feature_groups:
            result = self._create_behavioral_features(result)
        
        # Create temporal features
        if 'temporal' in self.feature_groups:
            result = self._create_temporal_features(result)
        
        # Create merchant features
        if 'merchant' in self.feature_groups:
            result = self._create_merchant_features(result)
        
        # Create graph features
        if self.create_graph_features and 'graph' in self.feature_groups:
            result = self._create_graph_features(result)
        
        # Create interaction features
        if self.create_interactions and 'interaction' in self.feature_groups:
            result = self._create_interaction_features(result)
        
        logger.info(f"Feature engineering complete. New shape: {result.shape}")
        return result
    
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
    
    def _build_customer_profiles(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict:
        """
        Build customer profiles for behavioral analysis.
        
        Args:
            df: Input DataFrame
            y: Target values
            
        Returns:
            Dictionary with customer profiles
        """
        profiles = {}
        
        # Group by customer
        for customer_id, group in df.groupby(self.customer_id_column):
            profile = {
                'total_transactions': len(group),
                'avg_amount': group[self.amount_column].mean(),
                'std_amount': group[self.amount_column].std(),
                'min_amount': group[self.amount_column].min(),
                'max_amount': group[self.amount_column].max(),
                'total_spent': group[self.amount_column].sum(),
                'unique_merchants': group['merchant_id'].nunique() if 'merchant_id' in group.columns else 0,
                'avg_time_between_txs': self._calculate_avg_time_between(group)
            }
            
            # Add fraud statistics if available
            if y is not None and customer_id in y.index:
                profile['fraud_count'] = y[group.index].sum() if customer_id in y.index else 0
            
            profiles[customer_id] = profile
        
        logger.info(f"Built profiles for {len(profiles)} customers")
        return profiles
    
    def _build_merchant_profiles(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict:
        """
        Build merchant profiles for risk analysis.
        
        Args:
            df: Input DataFrame
            y: Target values
            
        Returns:
            Dictionary with merchant profiles
        """
        profiles = {}
        
        if 'merchant_id' not in df.columns:
            return profiles
        
        for merchant_id, group in df.groupby('merchant_id'):
            profile = {
                'total_transactions': len(group),
                'avg_amount': group[self.amount_column].mean(),
                'std_amount': group[self.amount_column].std(),
                'total_volume': group[self.amount_column].sum(),
                'unique_customers': group[self.customer_id_column].nunique(),
            }
            
            # Add fraud rate if available
            if y is not None:
                fraud_count = y[group.index].sum() if merchant_id in y.index else 0
                profile['fraud_rate'] = fraud_count / len(group) if len(group) > 0 else 0
                profile['fraud_count'] = fraud_count
            
            profiles[merchant_id] = profile
        
        logger.info(f"Built profiles for {len(profiles)} merchants")
        return profiles
    
    def _create_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create transaction velocity features.
        
        Features:
        - Transaction count in various time windows
        - Total amount in various time windows
        - Velocity of transactions (transactions per hour)
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with velocity features
        """
        result = df.copy()
        
        if self.customer_id_column not in df.columns or self.transaction_time_column not in df.columns:
            logger.warning("Cannot create velocity features: missing required columns")
            return result
        
        # Ensure datetime type
        if not pd.api.types.is_datetime64_any_dtype(result[self.transaction_time_column]):
            result[self.transaction_time_column] = pd.to_datetime(result[self.transaction_time_column])
        
        # Sort by time for rolling windows
        result = result.sort_values([self.customer_id_column, self.transaction_time_column])
        
        for window in self.window_sizes:
            # Transaction count in window
            count_col = f'tx_count_{window}'
            result[count_col] = result.groupby(self.customer_id_column)[self.transaction_time_column].transform(
                lambda x: x.rolling(window, min_periods=1).count()
            )
            
            # Total amount in window
            amount_col = f'tx_amount_sum_{window}'
            result[amount_col] = result.groupby(self.customer_id_column)[self.amount_column].transform(
                lambda x: x.rolling(window, min_periods=1).sum()
            )
            
            # Average amount in window
            avg_col = f'tx_amount_avg_{window}'
            result[avg_col] = result.groupby(self.customer_id_column)[self.amount_column].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            
            # Standard deviation in window
            std_col = f'tx_amount_std_{window}'
            result[std_col] = result.groupby(self.customer_id_column)[self.amount_column].transform(
                lambda x: x.rolling(window, min_periods=2).std()
            )
            
            # Velocity (transactions per hour)
            if window.endswith('H'):
                hours = int(window[:-1])
                velocity_col = f'tx_velocity_{window}'
                result[velocity_col] = result[count_col] / hours
        
        return result
    
    def _create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create behavioral pattern features.
        
        Features:
        - Deviation from customer's typical behavior
        - Unusual transaction times
        - Unusual amounts
        - Category preferences
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with behavioral features
        """
        result = df.copy()
        
        if self.customer_id_column not in df.columns:
            logger.warning("Cannot create behavioral features: missing customer_id column")
            return result
        
        # Amount deviation from customer average
        if self.customer_id_column in self.customer_profiles:
            result['amount_deviation'] = result.apply(
                lambda row: self._calculate_amount_deviation(row), axis=1
            )
        
        # Time deviation from customer's typical transaction time
        if self.transaction_time_column in df.columns:
            result['hour_deviation'] = result.apply(
                lambda row: self._calculate_time_deviation(row), axis=1
            )
        
        # Unusual amount flag (beyond 3 standard deviations)
        result['unusual_amount'] = result.apply(
            lambda row: self._is_unusual_amount(row), axis=1
        ).astype(int)
        
        # Unusual time flag (late night/early morning)
        if 'hour_of_day' in df.columns:
            result['unusual_time'] = (
                (result['hour_of_day'] < 4) | (result['hour_of_day'] > 23)
            ).astype(int)
        
        # Weekend transaction flag
        if 'is_weekend' in df.columns:
            result['weekend_tx'] = result['is_weekend']
        
        # Rapid transaction flag (less than 1 minute since last)
        if 'time_since_last_tx' in df.columns:
            result['rapid_tx'] = (result['time_since_last_tx'] < 60).astype(int)
        
        return result
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced temporal features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with temporal features
        """
        result = df.copy()
        
        if self.transaction_time_column not in df.columns:
            return result
        
        # Time since last transaction (already in seconds)
        if 'time_since_last_tx' not in df.columns:
            result = result.sort_values([self.customer_id_column, self.transaction_time_column])
            result['time_since_last_tx'] = result.groupby(self.customer_id_column)[
                self.transaction_time_column
            ].diff().dt.total_seconds()
        
        # Time to next transaction
        result['time_to_next_tx'] = result.groupby(self.customer_id_column)[
            self.transaction_time_column
        ].diff(-1).dt.total_seconds().abs()
        
        # Is first transaction of the day
        result['is_first_tx_day'] = result.groupby(
            [self.customer_id_column, result[self.transaction_time_column].dt.date]
        ).cumcount() == 0
        result['is_first_tx_day'] = result['is_first_tx_day'].astype(int)
        
        # Transaction frequency (transactions per day)
        result['tx_frequency'] = 1 / (result['time_since_last_tx'] / 86400 + 1e-10)
        
        return result
    
    def _create_merchant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create merchant-related features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with merchant features
        """
        result = df.copy()
        
        if 'merchant_id' not in df.columns:
            return result
        
        # Merchant risk score (based on historical fraud rate)
        if self.merchant_profiles:
            result['merchant_risk_score'] = result['merchant_id'].map(
                lambda x: self.merchant_profiles.get(x, {}).get('fraud_rate', 0.5)
            )
        
        # Merchant transaction volume
        result['merchant_volume'] = result['merchant_id'].map(
            lambda x: self.merchant_profiles.get(x, {}).get('total_volume', 0)
        )
        
        # Is new merchant for this customer
        result['is_new_merchant'] = result.groupby(self.customer_id_column)['merchant_id'].transform(
            lambda x: ~x.duplicated()
        ).astype(int)
        
        # Previous transactions with this merchant
        result['merchant_tx_count'] = result.groupby(
            [self.customer_id_column, 'merchant_id']
        ).cumcount() + 1
        
        return result
    
    def _create_graph_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create graph-based features (connections between entities).
        
        Features:
        - Number of customers sharing this device
        - Number of devices used by this customer
        - IP address reputation
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with graph features
        """
        result = df.copy()
        
        # Device sharing features
        if 'device_id' in df.columns:
            # Customers per device
            device_to_customers = df.groupby('device_id')[self.customer_id_column].nunique()
            result['customers_per_device'] = result['device_id'].map(device_to_customers)
            
            # Is shared device
            result['is_shared_device'] = (result['customers_per_device'] > 1).astype(int)
        
        # Devices per customer
        if 'device_id' in df.columns and self.customer_id_column in df.columns:
            customer_devices = df.groupby(self.customer_id_column)['device_id'].nunique()
            result['devices_per_customer'] = result[self.customer_id_column].map(customer_devices)
        
        # IP address features
        if 'ip_address' in df.columns:
            # Customers per IP
            ip_customers = df.groupby('ip_address')[self.customer_id_column].nunique()
            result['customers_per_ip'] = result['ip_address'].map(ip_customers)
            
            # Is shared IP
            result['is_shared_ip'] = (result['customers_per_ip'] > 1).astype(int)
        
        return result
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        result = df.copy()
        
        # Define important numerical columns for interactions
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Select top features based on variance or importance
        if len(numerical_cols) > 10:
            # Use top 10 features by variance
            variances = df[numerical_cols].var()
            top_features = variances.nlargest(10).index.tolist()
        else:
            top_features = numerical_cols
        
        # Create pairwise interactions
        interaction_count = 0
        for i in range(len(top_features)):
            for j in range(i+1, len(top_features)):
                if interaction_count >= self.max_interaction_features:
                    break
                
                col1, col2 = top_features[i], top_features[j]
                interaction_name = f"{col1}_x_{col2}"
                
                # Create product interaction
                result[interaction_name] = df[col1] * df[col2]
                interaction_count += 1
                
                # Create ratio interaction (if denominator not zero)
                ratio_name = f"{col1}_div_{col2}"
                result[ratio_name] = df[col1] / (df[col2] + 1e-10)
                interaction_count += 1
        
        logger.info(f"Created {interaction_count} interaction features")
        return result
    
    def _calculate_amount_deviation(self, row: pd.Series) -> float:
        """
        Calculate deviation of transaction amount from customer average.
        
        Args:
            row: DataFrame row
            
        Returns:
            Deviation score
        """
        customer_id = row.get(self.customer_id_column)
        amount = row.get(self.amount_column, 0)
        
        if customer_id in self.customer_profiles:
            avg_amount = self.customer_profiles[customer_id].get('avg_amount', amount)
            std_amount = self.customer_profiles[customer_id].get('std_amount', 1)
            
            if std_amount > 0:
                return (amount - avg_amount) / std_amount
        
        return 0
    
    def _calculate_time_deviation(self, row: pd.Series) -> float:
        """
        Calculate deviation of transaction time from customer's typical time.
        
        Args:
            row: DataFrame row
            
        Returns:
            Deviation score
        """
        # Simplified version - in production, would use historical time distribution
        if 'hour_of_day' in row:
            # Assume typical shopping hours are 9-17
            typical_hour = 12
            return abs(row['hour_of_day'] - typical_hour) / 12
        
        return 0
    
    def _is_unusual_amount(self, row: pd.Series) -> bool:
        """
        Check if amount is unusual for this customer.
        
        Args:
            row: DataFrame row
            
        Returns:
            True if amount is unusual
        """
        customer_id = row.get(self.customer_id_column)
        amount = row.get(self.amount_column, 0)
        
        if customer_id in self.customer_profiles:
            avg_amount = self.customer_profiles[customer_id].get('avg_amount', amount)
            std_amount = self.customer_profiles[customer_id].get('std_amount', avg_amount * 0.5)
            
            # More than 3 standard deviations from mean
            return abs(amount - avg_amount) > 3 * std_amount
        
        # Without profile, use global threshold
        return amount > 10000  # Example threshold
    
    def _calculate_avg_time_between(self, group: pd.DataFrame) -> float:
        """
        Calculate average time between transactions for a customer.
        
        Args:
            group: Customer's transaction group
            
        Returns:
            Average time in seconds
        """
        if self.transaction_time_column in group.columns and len(group) > 1:
            sorted_times = group[self.transaction_time_column].sort_values()
            time_diffs = sorted_times.diff().dt.total_seconds().iloc[1:]
            return time_diffs.mean() if len(time_diffs) > 0 else 0
        
        return 0
    
    def _calculate_feature_stats(self, df: pd.DataFrame) -> Dict:
        """
        Calculate statistics for created features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with feature statistics
        """
        stats = {}
        
        for col in df.select_dtypes(include=[np.number]).columns:
            stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'missing': df[col].isna().sum()
            }
        
        return stats
    
    def _generate_feature_names(self, X: pd.DataFrame):
        """
        Generate names of features after transformation.
        
        Args:
            X: Input DataFrame
        """
        # This would be populated during transform
        pass
    
    def get_feature_names_out(self, input_features=None) -> List[str]:
        """
        Get output feature names.
        
        Returns:
            List of feature names
        """
        return self.feature_names_
    
    def get_feature_importance(self) -> Dict:
        """
        Get feature importance scores (if available).
        
        Returns:
            Dictionary with feature importance
        """
        # This would be populated after model training
        return {}