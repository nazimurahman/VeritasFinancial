"""
Behavioral Features Module
=========================
This module creates features that capture customer behavioral patterns.
Behavioral analysis is crucial for fraud detection because fraud often
manifests as deviations from normal behavior patterns.

Key Concepts:
- Spending patterns: How customers typically spend
- Temporal patterns: When customers typically transact
- Geographic patterns: Where customers typically transact
- Velocity patterns: How quickly customers transact
- Deviation detection: Identifying anomalous behavior

Author: VeritasFinancial Data Science Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class BehavioralFeatureEngineer:
    """
    Master class for creating all behavioral features.
    
    This class orchestrates the creation of features that capture
    customer behavioral patterns and deviations.
    
    Attributes:
        config (dict): Configuration parameters
        spending_analyzer (SpendingPatternAnalyzer)
        deviation_detector (DeviationDetector)
        velocity_calculator (VelocityCalculator)
        fitted (bool): Whether the engineer has been fitted
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the BehavioralFeatureEngineer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.fitted = False
        
        # Initialize specialized analyzers
        self.spending_analyzer = SpendingPatternAnalyzer(self.config)
        self.deviation_detector = DeviationDetector(self.config)
        self.velocity_calculator = VelocityCalculator(self.config)
        
        self.feature_names = []
        self.customer_profiles = {}  # Store customer behavioral profiles
        
    def fit(self, df: pd.DataFrame) -> 'BehavioralFeatureEngineer':
        """
        Fit the behavioral engineer to the data.
        
        This builds behavioral profiles for each customer based on
        their historical transaction patterns.
        
        Args:
            df: DataFrame with transaction history
            
        Returns:
            self: The fitted instance
        """
        # Fit each component
        self.spending_analyzer.fit(df)
        self.deviation_detector.fit(df)
        self.velocity_calculator.fit(df)
        
        # Build customer profiles
        self._build_customer_profiles(df)
        
        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by creating all behavioral features.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with behavioral features added
        """
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")
        
        result_df = df.copy()
        
        # Generate features from each component
        result_df = self.spending_analyzer.transform(result_df)
        result_df = self.deviation_detector.transform(result_df)
        result_df = self.velocity_calculator.transform(result_df)
        
        # Add profile-based features
        result_df = self._add_profile_features(result_df)
        
        # Update feature names
        self.feature_names = [
            col for col in result_df.columns 
            if col not in df.columns or col.startswith(('behavioral_', 'spending_', 'deviation_'))
        ]
        
        return result_df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with behavioral features added
        """
        return self.fit(df).transform(df)
    
    def _build_customer_profiles(self, df: pd.DataFrame) -> None:
        """
        Build behavioral profiles for each customer.
        
        A profile captures the typical behavior of a customer across
        multiple dimensions.
        
        Args:
            df: DataFrame with transaction history
        """
        if 'customer_id' not in df.columns:
            return
        
        for customer_id in df['customer_id'].unique():
            customer_df = df[df['customer_id'] == customer_id]
            
            if len(customer_df) < 5:  # Need minimum data for profile
                continue
            
            profile = {
                'customer_id': customer_id,
                'avg_transaction_amount': customer_df['amount'].mean(),
                'std_transaction_amount': customer_df['amount'].std(),
                'median_transaction_amount': customer_df['amount'].median(),
                'preferred_hours': self._get_preferred_hours(customer_df),
                'preferred_days': self._get_preferred_days(customer_df),
                'preferred_merchant_categories': self._get_preferred_categories(customer_df),
                'typical_locations': self._get_typical_locations(customer_df),
                'transaction_frequency': len(customer_df) / 30,  # Approx per day
                'amount_distribution': self._get_amount_distribution(customer_df['amount'])
            }
            
            self.customer_profiles[customer_id] = profile
    
    def _get_preferred_hours(self, df: pd.DataFrame) -> List[int]:
        """Get hours when customer typically transacts."""
        if 'transaction_time' not in df.columns:
            return []
        
        hours = pd.to_datetime(df['transaction_time']).dt.hour
        # Hours that appear more than once
        hour_counts = hours.value_counts()
        preferred = hour_counts[hour_counts > 1].index.tolist()
        return preferred
    
    def _get_preferred_days(self, df: pd.DataFrame) -> List[int]:
        """Get days of week when customer typically transacts."""
        if 'transaction_time' not in df.columns:
            return []
        
        days = pd.to_datetime(df['transaction_time']).dt.dayofweek
        day_counts = days.value_counts()
        preferred = day_counts[day_counts > 1].index.tolist()
        return preferred
    
    def _get_preferred_categories(self, df: pd.DataFrame) -> List[str]:
        """Get merchant categories customer prefers."""
        if 'merchant_category' not in df.columns:
            return []
        
        category_counts = df['merchant_category'].value_counts()
        # Categories that appear more than once
        preferred = category_counts[category_counts > 1].index.tolist()[:3]  # Top 3
        return preferred
    
    def _get_typical_locations(self, df: pd.DataFrame) -> List[Tuple[float, float]]:
        """Get typical locations for customer."""
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            return []
        
        # Cluster locations to find typical ones
        from sklearn.cluster import DBSCAN
        
        coords = df[['latitude', 'longitude']].values
        if len(coords) < 3:
            return []
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=0.1, min_samples=2).fit(coords)
        labels = clustering.labels_
        
        # Get centroids of clusters (excluding noise -1)
        typical = []
        for label in set(labels):
            if label != -1:
                cluster_points = coords[labels == label]
                centroid = cluster_points.mean(axis=0)
                typical.append(tuple(centroid))
        
        return typical
    
    def _get_amount_distribution(self, amounts: pd.Series) -> Dict:
        """Get statistical distribution of transaction amounts."""
        return {
            'percentiles': {
                '10': amounts.quantile(0.1),
                '25': amounts.quantile(0.25),
                '50': amounts.quantile(0.5),
                '75': amounts.quantile(0.75),
                '90': amounts.quantile(0.9)
            },
            'skew': amounts.skew(),
            'kurtosis': amounts.kurtosis()
        }
    
    def _add_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features based on customer profiles.
        
        These features compare current transaction to customer's profile.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with profile-based features added
        """
        if 'customer_id' not in df.columns:
            return df
        
        # Amount deviation from profile
        df['amount_deviation_from_profile'] = df.apply(
            lambda row: self._amount_deviation(row), axis=1
        )
        
        # Time pattern match
        df['time_pattern_match'] = df.apply(
            lambda row: self._time_pattern_match(row), axis=1
        )
        
        # Location pattern match
        df['location_pattern_match'] = df.apply(
            lambda row: self._location_pattern_match(row), axis=1
        )
        
        # Category pattern match
        df['category_pattern_match'] = df.apply(
            lambda row: self._category_pattern_match(row), axis=1
        )
        
        # Overall behavioral anomaly score
        df['behavioral_anomaly_score'] = (
            (1 - df['time_pattern_match']) * 0.3 +
            (1 - df['location_pattern_match']) * 0.4 +
            df['amount_deviation_from_profile'] * 0.3
        )
        
        return df
    
    def _amount_deviation(self, row) -> float:
        """Calculate how much amount deviates from customer profile."""
        customer_id = row.get('customer_id')
        amount = row.get('amount', 0)
        
        profile = self.customer_profiles.get(customer_id)
        if not profile:
            return 0
        
        avg = profile['avg_transaction_amount']
        std = profile['std_transaction_amount'] or 1
        
        # Z-score based deviation
        deviation = abs(amount - avg) / std
        return min(deviation, 5) / 5  # Normalize to 0-1
    
    def _time_pattern_match(self, row) -> float:
        """Calculate how well transaction time matches customer profile."""
        customer_id = row.get('customer_id')
        
        profile = self.customer_profiles.get(customer_id)
        if not profile:
            return 1  # Default to match
        
        if 'transaction_time' not in row:
            return 1
        
        hour = pd.to_datetime(row['transaction_time']).hour
        day = pd.to_datetime(row['transaction_time']).dayofweek
        
        # Check if hour is preferred
        hour_match = hour in profile.get('preferred_hours', [])
        day_match = day in profile.get('preferred_days', [])
        
        if hour_match and day_match:
            return 1.0
        elif hour_match or day_match:
            return 0.5
        else:
            return 0.0
    
    def _location_pattern_match(self, row) -> float:
        """Calculate how well location matches customer profile."""
        customer_id = row.get('customer_id')
        
        profile = self.customer_profiles.get(customer_id)
        if not profile:
            return 1
        
        if 'latitude' not in row or 'longitude' not in row:
            return 1
        
        # Calculate distance to nearest typical location
        from haversine import haversine
        
        current_loc = (row['latitude'], row['longitude'])
        typical_locs = profile.get('typical_locations', [])
        
        if not typical_locs:
            return 1
        
        # Find minimum distance to any typical location
        min_distance = min(
            haversine(current_loc, loc) for loc in typical_locs
        )
        
        # Convert distance to match score (closer = higher score)
        if min_distance < 10:  # Within 10km
            return 1.0
        elif min_distance < 50:  # Within 50km
            return 0.7
        elif min_distance < 100:  # Within 100km
            return 0.4
        else:
            return 0.1
    
    def _category_pattern_match(self, row) -> float:
        """Calculate how well merchant category matches customer profile."""
        customer_id = row.get('customer_id')
        
        profile = self.customer_profiles.get(customer_id)
        if not profile:
            return 1
        
        category = row.get('merchant_category')
        preferred = profile.get('preferred_merchant_categories', [])
        
        if category in preferred:
            return 1.0
        elif preferred and category in self._get_related_categories(category, preferred):
            return 0.5
        else:
            return 0.0
    
    def _get_related_categories(self, category: str, preferred: List[str]) -> List[str]:
        """Get categories related to the preferred ones."""
        # Simplified mapping - in production, use proper category hierarchy
        category_groups = {
            'grocery': ['supermarket', 'convenience_store'],
            'restaurant': ['fast_food', 'cafe', 'bar'],
            'retail': ['clothing', 'electronics', 'home_goods'],
            'travel': ['airline', 'hotel', 'car_rental'],
            'entertainment': ['movie', 'concert', 'sports']
        }
        
        related = []
        for pref in preferred:
            for group, members in category_groups.items():
                if pref in members:
                    related.extend(members)
        
        return list(set(related))


class SpendingPatternAnalyzer:
    """
    Analyze customer spending patterns.
    
    This class creates features that capture how customers typically spend,
    including their preferred categories, amounts, and frequencies.
    
    Features created:
    - spending_by_category: Amount spent in each category
    - spending_regularity: How regular spending patterns are
    - spending_trend: Trend in spending over time
    - spending_seasonality: Seasonal patterns in spending
    - category_diversity: Diversity of categories used
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the SpendingPatternAnalyzer."""
        self.config = config or {}
        self.fitted = False
        self.customer_patterns = {}
        
    def fit(self, df: pd.DataFrame) -> 'SpendingPatternAnalyzer':
        """Fit the analyzer to historical data."""
        if 'customer_id' in df.columns and 'amount' in df.columns:
            # Calculate spending patterns per customer
            for customer_id in df['customer_id'].unique():
                customer_df = df[df['customer_id'] == customer_id]
                
                if len(customer_df) < 5:
                    continue
                
                pattern = {
                    'avg_daily_spend': self._calculate_avg_daily_spend(customer_df),
                    'spending_variance': customer_df['amount'].var(),
                    'preferred_categories': self._get_category_preferences(customer_df),
                    'spending_regularity': self._calculate_regularity(customer_df),
                    'spending_trend': self._calculate_trend(customer_df)
                }
                
                self.customer_patterns[customer_id] = pattern
        
        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create spending pattern features."""
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")
        
        result_df = df.copy()
        
        # Add spending pattern features
        if 'customer_id' in result_df.columns:
            result_df['spending_regularity'] = result_df['customer_id'].map(
                lambda x: self.customer_patterns.get(x, {}).get('spending_regularity', 0.5)
            )
            
            result_df['spending_trend'] = result_df['customer_id'].map(
                lambda x: self.customer_patterns.get(x, {}).get('spending_trend', 0)
            )
            
            result_df['category_diversity'] = result_df.apply(
                lambda row: self._category_diversity_score(row), axis=1
            )
        
        return result_df
    
    def _calculate_avg_daily_spend(self, df: pd.DataFrame) -> float:
        """Calculate average daily spending."""
        if 'transaction_time' not in df.columns:
            return df['amount'].mean()
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['transaction_time']).dt.date
        daily_spend = df.groupby('date')['amount'].sum()
        return daily_spend.mean()
    
    def _get_category_preferences(self, df: pd.DataFrame) -> Dict:
        """Get spending preferences by category."""
        if 'merchant_category' not in df.columns:
            return {}
        
        category_spend = df.groupby('merchant_category')['amount'].agg(['sum', 'mean', 'count'])
        return category_spend.to_dict('index')
    
    def _calculate_regularity(self, df: pd.DataFrame) -> float:
        """Calculate how regular spending patterns are."""
        if 'transaction_time' not in df.columns or len(df) < 5:
            return 0.5
        
        # Sort by time
        df = df.sort_values('transaction_time')
        
        # Calculate time between transactions
        time_diffs = df['transaction_time'].diff().dt.total_seconds().dropna()
        
        if len(time_diffs) < 2:
            return 0.5
        
        # Coefficient of variation of time differences
        # Lower CV = more regular
        cv = time_diffs.std() / (time_diffs.mean() + 1)
        
        # Convert to 0-1 scale where 1 = very regular
        regularity = 1 / (1 + cv)
        return regularity
    
    def _calculate_trend(self, df: pd.DataFrame) -> float:
        """Calculate trend in spending over time."""
        if 'transaction_time' not in df.columns or len(df) < 5:
            return 0
        
        # Sort by time
        df = df.sort_values('transaction_time')
        
        # Create time index
        time_idx = np.arange(len(df))
        
        # Linear regression of amount over time
        slope, _, _, _, _ = stats.linregress(time_idx, df['amount'].values)
        
        # Normalize slope
        max_slope = df['amount'].std() / len(df) * 10
        normalized_slope = np.tanh(slope / (max_slope + 1))
        
        return normalized_slope
    
    def _category_diversity_score(self, row) -> float:
        """Calculate diversity of categories used by customer."""
        customer_id = row.get('customer_id')
        
        pattern = self.customer_patterns.get(customer_id, {})
        preferences = pattern.get('preferred_categories', {})
        
        if not preferences:
            return 0.5
        
        # Number of categories used
        num_categories = len(preferences)
        
        # Normalize to 0-1 (assuming max 20 categories)
        diversity = min(num_categories / 20, 1)
        
        return diversity


class DeviationDetector:
    """
    Detect deviations from normal behavior.
    
    This class identifies when a transaction deviates from a customer's
    normal patterns, which is a key indicator of potential fraud.
    
    Features created:
    - amount_deviation: Deviation from normal transaction amount
    - time_deviation: Deviation from normal transaction time
    - location_deviation: Deviation from normal location
    - category_deviation: Deviation from normal merchant categories
    - velocity_deviation: Deviation from normal transaction velocity
    - combined_deviation_score: Overall deviation score
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the DeviationDetector."""
        self.config = config or {}
        self.fitted = False
        self.baselines = {}
        
    def fit(self, df: pd.DataFrame) -> 'DeviationDetector':
        """Fit the detector by establishing baselines."""
        if 'customer_id' in df.columns:
            for customer_id in df['customer_id'].unique():
                customer_df = df[df['customer_id'] == customer_id]
                
                if len(customer_df) < 5:
                    continue
                
                baseline = {
                    'amount_mean': customer_df['amount'].mean(),
                    'amount_std': customer_df['amount'].std(),
                    'amount_percentiles': self._calculate_percentiles(customer_df['amount']),
                    'hour_distribution': self._calculate_hour_distribution(customer_df),
                    'day_distribution': self._calculate_day_distribution(customer_df),
                    'location_centroid': self._calculate_location_centroid(customer_df),
                    'category_distribution': self._calculate_category_distribution(customer_df),
                    'velocity_baseline': self._calculate_velocity_baseline(customer_df)
                }
                
                self.baselines[customer_id] = baseline
        
        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create deviation features."""
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")
        
        result_df = df.copy()
        
        # Calculate deviations for each transaction
        if 'customer_id' in result_df.columns:
            result_df['amount_deviation'] = result_df.apply(
                self._amount_deviation, axis=1
            )
            
            result_df['time_deviation'] = result_df.apply(
                self._time_deviation, axis=1
            )
            
            result_df['location_deviation'] = result_df.apply(
                self._location_deviation, axis=1
            )
            
            result_df['category_deviation'] = result_df.apply(
                self._category_deviation, axis=1
            )
            
            result_df['velocity_deviation'] = result_df.apply(
                self._velocity_deviation, axis=1
            )
            
            # Combined deviation score
            result_df['combined_deviation_score'] = (
                result_df['amount_deviation'] * 0.3 +
                result_df['time_deviation'] * 0.2 +
                result_df['location_deviation'] * 0.3 +
                result_df['category_deviation'] * 0.1 +
                result_df['velocity_deviation'] * 0.1
            )
            
            # Flag for high deviation
            result_df['is_highly_deviant'] = (
                result_df['combined_deviation_score'] > 0.7
            ).astype(int)
        
        return result_df
    
    def _calculate_percentiles(self, amounts: pd.Series) -> Dict:
        """Calculate amount percentiles."""
        return {
            '10': amounts.quantile(0.1),
            '25': amounts.quantile(0.25),
            '50': amounts.quantile(0.5),
            '75': amounts.quantile(0.75),
            '90': amounts.quantile(0.9)
        }
    
    def _calculate_hour_distribution(self, df: pd.DataFrame) -> Dict:
        """Calculate distribution of transaction hours."""
        if 'transaction_time' not in df.columns:
            return {}
        
        hours = pd.to_datetime(df['transaction_time']).dt.hour
        distribution = hours.value_counts(normalize=True).to_dict()
        return distribution
    
    def _calculate_day_distribution(self, df: pd.DataFrame) -> Dict:
        """Calculate distribution of transaction days."""
        if 'transaction_time' not in df.columns:
            return {}
        
        days = pd.to_datetime(df['transaction_time']).dt.dayofweek
        distribution = days.value_counts(normalize=True).to_dict()
        return distribution
    
    def _calculate_location_centroid(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate centroid of transaction locations."""
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            return (0, 0)
        
        return (df['latitude'].mean(), df['longitude'].mean())
    
    def _calculate_category_distribution(self, df: pd.DataFrame) -> Dict:
        """Calculate distribution of merchant categories."""
        if 'merchant_category' not in df.columns:
            return {}
        
        distribution = df['merchant_category'].value_counts(normalize=True).to_dict()
        return distribution
    
    def _calculate_velocity_baseline(self, df: pd.DataFrame) -> float:
        """Calculate baseline transaction velocity."""
        if 'transaction_time' not in df.columns or len(df) < 5:
            return 1.0
        
        df = df.sort_values('transaction_time')
        time_diffs = df['transaction_time'].diff().dt.total_seconds().dropna()
        
        # Average time between transactions in hours
        avg_gap = time_diffs.mean() / 3600 if len(time_diffs) > 0 else 24
        
        return avg_gap
    
    def _amount_deviation(self, row) -> float:
        """Calculate amount deviation score."""
        customer_id = row.get('customer_id')
        amount = row.get('amount', 0)
        
        baseline = self.baselines.get(customer_id)
        if not baseline:
            return 0
        
        mean = baseline['amount_mean']
        std = baseline['amount_std'] or mean * 0.1  # Handle zero std
        
        # Z-score based deviation
        z_score = abs(amount - mean) / std
        
        # Convert to 0-1 score
        deviation = min(z_score / 3, 1)  # 3 sigma = 1.0
        
        return deviation
    
    def _time_deviation(self, row) -> float:
        """Calculate time deviation score."""
        customer_id = row.get('customer_id')
        
        baseline = self.baselines.get(customer_id)
        if not baseline or 'transaction_time' not in row:
            return 0
        
        hour = pd.to_datetime(row['transaction_time']).hour
        day = pd.to_datetime(row['transaction_time']).dayofweek
        
        hour_dist = baseline.get('hour_distribution', {})
        day_dist = baseline.get('day_distribution', {})
        
        # Get probability of this hour/day in baseline
        hour_prob = hour_dist.get(hour, 0)
        day_prob = day_dist.get(day, 0)
        
        # Combined probability
        prob = hour_prob * day_prob
        
        # Convert to deviation (lower probability = higher deviation)
        if prob > 0:
            deviation = 1 - min(prob * 24, 1)  # Scale by 24 hours
        else:
            deviation = 1.0
        
        return deviation
    
    def _location_deviation(self, row) -> float:
        """Calculate location deviation score."""
        customer_id = row.get('customer_id')
        
        baseline = self.baselines.get(customer_id)
        if not baseline or 'latitude' not in row or 'longitude' not in row:
            return 0
        
        from haversine import haversine
        
        centroid = baseline.get('location_centroid', (0, 0))
        current = (row['latitude'], row['longitude'])
        
        # Calculate distance from centroid
        distance = haversine(centroid, current)
        
        # Convert to deviation score
        # Assume 100km is maximum reasonable deviation
        deviation = min(distance / 100, 1)
        
        return deviation
    
    def _category_deviation(self, row) -> float:
        """Calculate category deviation score."""
        customer_id = row.get('customer_id')
        category = row.get('merchant_category')
        
        baseline = self.baselines.get(customer_id)
        if not baseline or not category:
            return 0
        
        cat_dist = baseline.get('category_distribution', {})
        
        # Probability of this category
        prob = cat_dist.get(category, 0)
        
        # Convert to deviation
        if prob > 0:
            deviation = 1 - min(prob * 10, 1)  # Scale by number of categories
        else:
            deviation = 1.0
        
        return deviation
    
    def _velocity_deviation(self, row) -> float:
        """Calculate velocity deviation score."""
        customer_id = row.get('customer_id')
        
        baseline = self.baselines.get(customer_id)
        if not baseline:
            return 0
        
        baseline_velocity = baseline.get('velocity_baseline', 24)  # hours
        
        # Current velocity (time since last transaction)
        # This would need to be calculated with access to transaction history
        # Simplified version - assume we have this feature
        current_gap = row.get('time_since_last_tx', baseline_velocity)
        
        # Calculate deviation ratio
        if current_gap < baseline_velocity * 0.1:  # Much faster than usual
            deviation = 1.0
        elif current_gap < baseline_velocity * 0.5:  # Somewhat faster
            deviation = 0.5
        else:
            deviation = 0.0
        
        return deviation


class VelocityCalculator:
    """
    Calculate transaction velocity features.
    
    Velocity measures how quickly transactions occur, which is crucial
    for detecting automated fraud or unusual activity.
    
    Features created:
    - tx_per_hour: Transactions per hour
    - tx_per_day: Transactions per day
    - amount_per_hour: Total amount per hour
    - amount_per_day: Total amount per day
    - velocity_changes: Changes in velocity over time
    - velocity_outliers: Outlier velocity events
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the VelocityCalculator."""
        self.config = config or {}
        self.fitted = False
        self.windows = self.config.get('velocity_windows', ['1H', '24H', '7D'])
        
    def fit(self, df: pd.DataFrame) -> 'VelocityCalculator':
        """Fit the velocity calculator."""
        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create velocity features."""
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")
        
        result_df = df.copy()
        
        # Calculate velocity features per customer
        if 'customer_id' in result_df.columns and 'transaction_time' in result_df.columns:
            result_df = self._calculate_customer_velocity(result_df)
        
        # Calculate velocity features per device
        if 'device_id' in result_df.columns and 'transaction_time' in result_df.columns:
            result_df = self._calculate_device_velocity(result_df)
        
        # Calculate global velocity features
        result_df = self._calculate_global_velocity(result_df)
        
        return result_df
    
    def _calculate_customer_velocity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate customer-level velocity features."""
        df = df.sort_values(['customer_id', 'transaction_time'])
        
        for window in self.windows:
            # Transaction count velocity
            count_col = f'customer_tx_{window}'
            df[count_col] = df.groupby('customer_id').rolling(
                window, on='transaction_time'
            )['transaction_time'].count().reset_index(0, drop=True)
            
            # Amount velocity
            amount_col = f'customer_amount_{window}'
            df[amount_col] = df.groupby('customer_id').rolling(
                window, on='transaction_time'
            )['amount'].sum().reset_index(0, drop=True)
        
        # Detect velocity changes
        if 'customer_tx_1H' in df.columns and 'customer_tx_24H' in df.columns:
            df['customer_velocity_ratio'] = (
                df['customer_tx_1H'] * 24 / (df['customer_tx_24H'] + 1)
            )
            
            df['customer_velocity_increasing'] = (
                df.groupby('customer_id')['customer_tx_1H']
                .diff()
                .fillna(0) > 0
            ).astype(int)
        
        return df
    
    def _calculate_device_velocity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate device-level velocity features."""
        df = df.sort_values(['device_id', 'transaction_time'])
        
        for window in self.windows:
            # Transaction count velocity
            count_col = f'device_tx_{window}'
            df[count_col] = df.groupby('device_id').rolling(
                window, on='transaction_time'
            )['transaction_time'].count().reset_index(0, drop=True)
            
            # Amount velocity
            amount_col = f'device_amount_{window}'
            df[amount_col] = df.groupby('device_id').rolling(
                window, on='transaction_time'
            )['amount'].sum().reset_index(0, drop=True)
        
        return df
    
    def _calculate_global_velocity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate global velocity features."""
        # Overall system velocity
        if 'transaction_time' in df.columns:
            df = df.sort_values('transaction_time')
            
            # System-wide transaction rate
            df['system_tx_per_minute'] = df.rolling(
                '5T', on='transaction_time'
            )['transaction_time'].count()
        
        return df