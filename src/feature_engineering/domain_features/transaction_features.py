"""
Transaction Features Module
===========================
This module creates features directly from transaction data. These features capture
the essential characteristics of each transaction and form the foundation for
fraud detection.

Key Concepts:
- Amount features: Capture monetary patterns and anomalies
- Location features: Capture geographic patterns and risks
- Merchant features: Capture merchant-specific patterns
- Interaction features: Capture relationships between different transaction attributes

Author: VeritasFinancial Data Science Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from scipy import stats
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

class TransactionFeatureEngineer:
    """
    Master class for creating all transaction-level features.
    
    This class orchestrates the creation of various transaction features by
    delegating to specialized extractors. It ensures all features are created
    consistently and handles any dependencies between features.
    
    Attributes:
        config (dict): Configuration parameters for feature engineering
        amount_extractor (AmountFeatureExtractor): For amount-based features
        location_extractor (LocationFeatureExtractor): For location-based features
        merchant_extractor (MerchantFeatureExtractor): For merchant-based features
        fitted (bool): Whether the engineer has been fitted
        feature_names (list): Names of all generated features
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the TransactionFeatureEngineer.
        
        Args:
            config: Configuration dictionary with parameters like:
                - amount_scaling: Whether to scale amount features
                - location_radius: Radius for location clustering (km)
                - merchant_categories: List of merchant categories to encode
                - interaction_features: Whether to create interaction features
        """
        self.config = config or {}
        self.fitted = False
        self.feature_names = []
        
        # Initialize specialized extractors
        self.amount_extractor = AmountFeatureExtractor(self.config)
        self.location_extractor = LocationFeatureExtractor(self.config)
        self.merchant_extractor = MerchantFeatureExtractor(self.config)
        
        # Track created features
        self.created_features = {}
        
    def fit(self, df: pd.DataFrame) -> 'TransactionFeatureEngineer':
        """
        Fit the feature engineer to the data.
        
        This method learns necessary parameters from the data without transforming it.
        For example, it learns the distribution of amounts for scaling, or the
        typical locations for distance calculations.
        
        Args:
            df: DataFrame containing transaction data
            
        Returns:
            self: The fitted instance
        """
        # Fit each extractor
        self.amount_extractor.fit(df)
        self.location_extractor.fit(df)
        self.merchant_extractor.fit(df)
        
        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by creating all transaction features.
        
        Args:
            df: DataFrame containing transaction data
            
        Returns:
            DataFrame with original columns plus new transaction features
        """
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")
        
        # Create a copy to avoid modifying original
        result_df = df.copy()
        
        # Generate features from each extractor
        result_df = self.amount_extractor.transform(result_df)
        result_df = self.location_extractor.transform(result_df)
        result_df = self.merchant_extractor.transform(result_df)
        
        # Create interaction features
        result_df = self._create_interaction_features(result_df)
        
        # Update feature names
        self.feature_names = [
            col for col in result_df.columns 
            if col not in df.columns or col in self._get_new_feature_patterns()
        ]
        
        return result_df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the engineer and transform the data in one step.
        
        Args:
            df: DataFrame containing transaction data
            
        Returns:
            DataFrame with all transaction features added
        """
        return self.fit(df).transform(df)
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between different transaction attributes.
        
        Interaction features capture relationships between variables that might
        indicate fraud. For example, a high amount from a new location might be
        more suspicious than either feature alone.
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with interaction features added
        """
        # Amount-Location interactions
        if 'amount_scaled' in df.columns and 'distance_from_home' in df.columns:
            # High amount + far from home is suspicious
            df['amount_distance_interaction'] = (
                df['amount_scaled'] * df['distance_from_home']
            )
            
        # Amount-Time interactions
        if 'amount_scaled' in df.columns and 'hour_of_day' in df.columns:
            # Late night high amounts are suspicious
            df['amount_hour_interaction'] = (
                df['amount_scaled'] * 
                np.sin(2 * np.pi * df['hour_of_day'] / 24)  # Cyclical encoding
            )
            
        # Merchant-Amount interactions
        if 'merchant_risk_score' in df.columns and 'amount_anomaly_score' in df.columns:
            # High risk merchant + anomalous amount
            df['merchant_amount_risk'] = (
                df['merchant_risk_score'] * df['amount_anomaly_score']
            )
            
        return df
    
    def _get_new_feature_patterns(self) -> List[str]:
        """
        Get patterns for identifying newly created features.
        
        Returns:
            List of strings that identify new feature columns
        """
        patterns = [
            'amount_', 'location_', 'merchant_', '_distance_',
            '_risk_', '_anomaly_', '_interaction_', '_scaled',
            '_zscore', '_log', '_ratio'
        ]
        return patterns
    
    def get_feature_importance_hints(self) -> Dict[str, float]:
        """
        Provide hints about which features are typically important.
        
        This is based on domain knowledge and previous analysis of fraud patterns.
        These hints can be used for feature selection or as priors for ML models.
        
        Returns:
            Dictionary mapping feature patterns to importance weights
        """
        return {
            'amount_anomaly_score': 0.9,
            'distance_from_home': 0.8,
            'merchant_risk_score': 0.85,
            'amount_zscore': 0.75,
            'location_risk_score': 0.7,
            'amount_distance_interaction': 0.8,
            'unusual_time_flag': 0.6,
            'is_high_value': 0.5
        }


class AmountFeatureExtractor:
    """
    Extract features related to transaction amounts.
    
    Amount-based features are crucial for fraud detection because fraudulent
    transactions often involve unusual amounts. This class creates various
    transformations and statistical measures of transaction amounts.
    
    Features created:
    - amount_log: Log transformation for skewed distributions
    - amount_zscore: Standardized amount based on global distribution
    - amount_scaled: Robust scaling to handle outliers
    - amount_anomaly_score: How anomalous this amount is
    - is_high_value: Binary flag for high-value transactions
    - amount_category: Categorical amount ranges
    - amount_ratio_to_avg: Ratio to customer's average transaction
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the AmountFeatureExtractor.
        
        Args:
            config: Configuration containing:
                - high_value_threshold: Threshold for high-value flag
                - amount_categories: List of amount category boundaries
                - scaling_method: 'robust', 'standard', or 'minmax'
        """
        self.config = config or {}
        self.fitted = False
        self.amount_stats = {}  # Store amount statistics
        self.scaler = RobustScaler()  # Default to robust scaling
        self.high_value_threshold = self.config.get('high_value_threshold', 10000)
        
    def fit(self, df: pd.DataFrame) -> 'AmountFeatureExtractor':
        """
        Fit the amount extractor to the data.
        
        Learns the distribution of amounts for scaling and anomaly detection.
        
        Args:
            df: DataFrame with 'amount' column
        """
        if 'amount' not in df.columns:
            raise ValueError("DataFrame must contain 'amount' column")
            
        # Calculate amount statistics
        amounts = df['amount'].dropna()
        
        self.amount_stats = {
            'mean': amounts.mean(),
            'std': amounts.std(),
            'median': amounts.median(),
            'q1': amounts.quantile(0.25),
            'q3': amounts.quantile(0.75),
            'iqr': amounts.quantile(0.75) - amounts.quantile(0.25),
            'min': amounts.min(),
            'max': amounts.max(),
            'skew': amounts.skew(),
            'kurtosis': amounts.kurtosis()
        }
        
        # Fit the scaler
        self.scaler.fit(amounts.values.reshape(-1, 1))
        
        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create amount-based features.
        
        Args:
            df: DataFrame with 'amount' column
            
        Returns:
            DataFrame with amount features added
        """
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")
            
        result_df = df.copy()
        
        # Basic transformations
        result_df = self._create_basic_transformations(result_df)
        
        # Statistical features
        result_df = self._create_statistical_features(result_df)
        
        # Anomaly scores
        result_df = self._create_anomaly_features(result_df)
        
        # Categorical features
        result_df = self._create_categorical_features(result_df)
        
        return result_df
    
    def _create_basic_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create basic mathematical transformations of amount.
        
        These transformations help handle skewed distributions and make
        the amount feature more suitable for machine learning models.
        
        Args:
            df: DataFrame with amount column
            
        Returns:
            DataFrame with transformed amount features
        """
        # Log transformation (handles positive amounts, add small constant for zero)
        df['amount_log'] = np.log1p(df['amount'])  # log(1 + amount)
        
        # Square root transformation (less extreme than log)
        df['amount_sqrt'] = np.sqrt(df['amount'] + 1e-8)
        
        # Box-Cox transformation (if all values > 0)
        if (df['amount'] > 0).all():
            df['amount_boxcox'], _ = stats.boxcox(df['amount'] + 1)
            
        # Robust scaling (handles outliers better than standard scaling)
        df['amount_scaled'] = self.scaler.transform(df['amount'].values.reshape(-1, 1))
        
        # Standard scaling (z-score)
        df['amount_zscore'] = (
            (df['amount'] - self.amount_stats['mean']) / 
            self.amount_stats['std']
        )
        
        # Robust z-score (using median and IQR)
        df['amount_robust_zscore'] = (
            (df['amount'] - self.amount_stats['median']) / 
            self.amount_stats['iqr']
        )
        
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical features based on amount distribution.
        
        Args:
            df: DataFrame with amount column
            
        Returns:
            DataFrame with statistical features added
        """
        # Percentile-based features
        df['amount_percentile'] = df['amount'].rank(pct=True)
        
        # Flag amounts in extreme percentiles
        df['is_amount_high_percentile'] = (df['amount_percentile'] > 0.95).astype(int)
        df['is_amount_low_percentile'] = (df['amount_percentile'] < 0.05).astype(int)
        
        # Deviation from median
        df['amount_deviation_from_median'] = (
            (df['amount'] - self.amount_stats['median']) / 
            self.amount_stats['median']
        )
        
        # IQR-based outlier detection
        lower_bound = self.amount_stats['q1'] - 1.5 * self.amount_stats['iqr']
        upper_bound = self.amount_stats['q3'] + 1.5 * self.amount_stats['iqr']
        
        df['is_amount_outlier_iqr'] = (
            (df['amount'] < lower_bound) | (df['amount'] > upper_bound)
        ).astype(int)
        
        return df
    
    def _create_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features indicating how anomalous each amount is.
        
        Args:
            df: DataFrame with amount column
            
        Returns:
            DataFrame with anomaly features added
        """
        # Z-score based anomaly score (absolute z-score)
        df['amount_anomaly_score'] = np.abs(df['amount_zscore'])
        
        # Modified Z-score using median (more robust)
        df['amount_robust_anomaly_score'] = np.abs(df['amount_robust_zscore'])
        
        # Exponential of deviation (emphasizes large deviations)
        df['amount_deviation_exponential'] = np.expm1(np.abs(df['amount_zscore']))
        
        # Flag for extremely anomalous amounts
        df['is_highly_anomalous'] = (df['amount_anomaly_score'] > 3).astype(int)
        
        return df
    
    def _create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create categorical features from amount.
        
        Args:
            df: DataFrame with amount column
            
        Returns:
            DataFrame with categorical features added
        """
        # High value flag
        df['is_high_value'] = (
            df['amount'] > self.high_value_threshold
        ).astype(int)
        
        # Very high value flag (for extreme cases)
        df['is_very_high_value'] = (
            df['amount'] > self.high_value_threshold * 10
        ).astype(int)
        
        # Zero or near-zero amount flag
        df['is_zero_amount'] = (df['amount'] < 0.01).astype(int)
        
        # Amount categories based on quantiles
        df['amount_category'] = pd.qcut(
            df['amount'], 
            q=5, 
            labels=['very_low', 'low', 'medium', 'high', 'very_high'],
            duplicates='drop'
        )
        
        # One-hot encode categories (but keep original for flexibility)
        # Note: In practice, you might want to one-hot encode this
        # but we'll keep it as labels for now
        
        return df


class LocationFeatureExtractor:
    """
    Extract features related to transaction locations.
    
    Location-based features are important because fraud often involves
    unusual locations. This class creates features that capture:
    - Distance from typical locations
    - Location risk scores
    - Travel patterns
    - Geographic anomalies
    
    Features created:
    - distance_from_home: How far from customer's home location
    - location_risk_score: Risk score based on location
    - is_foreign_transaction: Whether transaction is in foreign country
    - travel_speed: Implied travel speed between transactions
    - location_cluster: Cluster of similar locations
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the LocationFeatureExtractor.
        
        Args:
            config: Configuration containing:
                - home_location_radius: Radius for home location (km)
                - country_risk_scores: Dict mapping countries to risk scores
                - max_travel_speed: Maximum plausible travel speed (km/h)
                - use_geohash: Whether to use geohashing for clustering
        """
        self.config = config or {}
        self.fitted = False
        self.customer_home_locations = {}  # Store each customer's home location
        self.country_risk_scores = self.config.get('country_risk_scores', {})
        self.max_travel_speed = self.config.get('max_travel_speed', 1000)  # km/h
        
    def fit(self, df: pd.DataFrame) -> 'LocationFeatureExtractor':
        """
        Fit the location extractor to the data.
        
        Learns typical locations for each customer based on their transaction history.
        
        Args:
            df: DataFrame with location data (latitude, longitude, country, customer_id)
        """
        required_cols = ['customer_id', 'latitude', 'longitude', 'country']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"DataFrame must contain '{col}' column")
        
        # Determine home location for each customer
        # Home location is defined as the most frequent location with sufficient transactions
        for customer_id in df['customer_id'].unique():
            customer_df = df[df['customer_id'] == customer_id]
            
            if len(customer_df) >= 5:  # Need at least 5 transactions
                # Group by location and count
                location_counts = customer_df.groupby(
                    ['latitude', 'longitude', 'country']
                ).size().reset_index(name='count')
                
                # Home location is the most frequent
                home = location_counts.loc[location_counts['count'].idxmax()]
                
                self.customer_home_locations[customer_id] = {
                    'latitude': home['latitude'],
                    'longitude': home['longitude'],
                    'country': home['country']
                }
            else:
                # Not enough data, use median location
                self.customer_home_locations[customer_id] = {
                    'latitude': customer_df['latitude'].median(),
                    'longitude': customer_df['longitude'].median(),
                    'country': customer_df['country'].mode()[0] if len(customer_df) > 0 else None
                }
        
        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create location-based features.
        
        Args:
            df: DataFrame with location data
            
        Returns:
            DataFrame with location features added
        """
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")
            
        result_df = df.copy()
        
        # Distance features
        result_df = self._create_distance_features(result_df)
        
        # Risk features
        result_df = self._create_risk_features(result_df)
        
        # Travel features
        result_df = self._create_travel_features(result_df)
        
        # Clustering features
        result_df = self._create_clustering_features(result_df)
        
        return result_df
    
    def _create_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on distances between locations.
        
        Uses the Haversine formula to calculate great-circle distances between
        points on the Earth's surface.
        
        Args:
            df: DataFrame with location data
            
        Returns:
            DataFrame with distance features added
        """
        def haversine_distance(lat1, lon1, lat2, lon2):
            """
            Calculate the great-circle distance between two points on Earth.
            
            Uses the Haversine formula which accounts for Earth's curvature.
            
            Args:
                lat1, lon1: Coordinates of first point
                lat2, lon2: Coordinates of second point
                
            Returns:
                Distance in kilometers
            """
            R = 6371  # Earth's radius in kilometers
            
            # Convert to radians
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            
            # Haversine formula
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            
            return R * c
        
        # Distance from home location
        distances = []
        for idx, row in df.iterrows():
            customer_id = row['customer_id']
            home = self.customer_home_locations.get(customer_id)
            
            if home:
                dist = haversine_distance(
                    row['latitude'], row['longitude'],
                    home['latitude'], home['longitude']
                )
                distances.append(dist)
            else:
                distances.append(np.nan)
        
        df['distance_from_home'] = distances
        
        # Log distance (handles skew)
        df['distance_from_home_log'] = np.log1p(df['distance_from_home'])
        
        # Binary flags for distance thresholds
        df['is_far_from_home'] = (df['distance_from_home'] > 100).astype(int)
        df['is_very_far_from_home'] = (df['distance_from_home'] > 1000).astype(int)
        
        return df
    
    def _create_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create location-based risk scores.
        
        Args:
            df: DataFrame with location data
            
        Returns:
            DataFrame with risk features added
        """
        # Country risk score
        df['country_risk_score'] = df['country'].map(
            self.country_risk_scores
        ).fillna(0.5)  # Default to medium risk
        
        # Foreign transaction flag
        df['is_foreign_transaction'] = (
            df.apply(
                lambda row: row['country'] != self.customer_home_locations.get(
                    row['customer_id'], {'country': None}
                )['country'],
                axis=1
            )
        ).astype(int)
        
        # Location risk score (combines country risk and distance)
        df['location_risk_score'] = (
            df['country_risk_score'] * 
            (1 + np.log1p(df['distance_from_home']) / 10)
        ).clip(0, 1)  # Clip to [0, 1] range
        
        # High-risk location flag
        df['is_high_risk_location'] = (
            (df['country_risk_score'] > 0.7) | 
            (df['is_foreign_transaction'] & (df['distance_from_home'] > 1000))
        ).astype(int)
        
        return df
    
    def _create_travel_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features related to travel patterns.
        
        Calculates implied travel speed between consecutive transactions,
        which can detect impossible travel (e.g., transactions in different
        countries too close together in time).
        
        Args:
            df: DataFrame with location and time data
            
        Returns:
            DataFrame with travel features added
        """
        # Sort by customer and time
        df = df.sort_values(['customer_id', 'transaction_time'])
        
        # Calculate time difference to previous transaction (in hours)
        df['time_since_last_tx'] = df.groupby('customer_id')['transaction_time'].diff().dt.total_seconds() / 3600
        
        # Calculate distance to previous transaction
        prev_lat = df.groupby('customer_id')['latitude'].shift(1)
        prev_lon = df.groupby('customer_id')['longitude'].shift(1)
        
        # Vectorized distance calculation
        def distance_to_prev(row):
            if pd.isna(row['prev_lat']) or pd.isna(row['prev_lon']):
                return np.nan
            return haversine_distance(
                row['latitude'], row['longitude'],
                row['prev_lat'], row['prev_lon']
            )
        
        df['prev_lat'] = prev_lat
        df['prev_lon'] = prev_lon
        df['distance_to_prev'] = df.apply(distance_to_prev, axis=1)
        
        # Calculate implied travel speed (km/h)
        df['travel_speed'] = df['distance_to_prev'] / df['time_since_last_tx']
        
        # Flag impossible travel (faster than physically possible)
        df['is_impossible_travel'] = (
            (df['travel_speed'] > self.max_travel_speed) & 
            (df['distance_to_prev'] > 100)  # Only flag for significant distances
        ).astype(int)
        
        # Clean up temporary columns
        df = df.drop(['prev_lat', 'prev_lon'], axis=1)
        
        return df
    
    def _create_clustering_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on location clustering.
        
        Groups locations into clusters to identify regions with similar
        fraud patterns.
        
        Args:
            df: DataFrame with location data
            
        Returns:
            DataFrame with clustering features added
        """
        from sklearn.cluster import DBSCAN
        
        # Use DBSCAN for clustering based on geographic coordinates
        # Note: This requires scaling coordinates appropriately
        coords = df[['latitude', 'longitude']].values
        
        # DBSCAN with epsilon in kilometers (convert to radians for haversine)
        eps_km = 50  # 50km radius for clusters
        eps_rad = eps_km / 6371  # Convert to radians
        
        clustering = DBSCAN(eps=eps_rad, min_samples=5, metric='haversine')
        df['location_cluster'] = clustering.fit_predict(np.radians(coords))
        
        # Cluster size
        cluster_sizes = df['location_cluster'].value_counts()
        df['location_cluster_size'] = df['location_cluster'].map(cluster_sizes)
        
        # Flag for isolated locations (not in any cluster)
        df['is_isolated_location'] = (df['location_cluster'] == -1).astype(int)
        
        return df


class MerchantFeatureExtractor:
    """
    Extract features related to merchants.
    
    Merchant-based features capture patterns associated with specific merchants
    or merchant categories. Certain merchants may have higher fraud rates or
    different risk profiles.
    
    Features created:
    - merchant_risk_score: Risk score for the merchant
    - merchant_category_risk: Risk score for the merchant category
    - merchant_transaction_velocity: How many transactions at this merchant
    - merchant_avg_amount: Average transaction amount at this merchant
    - is_high_risk_merchant: Binary flag for high-risk merchants
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the MerchantFeatureExtractor.
        
        Args:
            config: Configuration containing:
                - merchant_risk_scores: Dict mapping merchants to risk scores
                - category_risk_scores: Dict mapping categories to risk scores
                - high_risk_threshold: Threshold for high-risk classification
        """
        self.config = config or {}
        self.fitted = False
        self.merchant_stats = {}  # Statistics per merchant
        self.category_stats = {}  # Statistics per merchant category
        
    def fit(self, df: pd.DataFrame) -> 'MerchantFeatureExtractor':
        """
        Fit the merchant extractor to the data.
        
        Learns statistics for each merchant and merchant category.
        
        Args:
            df: DataFrame with merchant data
        """
        required_cols = ['merchant_id', 'merchant_category', 'amount']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"DataFrame must contain '{col}' column")
        
        # Calculate merchant statistics
        merchant_groups = df.groupby('merchant_id')
        self.merchant_stats = {
            merchant_id: {
                'count': len(group),
                'avg_amount': group['amount'].mean(),
                'std_amount': group['amount'].std(),
                'min_amount': group['amount'].min(),
                'max_amount': group['amount'].max(),
                'fraud_rate': group['is_fraud'].mean() if 'is_fraud' in group.columns else None,
                'categories': group['merchant_category'].mode().tolist() if len(group) > 0 else []
            }
            for merchant_id, group in merchant_groups
        }
        
        # Calculate category statistics
        category_groups = df.groupby('merchant_category')
        self.category_stats = {
            category: {
                'count': len(group),
                'avg_amount': group['amount'].mean(),
                'std_amount': group['amount'].std(),
                'fraud_rate': group['is_fraud'].mean() if 'is_fraud' in group.columns else None,
                'unique_merchants': group['merchant_id'].nunique()
            }
            for category, group in category_groups
        }
        
        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create merchant-based features.
        
        Args:
            df: DataFrame with merchant data
            
        Returns:
            DataFrame with merchant features added
        """
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")
            
        result_df = df.copy()
        
        # Basic merchant features
        result_df = self._create_basic_merchant_features(result_df)
        
        # Merchant risk features
        result_df = self._create_merchant_risk_features(result_df)
        
        # Category-based features
        result_df = self._create_category_features(result_df)
        
        # Velocity features
        result_df = self._create_merchant_velocity_features(result_df)
        
        return result_df
    
    def _create_basic_merchant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create basic merchant-level features.
        
        Args:
            df: DataFrame with merchant data
            
        Returns:
            DataFrame with basic merchant features added
        """
        # Merchant transaction count (how many times this merchant appears)
        merchant_counts = df['merchant_id'].value_counts()
        df['merchant_transaction_count'] = df['merchant_id'].map(merchant_counts)
        
        # Merchant transaction count in last 24h (rolling window)
        # This is a simplified version - in production, use proper rolling windows
        df['merchant_24h_count'] = df.groupby('merchant_id').cumcount() + 1
        
        # Merchant average amount from learned stats
        df['merchant_avg_amount'] = df['merchant_id'].map(
            lambda x: self.merchant_stats.get(x, {}).get('avg_amount', np.nan)
        )
        
        # Deviation from merchant's average amount
        df['amount_deviation_from_merchant_avg'] = (
            (df['amount'] - df['merchant_avg_amount']) / 
            df['merchant_avg_amount']
        )
        
        return df
    
    def _create_merchant_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create merchant risk features.
        
        Args:
            df: DataFrame with merchant data
            
        Returns:
            DataFrame with merchant risk features added
        """
        # Historical fraud rate for merchant
        df['merchant_historical_fraud_rate'] = df['merchant_id'].map(
            lambda x: self.merchant_stats.get(x, {}).get('fraud_rate', 0.01)
        )
        
        # Risk score based on fraud rate (smoothed)
        df['merchant_risk_score'] = (
            df['merchant_historical_fraud_rate'] * 5  # Scale to 0-5 range
        ).clip(0, 1)  # Clip to 0-1 for probability-like score
        
        # High-risk merchant flag
        high_risk_threshold = self.config.get('high_risk_threshold', 0.3)
        df['is_high_risk_merchant'] = (
            df['merchant_risk_score'] > high_risk_threshold
        ).astype(int)
        
        # New merchant flag (few transactions)
        df['is_new_merchant'] = (
            df['merchant_transaction_count'] < 10
        ).astype(int)
        
        return df
    
    def _create_category_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create merchant category-based features.
        
        Args:
            df: DataFrame with merchant data
            
        Returns:
            DataFrame with category features added
        """
        # Category fraud rate
        df['category_fraud_rate'] = df['merchant_category'].map(
            lambda x: self.category_stats.get(x, {}).get('fraud_rate', 0.01)
        )
        
        # Category risk score
        df['category_risk_score'] = (
            df['category_fraud_rate'] * 5
        ).clip(0, 1)
        
        # Category average amount
        df['category_avg_amount'] = df['merchant_category'].map(
            lambda x: self.category_stats.get(x, {}).get('avg_amount', np.nan)
        )
        
        # Deviation from category average
        df['amount_deviation_from_category_avg'] = (
            (df['amount'] - df['category_avg_amount']) / 
            df['category_avg_amount']
        )
        
        # Category popularity (number of unique merchants)
        df['category_unique_merchants'] = df['merchant_category'].map(
            lambda x: self.category_stats.get(x, {}).get('unique_merchants', 1)
        )
        
        return df
    
    def _create_merchant_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create merchant velocity features.
        
        Measures how quickly transactions are occurring at the same merchant,
        which can indicate systematic fraud.
        
        Args:
            df: DataFrame with merchant data
            
        Returns:
            DataFrame with merchant velocity features added
        """
        # Sort for rolling calculations
        df = df.sort_values(['merchant_id', 'transaction_time'])
        
        # Time since last transaction at same merchant
        df['time_since_last_merchant_tx'] = (
            df.groupby('merchant_id')['transaction_time']
            .diff()
            .dt.total_seconds() / 3600  # Convert to hours
        )
        
        # Rapid consecutive transactions at same merchant
        df['is_rapid_same_merchant'] = (
            df['time_since_last_merchant_tx'] < 0.1  # Less than 6 minutes
        ).astype(int)
        
        # Merchant transaction intensity (transactions per hour)
        # Using rolling window of 1 hour
        df['merchant_tx_per_hour'] = df.groupby('merchant_id').rolling(
            '1H', on='transaction_time'
        )['amount'].count().reset_index(0, drop=True)
        
        # Merchant amount velocity (total amount per hour)
        df['merchant_amount_per_hour'] = df.groupby('merchant_id').rolling(
            '1H', on='transaction_time'
        )['amount'].sum().reset_index(0, drop=True)
        
        return df