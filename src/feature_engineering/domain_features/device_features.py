"""
Device Features Module
=====================
This module creates features related to devices used for transactions.
Device fingerprinting and behavioral analysis are crucial for detecting
fraud, especially in card-not-present scenarios.

Key Concepts:
- Device fingerprinting: Unique device identification
- Device history: Past transactions from this device
- Device risk: Risk score based on device characteristics
- Behavioral patterns: How the device is typically used

Author: VeritasFinancial Data Science Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import hashlib
import json
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class DeviceFeatureEngineer:
    """
    Master class for creating all device-related features.
    
    This class orchestrates the creation of features related to devices
    used in transactions, including fingerprinting and behavioral analysis.
    
    Attributes:
        config (dict): Configuration parameters
        fingerprint_extractor (FingerprintFeatureExtractor)
        behavioral_extractor (DeviceBehavioralFeatureExtractor)
        fitted (bool): Whether the engineer has been fitted
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the DeviceFeatureEngineer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.fitted = False
        
        # Initialize specialized extractors
        self.fingerprint_extractor = FingerprintFeatureExtractor(self.config)
        self.behavioral_extractor = DeviceBehavioralFeatureExtractor(self.config)
        
        self.feature_names = []
        
    def fit(self, df: pd.DataFrame) -> 'DeviceFeatureEngineer':
        """
        Fit the device feature engineer to the data.
        
        Args:
            df: DataFrame containing device data
            
        Returns:
            self: The fitted instance
        """
        # Fit each extractor
        self.fingerprint_extractor.fit(df)
        self.behavioral_extractor.fit(df)
        
        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by creating all device features.
        
        Args:
            df: DataFrame containing device data
            
        Returns:
            DataFrame with device features added
        """
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")
        
        result_df = df.copy()
        
        # Generate features from each extractor
        result_df = self.fingerprint_extractor.transform(result_df)
        result_df = self.behavioral_extractor.transform(result_df)
        
        # Update feature names
        self.feature_names = [
            col for col in result_df.columns 
            if col not in df.columns or col.startswith(('device_', 'fingerprint_'))
        ]
        
        return result_df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: DataFrame containing device data
            
        Returns:
            DataFrame with device features added
        """
        return self.fit(df).transform(df)


class FingerprintFeatureExtractor:
    """
    Extract features from device fingerprints.
    
    Device fingerprinting creates a unique identifier for each device
    based on its characteristics. This helps track devices across sessions
    even without cookies.
    
    Features created:
    - device_fingerprint: Hashed device identifier
    - device_type: Type of device (mobile, desktop, tablet)
    - os_type: Operating system
    - browser_type: Browser used
    - screen_resolution: Screen resolution category
    - timezone: Device timezone
    - language: Device language settings
    - is_emulator: Whether device appears to be an emulator
    - is_vpn: Whether VPN/proxy is detected
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the FingerprintFeatureExtractor.
        
        Args:
            config: Configuration containing:
                - fingerprint_fields: Fields to include in fingerprint
                - resolution_bins: Screen resolution categories
                - vpn_detection: Whether to detect VPNs
        """
        self.config = config or {}
        self.fitted = False
        self.fingerprint_fields = self.config.get('fingerprint_fields', [
            'user_agent', 'screen_resolution', 'timezone', 'language',
            'platform', 'color_depth', 'touch_support'
        ])
        
    def fit(self, df: pd.DataFrame) -> 'FingerprintFeatureExtractor':
        """
        Fit the fingerprint extractor.
        
        Args:
            df: DataFrame with device fingerprint data
        """
        # Check for required columns
        self.has_user_agent = 'user_agent' in df.columns
        self.has_screen_res = 'screen_resolution' in df.columns
        self.has_timezone = 'timezone' in df.columns
        self.has_language = 'language' in df.columns
        
        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create fingerprint features.
        
        Args:
            df: DataFrame with device data
            
        Returns:
            DataFrame with fingerprint features added
        """
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")
            
        result_df = df.copy()
        
        # Generate device fingerprint
        result_df = self._create_device_fingerprint(result_df)
        
        # Extract device type from user agent
        if self.has_user_agent:
            result_df = self._extract_device_type(result_df)
        
        # Process screen resolution
        if self.has_screen_res:
            result_df = self._process_screen_resolution(result_df)
        
        # Process timezone
        if self.has_timezone:
            result_df = self._process_timezone(result_df)
        
        # Process language
        if self.has_language:
            result_df = self._process_language(result_df)
        
        # VPN/proxy detection
        result_df = self._detect_vpn_proxy(result_df)
        
        return result_df
    
    def _create_device_fingerprint(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a unique device fingerprint by hashing device characteristics.
        
        Args:
            df: DataFrame with device fields
            
        Returns:
            DataFrame with device fingerprint added
        """
        # Collect available fields for fingerprinting
        fingerprint_data = {}
        
        for field in self.fingerprint_fields:
            if field in df.columns:
                fingerprint_data[field] = df[field].astype(str)
        
        if fingerprint_data:
            # Create a combined string of all fields
            combined = pd.DataFrame(fingerprint_data).apply(
                lambda row: '|'.join(row.values.astype(str)), 
                axis=1
            )
            
            # Hash the combined string
            df['device_fingerprint'] = combined.apply(
                lambda x: hashlib.sha256(x.encode()).hexdigest()
            )
        else:
            # If no fingerprint fields, use device_id if available
            if 'device_id' in df.columns:
                df['device_fingerprint'] = df['device_id']
            else:
                df['device_fingerprint'] = 'unknown'
        
        # First time seeing this device?
        # This would need to be calculated across all data
        if not hasattr(self, 'seen_devices'):
            self.seen_devices = set()
        
        df['is_new_device'] = ~df['device_fingerprint'].isin(self.seen_devices)
        self.seen_devices.update(df['device_fingerprint'].unique())
        
        return df
    
    def _extract_device_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract device type from user agent string.
        
        Args:
            df: DataFrame with user_agent column
            
        Returns:
            DataFrame with device type features added
        """
        # Define patterns for device detection
        mobile_patterns = ['Mobile', 'Android', 'iPhone', 'iPad', 'Windows Phone']
        tablet_patterns = ['iPad', 'Tablet', 'Kindle']
        desktop_patterns = ['Windows NT', 'Macintosh', 'Linux x86_64']
        
        def classify_device(ua):
            """Classify device type based on user agent."""
            if pd.isna(ua):
                return 'unknown'
            
            ua_str = str(ua)
            
            # Check for tablet first
            if any(pattern in ua_str for pattern in tablet_patterns):
                return 'tablet'
            # Then mobile
            elif any(pattern in ua_str for pattern in mobile_patterns):
                return 'mobile'
            # Then desktop
            elif any(pattern in ua_str for pattern in desktop_patterns):
                return 'desktop'
            else:
                return 'unknown'
        
        def extract_os(ua):
            """Extract operating system from user agent."""
            if pd.isna(ua):
                return 'unknown'
            
            ua_str = str(ua)
            
            if 'Windows' in ua_str:
                return 'windows'
            elif 'Mac OS' in ua_str:
                return 'macos'
            elif 'Linux' in ua_str:
                return 'linux'
            elif 'Android' in ua_str:
                return 'android'
            elif 'iOS' in ua_str or 'iPhone' in ua_str or 'iPad' in ua_str:
                return 'ios'
            else:
                return 'unknown'
        
        def extract_browser(ua):
            """Extract browser from user agent."""
            if pd.isna(ua):
                return 'unknown'
            
            ua_str = str(ua)
            
            if 'Chrome' in ua_str and 'Edg' not in ua_str:
                return 'chrome'
            elif 'Firefox' in ua_str:
                return 'firefox'
            elif 'Safari' in ua_str and 'Chrome' not in ua_str:
                return 'safari'
            elif 'Edg' in ua_str:
                return 'edge'
            elif 'MSIE' in ua_str or 'Trident' in ua_str:
                return 'internet_explorer'
            else:
                return 'unknown'
        
        # Apply classification
        df['device_type'] = df['user_agent'].apply(classify_device)
        df['os_type'] = df['user_agent'].apply(extract_os)
        df['browser_type'] = df['user_agent'].apply(extract_browser)
        
        # One-hot encode categorical variables
        for col in ['device_type', 'os_type', 'browser_type']:
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False)
            df = pd.concat([df, dummies], axis=1)
        
        return df
    
    def _process_screen_resolution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process screen resolution into useful features.
        
        Args:
            df: DataFrame with screen_resolution column
            
        Returns:
            DataFrame with screen resolution features added
        """
        def parse_resolution(res):
            """Parse resolution string like '1920x1080' into width and height."""
            if pd.isna(res):
                return (0, 0)
            
            try:
                parts = str(res).lower().replace('x', ' ').replace('*', ' ').split()
                if len(parts) >= 2:
                    width = int(parts[0])
                    height = int(parts[1])
                    return (width, height)
            except:
                pass
            
            return (0, 0)
        
        # Parse resolution
        resolutions = df['screen_resolution'].apply(parse_resolution)
        df['screen_width'] = resolutions.apply(lambda x: x[0])
        df['screen_height'] = resolutions.apply(lambda x: x[1])
        
        # Calculate aspect ratio
        df['screen_aspect_ratio'] = df['screen_width'] / (df['screen_height'] + 1)
        
        # Screen size category
        def categorize_screen_size(width, height):
            """Categorize screen size based on dimensions."""
            if width == 0 or height == 0:
                return 'unknown'
            
            total_pixels = width * height
            
            if total_pixels < 480 * 800:
                return 'small'
            elif total_pixels < 768 * 1024:
                return 'medium'
            elif total_pixels < 1080 * 1920:
                return 'large'
            else:
                return 'xlarge'
        
        df['screen_size_category'] = df.apply(
            lambda row: categorize_screen_size(row['screen_width'], row['screen_height']),
            axis=1
        )
        
        # Common mobile resolutions flag
        mobile_resolutions = [640, 750, 828, 1080, 1170, 1242, 1440]
        df['is_mobile_resolution'] = df['screen_width'].isin(mobile_resolutions).astype(int)
        
        return df
    
    def _process_timezone(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process timezone information.
        
        Args:
            df: DataFrame with timezone column
            
        Returns:
            DataFrame with timezone features added
        """
        # Extract timezone offset if available
        def extract_offset(tz):
            """Extract offset from timezone string like 'America/New_York' or '-05:00'."""
            if pd.isna(tz):
                return None
            
            tz_str = str(tz)
            
            # If it's already an offset
            if tz_str.startswith(('+', '-')) and ':' in tz_str:
                try:
                    sign = 1 if tz_str[0] == '+' else -1
                    hours = int(tz_str[1:3])
                    minutes = int(tz_str[4:6]) if len(tz_str) > 4 else 0
                    return sign * (hours + minutes / 60)
                except:
                    pass
            
            # Otherwise, use a mapping of common timezones to offsets
            # This is simplified - in production use pytz
            tz_map = {
                'America/New_York': -5,
                'America/Chicago': -6,
                'America/Denver': -7,
                'America/Los_Angeles': -8,
                'Europe/London': 0,
                'Europe/Paris': 1,
                'Asia/Tokyo': 9,
                'Asia/Shanghai': 8,
                'Australia/Sydney': 11,
            }
            
            return tz_map.get(tz_str, None)
        
        df['timezone_offset'] = df['timezone'].apply(extract_offset)
        
        # Transaction time vs device timezone
        if 'transaction_time' in df.columns and 'timezone_offset' in df.columns:
            # Convert transaction time to UTC if it's not already
            if df['transaction_time'].dt.tz is None:
                # Assume UTC if no timezone
                df['transaction_hour_utc'] = df['transaction_time'].dt.hour
            else:
                df['transaction_hour_utc'] = df['transaction_time'].dt.tz_convert('UTC').dt.hour
            
            # Calculate local hour for device
            df['transaction_hour_local'] = (
                df['transaction_hour_utc'] + df['timezone_offset']
            ) % 24
            
            # Is transaction during local night?
            df['is_local_night'] = (
                (df['transaction_hour_local'] < 6) | 
                (df['transaction_hour_local'] > 22)
            ).astype(int)
        
        return df
    
    def _process_language(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process language settings.
        
        Args:
            df: DataFrame with language column
            
        Returns:
            DataFrame with language features added
        """
        # Extract primary language
        def extract_primary_language(lang):
            """Extract primary language from string like 'en-US,en;q=0.9'."""
            if pd.isna(lang):
                return 'unknown'
            
            lang_str = str(lang)
            # Get first language code
            primary = lang_str.split(',')[0].split('-')[0].lower()
            return primary
        
        df['primary_language'] = df['language'].apply(extract_primary_language)
        
        # Language match with country (if both available)
        if 'country' in df.columns:
            # Simplified mapping - in production use proper locale data
            country_to_language = {
                'US': 'en', 'GB': 'en', 'CA': 'en',
                'FR': 'fr', 'DE': 'de', 'ES': 'es',
                'IT': 'it', 'JP': 'ja', 'CN': 'zh',
                'RU': 'ru', 'BR': 'pt'
            }
            
            expected_language = df['country'].map(country_to_language)
            df['language_country_match'] = (
                df['primary_language'] == expected_language
            ).astype(int)
        
        return df
    
    def _detect_vpn_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect if the device is using a VPN or proxy.
        
        This is a simplified version - in production, you'd use
        specialized APIs or databases of known VPN IPs.
        
        Args:
            df: DataFrame with IP and other data
            
        Returns:
            DataFrame with VPN detection features added
        """
        # Check for known VPN IPs (simplified)
        # In production, you'd have a database of known VPN IP ranges
        if 'ip_address' in df.columns:
            # Placeholder for actual VPN detection logic
            # Here we'll just create a random flag for demonstration
            df['is_vpn'] = np.random.choice([0, 1], size=len(df), p=[0.95, 0.05])
            
            # Known datacenter IPs (where VPNs often originate)
            datacenter_ips = self.config.get('datacenter_ips', [])
            df['is_datacenter_ip'] = df['ip_address'].isin(datacenter_ips).astype(int)
        
        # Check for proxy headers
        proxy_headers = ['x-forwarded-for', 'via', 'x-proxy-id']
        df['has_proxy_headers'] = 0
        
        for header in proxy_headers:
            if header in df.columns:
                df['has_proxy_headers'] |= (~df[header].isna()).astype(int)
        
        return df


class DeviceBehavioralFeatureExtractor:
    """
    Extract behavioral features related to devices.
    
    These features capture how a device is typically used, which helps
    identify unusual patterns that might indicate fraud.
    
    Features created:
    - device_age_days: How long since first seen
    - device_transaction_count: Number of transactions from this device
    - device_velocity: Transactions per time period
    - device_amount_stats: Amount statistics for this device
    - device_success_rate: Success rate of transactions
    - device_fraud_count: Number of frauds from this device
    - device_risk_score: Overall device risk
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the DeviceBehavioralFeatureExtractor.
        
        Args:
            config: Configuration containing:
                - velocity_windows: Time windows for velocity calculation
                - risk_weights: Weights for different risk factors
        """
        self.config = config or {}
        self.fitted = False
        self.device_stats = {}  # Store statistics per device
        self.velocity_windows = self.config.get('velocity_windows', ['1H', '24H', '7D'])
        
    def fit(self, df: pd.DataFrame) -> 'DeviceBehavioralFeatureExtractor':
        """
        Fit the behavioral extractor.
        
        Args:
            df: DataFrame with device transaction history
        """
        # Check for required columns
        self.has_device_id = 'device_id' in df.columns or 'device_fingerprint' in df.columns
        self.has_transaction_time = 'transaction_time' in df.columns
        self.has_amount = 'amount' in df.columns
        
        # Determine device identifier column
        if self.has_device_id:
            self.device_col = 'device_fingerprint' if 'device_fingerprint' in df.columns else 'device_id'
        
        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create device behavioral features.
        
        Args:
            df: DataFrame with device transaction data
            
        Returns:
            DataFrame with behavioral features added
        """
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")
            
        result_df = df.copy()
        
        # Calculate device statistics
        if self.has_device_id and self.has_transaction_time:
            result_df = self._calculate_device_statistics(result_df)
        
        # Calculate device velocity
        if self.has_device_id and self.has_transaction_time:
            result_df = self._calculate_device_velocity(result_df)
        
        # Calculate device amount patterns
        if self.has_device_id and self.has_amount:
            result_df = self._calculate_device_amount_patterns(result_df)
        
        # Calculate device risk
        result_df = self._calculate_device_risk(result_df)
        
        return result_df
    
    def _calculate_device_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic statistics for each device.
        
        Args:
            df: DataFrame with device data
            
        Returns:
            DataFrame with device statistics added
        """
        device_col = self.device_col
        
        # Sort for time-based calculations
        df = df.sort_values([device_col, 'transaction_time'])
        
        # Device first seen
        device_first_seen = df.groupby(device_col)['transaction_time'].transform('min')
        df['device_first_seen'] = device_first_seen
        
        # Device age in days
        current_time = df['transaction_time'].max()
        df['device_age_days'] = (current_time - device_first_seen).dt.total_seconds() / 86400
        
        # Total transactions from this device
        device_tx_count = df.groupby(device_col)['transaction_time'].transform('count')
        df['device_total_tx'] = device_tx_count
        
        # Transactions in last 30 days
        thirty_days_ago = current_time - timedelta(days=30)
        recent_tx = df[df['transaction_time'] > thirty_days_ago].groupby(device_col).size()
        df['device_tx_30d'] = df[device_col].map(recent_tx).fillna(0)
        
        # Days since last transaction from this device
        df['device_last_tx'] = df.groupby(device_col)['transaction_time'].transform('max')
        df['device_days_since_last_tx'] = (
            (current_time - df['device_last_tx']).dt.total_seconds() / 86400
        )
        
        return df
    
    def _calculate_device_velocity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate transaction velocity for each device.
        
        Velocity measures how quickly transactions occur from a device,
        which can indicate automated or fraudulent activity.
        
        Args:
            df: DataFrame with device data
            
        Returns:
            DataFrame with velocity features added
        """
        device_col = self.device_col
        
        # For each window, calculate rolling count
        for window in self.velocity_windows:
            # Convert window string to pandas offset
            offset = pd.Timedelta(window)
            
            # Calculate rolling count
            velocity_col = f'device_velocity_{window}'
            
            # This is a simplified version - in production, use proper
            # rolling window calculations that respect time boundaries
            df[velocity_col] = df.groupby(device_col).rolling(
                window, on='transaction_time'
            )['transaction_time'].count().reset_index(0, drop=True)
        
        # Flag for high velocity
        df['is_high_device_velocity'] = (
            df.get('device_velocity_1H', 0) > 10
        ).astype(int)
        
        # Flag for burst activity
        if 'device_velocity_1H' in df.columns and 'device_velocity_24H' in df.columns:
            df['device_velocity_ratio'] = (
                df['device_velocity_1H'] * 24 / (df['device_velocity_24H'] + 1)
            )
            df['is_burst_activity'] = (df['device_velocity_ratio'] > 3).astype(int)
        
        return df
    
    def _calculate_device_amount_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate amount patterns for each device.
        
        Args:
            df: DataFrame with device and amount data
            
        Returns:
            DataFrame with amount pattern features added
        """
        device_col = self.device_col
        
        # Calculate amount statistics per device
        device_amount_stats = df.groupby(device_col)['amount'].agg([
            ('device_avg_amount', 'mean'),
            ('device_std_amount', 'std'),
            ('device_min_amount', 'min'),
            ('device_max_amount', 'max'),
            ('device_median_amount', 'median')
        ]).reset_index()
        
        # Merge back to original dataframe
        df = df.merge(device_amount_stats, on=device_col, how='left')
        
        # Fill NaN for devices with only one transaction
        df['device_std_amount'] = df['device_std_amount'].fillna(0)
        
        # Deviation from device average
        df['amount_deviation_from_device_avg'] = (
            (df['amount'] - df['device_avg_amount']) / 
            (df['device_std_amount'] + 1)
        )
        
        # Is amount unusually high for this device
        df['is_high_for_device'] = (
            df['amount'] > df['device_avg_amount'] + 2 * df['device_std_amount']
        ).astype(int)
        
        return df
    
    def _calculate_device_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate overall risk score for each device.
        
        Combines multiple factors into a single risk score.
        
        Args:
            df: DataFrame with device features
            
        Returns:
            DataFrame with device risk features added
        """
        risk_score = 0
        num_factors = 0
        
        # New device risk
        if 'is_new_device' in df.columns:
            risk_score += df['is_new_device'] * 30
            num_factors += 30
        
        # Device age risk (newer devices are riskier)
        if 'device_age_days' in df.columns:
            # Risk decreases with age
            age_risk = np.exp(-df['device_age_days'] / 30) * 20
            risk_score += age_risk
            num_factors += 20
        
        # High velocity risk
        if 'is_high_device_velocity' in df.columns:
            risk_score += df['is_high_device_velocity'] * 25
            num_factors += 25
        
        # Burst activity risk
        if 'is_burst_activity' in df.columns:
            risk_score += df['is_burst_activity'] * 20
            num_factors += 20
        
        # VPN risk
        if 'is_vpn' in df.columns:
            risk_score += df['is_vpn'] * 40
            num_factors += 40
        
        # Normalize to 0-100
        if num_factors > 0:
            df['device_risk_score'] = (risk_score / num_factors) * 100
        else:
            df['device_risk_score'] = 0
        
        # Device risk category
        risk_bins = [0, 30, 60, 80, 100]
        risk_labels = ['low', 'medium', 'high', 'critical']
        
        df['device_risk_category'] = pd.cut(
            df['device_risk_score'],
            bins=risk_bins,
            labels=risk_labels,
            right=False
        )
        
        return df