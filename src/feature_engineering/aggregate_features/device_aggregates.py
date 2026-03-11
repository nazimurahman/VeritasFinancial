"""
Device Aggregate Features for Banking Fraud Detection
=====================================================
This module implements features aggregated at the device level.
Device fingerprinting is crucial for detecting fraud rings and
identifying suspicious access patterns.

Key Concepts:
- Device fingerprinting: Unique device identifiers
- Device risk scoring: Historical fraud rates by device
- Device sharing: Multiple accounts from same device
- Device velocity: Transaction frequency per device
- Geolocation: Device location patterns
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


class DeviceAggregateFeatureEngineer:
    """
    Create device-level aggregate features.
    
    Banking Context:
    - Fraudsters often use multiple accounts from same device
    - Device changes (new device, factory reset) are risk indicators
    - Emulator/virtual devices are high risk
    - Device location should be consistent with customer location
    """
    
    def __init__(self):
        self.feature_columns = []
        self.device_profiles = {}
        
    def create_device_risk_features(self,
                                   df: pd.DataFrame,
                                   device_id_col: str = 'device_id',
                                   fraud_col: str = 'is_fraud') -> pd.DataFrame:
        """
        Create device risk features based on historical fraud.
        
        Parameters:
        -----------
        df : DataFrame with transaction data
        device_id_col : Device identifier
        fraud_col : Fraud label column
        
        Returns:
        --------
        DataFrame with device risk features
        """
        
        result_df = df.copy()
        
        if device_id_col not in result_df.columns:
            print("Warning: Device ID column not found")
            return result_df
        
        # Calculate device-level statistics
        device_stats = result_df.groupby(device_id_col).agg({
            fraud_col: [
                ('device_fraud_count', 'sum'),
                ('device_fraud_rate', 'mean'),
            ],
            'transaction_id': [
                ('device_tx_count', 'count')
            ],
            'customer_id': [
                ('device_unique_customers', 'nunique')
            ]
        })
        
        device_stats.columns = ['_'.join(col).strip() for col in device_stats.columns.values]
        device_stats = device_stats.reset_index()
        
        # Calculate derived risk metrics
        overall_fraud_rate = result_df[fraud_col].mean()
        
        # Bayesian smoothed risk score
        confidence_weight = 10  # Lower confidence for devices (fewer tx per device typically)
        device_stats['device_risk_score'] = (
            (device_stats['device_fraud_count'] + confidence_weight * overall_fraud_rate) /
            (device_stats['device_tx_count'] + confidence_weight)
        )
        
        # Device sharing score (more customers per device = higher risk)
        device_stats['device_sharing_score'] = (
            device_stats['device_unique_customers'] / (device_stats['device_tx_count'] + 1e-8)
        )
        
        # Flag for high-sharing devices (potential fraud rings)
        device_stats['is_shared_device'] = (
            device_stats['device_unique_customers'] > 1
        ).astype(int)
        
        device_stats['is_highly_shared'] = (
            device_stats['device_unique_customers'] > 3
        ).astype(int)
        
        # Flag for devices with fraud history
        device_stats['has_fraud_history'] = (
            device_stats['device_fraud_count'] > 0
        ).astype(int)
        
        # Merge back
        result_df = result_df.merge(
            device_stats[[
                device_id_col,
                'device_fraud_rate',
                'device_tx_count',
                'device_unique_customers',
                'device_risk_score',
                'device_sharing_score',
                'is_shared_device',
                'is_highly_shared',
                'has_fraud_history'
            ]],
            on=device_id_col,
            how='left'
        )
        
        # Fill NaN for new devices
        fill_values = {
            'device_fraud_rate': overall_fraud_rate,
            'device_tx_count': 0,
            'device_unique_customers': 1,
            'device_risk_score': overall_fraud_rate,
            'device_sharing_score': 0,
            'is_shared_device': 0,
            'is_highly_shared': 0,
            'has_fraud_history': 0
        }
        result_df = result_df.fillna(fill_values)
        
        # Device type flags
        result_df['is_new_device'] = (result_df['device_tx_count'] == 0).astype(int)
        result_df['is_low_volume_device'] = (result_df['device_tx_count'] < 5).astype(int)
        
        self.feature_columns.extend([
            'device_fraud_rate',
            'device_tx_count',
            'device_unique_customers',
            'device_risk_score',
            'device_sharing_score',
            'is_shared_device',
            'is_highly_shared',
            'has_fraud_history',
            'is_new_device',
            'is_low_volume_device'
        ])
        
        # Store device profiles
        self.device_profiles = device_stats.set_index(device_id_col).to_dict(orient='index')
        
        return result_df
    
    def create_device_velocity_features(self,
                                       df: pd.DataFrame,
                                       device_id_col: str = 'device_id',
                                       time_col: str = 'transaction_time') -> pd.DataFrame:
        """
        Create velocity features for device usage.
        
        High velocity from a single device can indicate:
        - Automated fraud attempts
        - Multiple accounts being tested
        - Brute force attacks
        """
        
        result_df = df.copy()
        result_df = result_df.sort_values([device_id_col, time_col])
        
        if device_id_col not in result_df.columns:
            return result_df
        
        # Time between transactions from same device
        result_df['device_time_since_last_tx'] = (
            result_df
            .groupby(device_id_col)[time_col]
            .diff()
            .dt.total_seconds() / 60  # in minutes
        )
        
        result_df['device_time_since_last_tx'] = result_df['device_time_since_last_tx'].fillna(-1)
        
        # Rolling transaction counts
        windows = ['1H', '24H']
        
        for window in windows:
            # Transaction count from this device
            count_feature = f'device_tx_count_{window}'
            result_df[count_feature] = (
                result_df
                .groupby(device_id_col)[time_col]
                .transform(lambda x: x.rolling(window, min_periods=1).count())
            )
            self.feature_columns.append(count_feature)
            
            # Unique customers on this device
            # More complex - would need to track unique customers in rolling window
            # For now, use a simpler approach
            result_df[f'device_unique_customers_{window}'] = (
                result_df
                .groupby(device_id_col)['customer_id']
                .transform(lambda x: x.rolling(window, min_periods=1)
                          .apply(lambda y: len(y.unique())))
            )
            self.feature_columns.append(f'device_unique_customers_{window}')
        
        # Device velocity flags
        if 'device_tx_count_1H' in result_df.columns:
            result_df['is_high_velocity_device'] = (
                result_df['device_tx_count_1H'] > 10
            ).astype(int)
            
            result_df['is_extreme_velocity_device'] = (
                result_df['device_tx_count_1H'] > 50
            ).astype(int)
            
            self.feature_columns.extend(['is_high_velocity_device', 'is_extreme_velocity_device'])
        
        return result_df
    
    def create_device_geography_features(self,
                                        df: pd.DataFrame,
                                        device_id_col: str = 'device_id',
                                        location_col: str = 'location',
                                        ip_col: str = 'ip_address') -> pd.DataFrame:
        """
        Create features about device location patterns.
        
        Inconsistent location patterns can indicate:
        - VPN/proxy usage
        - Device sharing across regions
        - Impossible travel (fraud indicator)
        """
        
        result_df = df.copy()
        
        if device_id_col not in result_df.columns:
            return result_df
        
        # Device location diversity
        if location_col in result_df.columns:
            location_diversity = result_df.groupby(device_id_col)[location_col].nunique()
            result_df['device_unique_locations'] = result_df[device_id_col].map(location_diversity)
            result_df['device_unique_locations'] = result_df['device_unique_locations'].fillna(1)
            
            # Flag for devices seen in multiple locations
            result_df['is_multilocation_device'] = (
                result_df['device_unique_locations'] > 1
            ).astype(int)
            
            self.feature_columns.extend(['device_unique_locations', 'is_multilocation_device'])
        
        # Device IP diversity
        if ip_col in result_df.columns:
            ip_diversity = result_df.groupby(device_id_col)[ip_col].nunique()
            result_df['device_unique_ips'] = result_df[device_id_col].map(ip_diversity)
            result_df['device_unique_ips'] = result_df['device_unique_ips'].fillna(1)
            
            # Flag for devices using multiple IPs
            result_df['is_multi_ip_device'] = (
                result_df['device_unique_ips'] > 1
            ).astype(int)
            
            self.feature_columns.extend(['device_unique_ips', 'is_multi_ip_device'])
        
        return result_df
    
    def create_device_characteristics_features(self,
                                              df: pd.DataFrame,
                                              device_id_col: str = 'device_id',
                                              device_type_col: str = 'device_type',
                                              os_col: str = 'os',
                                              browser_col: str = 'browser') -> pd.DataFrame:
        """
        Create features based on device characteristics.
        
        Device characteristics can indicate:
        - Emulator/virtual device usage
        - Outdated software (higher risk)
        - Inconsistent characteristics (spoofing)
        """
        
        result_df = df.copy()
        
        if device_id_col not in result_df.columns:
            return result_df
        
        # Device type risk scoring
        if device_type_col in result_df.columns:
            # Calculate fraud rate by device type
            device_type_risk = result_df.groupby(device_type_col)['is_fraud'].mean()
            result_df['device_type_risk'] = result_df[device_type_col].map(device_type_risk)
            
            # Known high-risk device types (emulators, virtual machines)
            high_risk_types = ['emulator', 'virtual_machine', 'simulator']
            result_df['is_high_risk_device_type'] = (
                result_df[device_type_col].isin(high_risk_types)
            ).astype(int)
            
            self.feature_columns.extend(['device_type_risk', 'is_high_risk_device_type'])
        
        # OS risk scoring
        if os_col in result_df.columns:
            os_risk = result_df.groupby(os_col)['is_fraud'].mean()
            result_df['os_risk'] = result_df[os_col].map(os_risk)
            
            # Outdated OS versions (higher risk)
            outdated_os = ['windows_7', 'windows_xp', 'ios_9', 'android_6']
            result_df['is_outdated_os'] = (
                result_df[os_col].isin(outdated_os)
            ).astype(int)
            
            self.feature_columns.extend(['os_risk', 'is_outdated_os'])
        
        # Browser risk scoring
        if browser_col in result_df.columns:
            browser_risk = result_df.groupby(browser_col)['is_fraud'].mean()
            result_df['browser_risk'] = result_df[browser_col].map(browser_risk)
            
            # Automated browser detection
            automated_browsers = ['headless', 'phantomjs', 'selenium']
            result_df['is_automated_browser'] = (
                result_df[browser_col].str.contains('|'.join(automated_browsers), case=False, na=False)
            ).astype(int)
            
            self.feature_columns.extend(['browser_risk', 'is_automated_browser'])
        
        return result_df
    
    def create_device_customer_features(self,
                                       df: pd.DataFrame,
                                       device_id_col: str = 'device_id',
                                       customer_id_col: str = 'customer_id') -> pd.DataFrame:
        """
        Create features about device-customer relationships.
        
        These capture how many devices a customer uses and vice versa,
        which can indicate account sharing or fraud rings.
        """
        
        result_df = df.copy()
        
        if device_id_col not in result_df.columns:
            return result_df
        
        # Devices per customer
        customer_device_count = result_df.groupby(customer_id_col)[device_id_col].nunique()
        result_df['customer_device_count'] = result_df[customer_id_col].map(customer_device_count)
        
        # Flag for customers using many devices
        result_df['is_multi_device_customer'] = (
            result_df['customer_device_count'] > 3
        ).astype(int)
        
        # Is this the customer's primary device?
        # Primary device = most frequently used by this customer
        customer_primary_device = (
            result_df
            .groupby([customer_id_col, device_id_col])
            .size()
            .reset_index(name='count')
            .sort_values([customer_id_col, 'count'], ascending=[True, False])
            .groupby(customer_id_col)
            .first()[device_id_col]
        )
        
        result_df['is_primary_device'] = (
            result_df[device_id_col] == result_df[customer_id_col].map(customer_primary_device)
        ).astype(int)
        
        # Flag for new device for this customer
        # First time this customer uses this device
        result_df['device_first_use_for_customer'] = (
            result_df
            .groupby([customer_id_col, device_id_col])
            .cumcount() == 0
        ).astype(int)
        
        self.feature_columns.extend([
            'customer_device_count',
            'is_multi_device_customer',
            'is_primary_device',
            'device_first_use_for_customer'
        ])
        
        return result_df
    
    def get_feature_names(self) -> List[str]:
        """Return list of created feature names."""
        return self.feature_columns