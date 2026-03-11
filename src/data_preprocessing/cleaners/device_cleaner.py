"""
Device Data Cleaner Module

This module handles cleaning and validation of device fingerprinting data.
Device information is critical for detecting fraud patterns like:
- Multiple accounts from same device
- Device spoofing
- Location-device mismatches
- Suspicious device characteristics

Key functionalities:
1. Device ID validation and normalization
2. Device type classification
3. Operating system validation
4. Browser fingerprinting
5. IP address geolocation
6. Device risk scoring
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
import socket
import struct
import ipaddress
import re

logger = logging.getLogger(__name__)


class DeviceCleaner:
    """
    Comprehensive device data cleaner for banking fraud detection.
    
    This class processes device fingerprinting data to identify suspicious
    patterns and calculate device-based risk scores.
    
    Attributes:
        config (dict): Configuration parameters for cleaning operations
        cleaned_stats (dict): Statistics about cleaning operations performed
        high_risk_os (list): Operating systems considered high risk
        high_risk_browsers (list): Browsers considered high risk
        vpn_ip_ranges (list): Known VPN IP ranges
        proxy_ip_ranges (list): Known proxy IP ranges
        tor_exit_nodes (list): Known Tor exit nodes
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the DeviceCleaner with configuration.
        
        Args:
            config: Dictionary containing cleaning parameters
                   If None, default configuration will be used
        """
        self.config = config or {
            'validate_device_id': True,
            'validate_ip': True,
            'validate_user_agent': True,
            'check_vpn_proxy': True,
            'calculate_risk_score': True,
            'max_devices_per_customer': 10,  # Maximum devices per customer (flag if exceeded)
            'known_bot_ips': [],  # List of known bot IPs
            'high_risk_countries': [],  # Countries considered high risk
            'ip_reputation_db': None  # Path to IP reputation database
        }
        
        # Initialize statistics tracking
        self.cleaned_stats = {
            'total_devices_processed': 0,
            'invalid_devices_removed': 0,
            'invalid_ips_corrected': 0,
            'vpn_proxy_detected': 0,
            'tor_detected': 0,
            'high_risk_devices': 0
        }
        
        # Define high-risk operating systems
        self.high_risk_os = [
            'windows xp', 'windows 95', 'windows 98', 'windows me',
            'android 4', 'android 5', 'ios 8', 'ios 9'
        ]
        
        # Define high-risk browsers
        self.high_risk_browsers = [
            'internet explorer', 'ie', 'netscape', 'opera mini'
        ]
        
        # Known VPN IP ranges (simplified - in production, use a database)
        self.vpn_ip_ranges = [
            '10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16'  # Private ranges
            # Add commercial VPN ranges here
        ]
        
        # Known proxy IP ranges
        self.proxy_ip_ranges = [
            # Add known proxy ranges here
        ]
        
        # Known Tor exit nodes
        self.tor_exit_nodes = [
            # Would load from Tor project list
        ]
        
        logger.info("DeviceCleaner initialized with config: %s", self.config)
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main cleaning method that orchestrates all device cleaning operations.
        
        Args:
            df: Raw device DataFrame
            
        Returns:
            Cleaned device DataFrame with additional features
        """
        logger.info(f"Starting device data cleaning on {len(df)} records")
        
        # Make a copy to avoid modifying original data
        cleaned_df = df.copy()
        
        # Track initial row count
        initial_count = len(cleaned_df)
        
        # Apply cleaning steps in sequence
        cleaned_df = self._validate_device_id(cleaned_df)
        cleaned_df = self._clean_device_info(cleaned_df)
        cleaned_df = self._clean_os_info(cleaned_df)
        cleaned_df = self._clean_browser_info(cleaned_df)
        cleaned_df = self._clean_ip_address(cleaned_df)
        cleaned_df = self._parse_user_agent(cleaned_df)
        
        if self.config['check_vpn_proxy']:
            cleaned_df = self._detect_vpn_proxy(cleaned_df)
            cleaned_df = self._detect_tor(cleaned_df)
        
        if self.config['calculate_risk_score']:
            cleaned_df = self._calculate_device_risk_score(cleaned_df)
        
        # Remove rows with critical missing data
        cleaned_df = self._remove_invalid_rows(cleaned_df)
        
        # Update statistics
        self.cleaned_stats['total_devices_processed'] = initial_count
        final_count = len(cleaned_df)
        logger.info(f"Device cleaning complete. Kept {final_count}/{initial_count} records")
        
        return cleaned_df
    
    def _validate_device_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and normalize device IDs.
        
        Args:
            df: DataFrame with device data
            
        Returns:
            DataFrame with validated device IDs
        """
        if 'device_id' not in df.columns:
            logger.warning("No device_id column found. Generating device IDs from fingerprints.")
            # Generate device ID from combination of device characteristics
            if all(col in df.columns for col in ['device_type', 'os', 'browser']):
                df['device_id'] = df.apply(
                    lambda row: hashlib.md5(
                        f"{row.get('device_type', '')}_{row.get('os', '')}_{row.get('browser', '')}".encode()
                    ).hexdigest(),
                    axis=1
                )
            else:
                df['device_id'] = [f"DEV_{i:010d}" for i in range(len(df))]
        
        # Convert to string and strip
        df['device_id'] = df['device_id'].astype(str).str.strip()
        
        # Handle empty IDs
        empty_mask = (df['device_id'] == '') | (df['device_id'] == 'nan') | (df['device_id'].isna())
        if empty_mask.any():
            logger.warning(f"Found {empty_mask.sum()} empty device IDs")
            df.loc[empty_mask, 'device_id'] = [f"DEV_{i:010d}" for i in range(empty_mask.sum())]
            self.cleaned_stats['null_values_filled'] += empty_mask.sum()
        
        return df
    
    def _clean_device_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize device information.
        
        Args:
            df: DataFrame with device data
            
        Returns:
            DataFrame with cleaned device information
        """
        # Clean device type
        if 'device_type' in df.columns:
            df['device_type'] = df['device_type'].astype(str).str.lower().str.strip()
            
            # Standardize device types
            device_type_mapping = {
                'desktop': 'DESKTOP', 'pc': 'DESKTOP', 'workstation': 'DESKTOP',
                'laptop': 'LAPTOP', 'notebook': 'LAPTOP', 'macbook': 'LAPTOP',
                'mobile': 'MOBILE', 'smartphone': 'MOBILE', 'iphone': 'MOBILE', 'android': 'MOBILE',
                'tablet': 'TABLET', 'ipad': 'TABLET', 'android tablet': 'TABLET',
                'server': 'SERVER', 'bot': 'BOT', 'crawler': 'BOT'
            }
            
            df['device_type_std'] = df['device_type'].map(device_type_mapping).fillna('OTHER')
            
            # Flag suspicious device types
            df['is_bot_device'] = (df['device_type_std'] == 'BOT').astype(int)
        
        # Clean device model
        if 'device_model' in df.columns:
            df['device_model'] = df['device_model'].astype(str).str.strip()
        
        # Clean screen resolution
        if 'screen_resolution' in df.columns:
            df['screen_resolution'] = df['screen_resolution'].astype(str).str.strip()
            
            # Extract width and height
            try:
                resolution_parts = df['screen_resolution'].str.extract(r'(\d+)[xX](\d+)')
                df['screen_width'] = pd.to_numeric(resolution_parts[0], errors='coerce')
                df['screen_height'] = pd.to_numeric(resolution_parts[1], errors='coerce')
                df['screen_area'] = df['screen_width'] * df['screen_height']
                
                # Flag unusual screen resolutions (bot detection)
                df['is_unusual_resolution'] = (
                    (df['screen_width'] < 800) | 
                    (df['screen_width'] > 3840) |
                    (df['screen_height'] < 600) |
                    (df['screen_height'] > 2160)
                ).astype(int)
            except:
                logger.warning("Could not parse screen resolutions")
        
        return df
    
    def _clean_os_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate operating system information.
        
        Args:
            df: DataFrame with device data
            
        Returns:
            DataFrame with cleaned OS information
        """
        if 'os' not in df.columns:
            return df
        
        df['os'] = df['os'].astype(str).str.lower().str.strip()
        
        # Extract OS family
        os_family_mapping = {
            'windows': 'WINDOWS', 'win': 'WINDOWS', 'win32': 'WINDOWS', 'win64': 'WINDOWS',
            'mac': 'MACOS', 'macos': 'MACOS', 'osx': 'MACOS', 'darwin': 'MACOS',
            'linux': 'LINUX', 'ubuntu': 'LINUX', 'debian': 'LINUX', 'centos': 'LINUX',
            'android': 'ANDROID', 'ios': 'IOS', 'iphone': 'IOS', 'ipad': 'IOS',
            'chrome': 'CHROME_OS', 'chromium': 'CHROME_OS'
        }
        
        for key, family in os_family_mapping.items():
            df.loc[df['os'].str.contains(key, na=False), 'os_family'] = family
        
        df['os_family'] = df['os_family'].fillna('OTHER')
        
        # Extract OS version
        version_pattern = r'(\d+\.?\d*\.?\d*)'
        df['os_version'] = df['os'].str.extract(version_pattern)[0]
        
        # Check if OS is outdated/high risk
        df['is_high_risk_os'] = df['os'].apply(
            lambda x: any(risk_os in str(x).lower() for risk_os in self.high_risk_os)
        ).astype(int)
        
        if df['is_high_risk_os'].sum() > 0:
            logger.warning(f"Found {df['is_high_risk_os'].sum()} high-risk operating systems")
            self.cleaned_stats['high_risk_devices'] += df['is_high_risk_os'].sum()
        
        return df
    
    def _clean_browser_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate browser information.
        
        Args:
            df: DataFrame with device data
            
        Returns:
            DataFrame with cleaned browser information
        """
        if 'browser' not in df.columns:
            return df
        
        df['browser'] = df['browser'].astype(str).str.lower().str.strip()
        
        # Extract browser family
        browser_family_mapping = {
            'chrome': 'CHROME', 'chromium': 'CHROME',
            'firefox': 'FIREFOX', 'mozilla': 'FIREFOX',
            'safari': 'SAFARI',
            'edge': 'EDGE', 'internet explorer': 'IE', 'ie': 'IE',
            'opera': 'OPERA', 'vivaldi': 'OPERA',
            'brave': 'BRAVE', 'tor': 'TOR'
        }
        
        for key, family in browser_family_mapping.items():
            df.loc[df['browser'].str.contains(key, na=False), 'browser_family'] = family
        
        df['browser_family'] = df['browser_family'].fillna('OTHER')
        
        # Extract browser version
        version_pattern = r'(\d+\.?\d*\.?\d*)'
        df['browser_version'] = df['browser'].str.extract(version_pattern)[0]
        
        # Check if browser is outdated/high risk
        df['is_high_risk_browser'] = df['browser'].apply(
            lambda x: any(risk_browser in str(x).lower() for risk_browser in self.high_risk_browsers)
        ).astype(int)
        
        # Check for headless browsers (bot detection)
        headless_indicators = ['headless', 'phantom', 'selenium', 'puppeteer']
        df['is_headless_browser'] = df['browser'].apply(
            lambda x: any(indicator in str(x).lower() for indicator in headless_indicators)
        ).astype(int)
        
        return df
    
    def _clean_ip_address(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate IP addresses.
        
        Args:
            df: DataFrame with device data
            
        Returns:
            DataFrame with cleaned IP information
        """
        if 'ip_address' not in df.columns:
            return df
        
        df['ip_address'] = df['ip_address'].astype(str).str.strip()
        
        # Function to validate and classify IP
        def process_ip(ip):
            try:
                # Validate IP
                ip_obj = ipaddress.ip_address(ip)
                
                # Determine IP version
                ip_version = ip_obj.version
                
                # Check if private
                is_private = ip_obj.is_private
                
                # Check if reserved
                is_reserved = ip_obj.is_reserved
                
                # Get IP type
                if ip_obj.is_loopback:
                    ip_type = 'LOOPBACK'
                elif ip_obj.is_multicast:
                    ip_type = 'MULTICAST'
                elif ip_obj.is_private:
                    ip_type = 'PRIVATE'
                elif ip_obj.is_global:
                    ip_type = 'PUBLIC'
                else:
                    ip_type = 'OTHER'
                
                return pd.Series({
                    'ip_valid': True,
                    'ip_version': ip_version,
                    'ip_type': ip_type,
                    'is_private_ip': int(is_private),
                    'is_reserved_ip': int(is_reserved)
                })
            except:
                return pd.Series({
                    'ip_valid': False,
                    'ip_version': None,
                    'ip_type': 'INVALID',
                    'is_private_ip': 0,
                    'is_reserved_ip': 0
                })
        
        # Apply IP processing
        ip_info = df['ip_address'].apply(process_ip)
        df = pd.concat([df, ip_info], axis=1)
        
        # Handle invalid IPs
        invalid_ips = (~df['ip_valid']).sum()
        if invalid_ips > 0:
            logger.warning(f"Found {invalid_ips} invalid IP addresses")
            # Replace invalid IPs with placeholder
            df.loc[~df['ip_valid'], 'ip_address'] = '0.0.0.0'
            self.cleaned_stats['invalid_ips_corrected'] += invalid_ips
        
        return df
    
    def _parse_user_agent(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse user agent string to extract device information.
        
        Args:
            df: DataFrame with device data
            
        Returns:
            DataFrame with parsed user agent information
        """
        if 'user_agent' not in df.columns:
            return df
        
        df['user_agent'] = df['user_agent'].astype(str)
        
        # Simple user agent parsing (in production, use a library like ua-parser)
        def parse_ua(ua):
            ua_lower = ua.lower()
            
            # Detect mobile
            is_mobile = any(x in ua_lower for x in ['mobile', 'android', 'iphone', 'ipad'])
            
            # Detect tablet
            is_tablet = 'ipad' in ua_lower or ('android' in ua_lower and 'mobile' not in ua_lower)
            
            # Detect bot
            is_bot = any(x in ua_lower for x in ['bot', 'crawler', 'spider', 'scraper'])
            
            # Detect automation tools
            is_automated = any(x in ua_lower for x in ['selenium', 'puppeteer', 'phantom'])
            
            return pd.Series({
                'ua_is_mobile': int(is_mobile),
                'ua_is_tablet': int(is_tablet),
                'ua_is_bot': int(is_bot),
                'ua_is_automated': int(is_automated)
            })
        
        ua_features = df['user_agent'].apply(parse_ua)
        df = pd.concat([df, ua_features], axis=1)
        
        return df
    
    def _detect_vpn_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect VPN and proxy usage.
        
        Args:
            df: DataFrame with device data
            
        Returns:
            DataFrame with VPN/proxy detection flags
        """
        if 'ip_address' not in df.columns:
            return df
        
        def check_vpn_proxy(ip):
            try:
                ip_obj = ipaddress.ip_address(ip)
                
                # Check against known VPN ranges
                is_vpn = any(ip_obj in ipaddress.ip_network(cidr) for cidr in self.vpn_ip_ranges)
                
                # Check against known proxy ranges
                is_proxy = any(ip_obj in ipaddress.ip_network(cidr) for cidr in self.proxy_ip_ranges)
                
                # Check for datacenter IPs (often used for VPNs)
                # In production, use IP geolocation database to check ASN
                
                return is_vpn or is_proxy
            except:
                return False
        
        df['is_vpn_proxy'] = df['ip_address'].apply(check_vpn_proxy).astype(int)
        
        vpn_count = df['is_vpn_proxy'].sum()
        if vpn_count > 0:
            logger.warning(f"Detected {vpn_count} VPN/proxy connections")
            self.cleaned_stats['vpn_proxy_detected'] += vpn_count
        
        return df
    
    def _detect_tor(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Tor network usage.
        
        Args:
            df: DataFrame with device data
            
        Returns:
            DataFrame with Tor detection flags
        """
        if 'ip_address' not in df.columns:
            return df
        
        # In production, check against Tor exit node list
        # This is a simplified version
        def check_tor(ip):
            # Check if IP is in Tor exit node list
            # Placeholder implementation
            return False
        
        df['is_tor'] = df['ip_address'].apply(check_tor).astype(int)
        
        tor_count = df['is_tor'].sum()
        if tor_count > 0:
            logger.warning(f"Detected {tor_count} Tor connections")
            self.cleaned_stats['tor_detected'] += tor_count
        
        return df
    
    def _calculate_device_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive device risk score.
        
        Args:
            df: DataFrame with device data
            
        Returns:
            DataFrame with device risk score
        """
        risk_factors = []
        
        # Base risk score (start at 0, add risk factors)
        risk_score = pd.Series(0, index=df.index)
        
        # Factor 1: High-risk OS
        if 'is_high_risk_os' in df.columns:
            risk_score += df['is_high_risk_os'] * 25
        
        # Factor 2: High-risk browser
        if 'is_high_risk_browser' in df.columns:
            risk_score += df['is_high_risk_browser'] * 20
        
        # Factor 3: Headless browser
        if 'is_headless_browser' in df.columns:
            risk_score += df['is_headless_browser'] * 30
        
        # Factor 4: Bot detection
        if 'ua_is_bot' in df.columns:
            risk_score += df['ua_is_bot'] * 40
        
        # Factor 5: Automated tools
        if 'ua_is_automated' in df.columns:
            risk_score += df['ua_is_automated'] * 35
        
        # Factor 6: VPN/Proxy
        if 'is_vpn_proxy' in df.columns:
            risk_score += df['is_vpn_proxy'] * 30
        
        # Factor 7: Tor
        if 'is_tor' in df.columns:
            risk_score += df['is_tor'] * 50
        
        # Factor 8: Unusual screen resolution
        if 'is_unusual_resolution' in df.columns:
            risk_score += df['is_unusual_resolution'] * 15
        
        # Factor 9: Private IP (unusual for real users)
        if 'is_private_ip' in df.columns:
            risk_score += df['is_private_ip'] * 10
        
        # Normalize to 0-100 range
        df['device_risk_score'] = risk_score.clip(0, 100)
        
        # Create risk categories
        df['device_risk_category'] = pd.cut(
            df['device_risk_score'],
            bins=[0, 20, 40, 60, 80, 100],
            labels=['very_low', 'low', 'medium', 'high', 'very_high']
        )
        
        # Flag high-risk devices
        df['is_high_risk_device'] = (df['device_risk_score'] >= 60).astype(int)
        
        high_risk_count = df['is_high_risk_device'].sum()
        if high_risk_count > 0:
            logger.warning(f"Identified {high_risk_count} high-risk devices")
            self.cleaned_stats['high_risk_devices'] += high_risk_count
        
        return df
    
    def _remove_invalid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with critical invalid data.
        
        Args:
            df: DataFrame with device data
            
        Returns:
            DataFrame with invalid rows removed
        """
        initial_count = len(df)
        
        # Define critical columns that must be valid
        critical_cols = ['device_id']
        for col in critical_cols:
            if col in df.columns:
                df = df[df[col].notna()]
        
        rows_removed = initial_count - len(df)
        if rows_removed > 0:
            logger.info(f"Removed {rows_removed} rows with critical missing data")
            self.cleaned_stats['invalid_devices_removed'] = rows_removed
        
        return df
    
    def get_cleaning_stats(self) -> Dict:
        """
        Return statistics about cleaning operations performed.
        
        Returns:
            Dictionary with cleaning statistics
        """
        return self.cleaned_stats