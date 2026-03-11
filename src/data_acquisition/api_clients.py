"""
Banking API Clients Module
=========================
This module provides robust API clients for connecting to various banking systems
and external data sources. Each client implements retry logic, connection pooling,
and comprehensive error handling for production reliability.
"""

import asyncio
import aiohttp
import requests
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import json
import hashlib
import hmac
import base64
from urllib.parse import urljoin, quote
import backoff
from ratelimit import limits, sleep_and_retry
import logging
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_log, after_log
)
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from functools import lru_cache
import time

# Configure logging
logger = logging.getLogger(__name__)

class APIClientBase:
    """
    Base class for all API clients with common functionality.
    
    This class provides:
    - Session management with connection pooling
    - Retry logic with exponential backoff
    - Request/response logging
    - Authentication handling
    - Rate limiting
    - Circuit breaker pattern
    
    Attributes:
        base_url (str): Base URL for the API
        timeout (int): Request timeout in seconds
        max_retries (int): Maximum number of retry attempts
        session (requests.Session): Persistent HTTP session
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        rate_limit_calls: int = 100,
        rate_limit_period: int = 60
    ):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL for the API
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            rate_limit_calls: Maximum number of calls per period
            rate_limit_period: Rate limit period in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limit_calls = rate_limit_calls
        self.rate_limit_period = rate_limit_period
        
        # Create session with connection pooling
        self.session = requests.Session()
        self.session.mount('https://', requests.adapters.HTTPAdapter(
            pool_connections=20,
            pool_maxsize=100,
            max_retries=max_retries,
            pool_block=False
        ))
        
        # Set default headers
        self.session.headers.update({
            'User-Agent': 'VeritasFinancial/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        
        # Add API key if provided
        if api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})
        
        # Circuit breaker state
        self.circuit_open = False
        self.circuit_open_until = None
        self.failure_count = 0
        self.circuit_threshold = 5
        self.circuit_timeout = 60
        
        logger.info(f"Initialized API client for {base_url}")
    
    def _check_circuit_breaker(self) -> bool:
        """
        Check if circuit breaker is open.
        
        Circuit breaker pattern prevents repeated calls to failing services.
        If circuit is open, requests are rejected immediately.
        
        Returns:
            bool: True if circuit is closed (requests allowed), False if open
        """
        if self.circuit_open:
            if datetime.now() > self.circuit_open_until:
                # Circuit reset after timeout
                self.circuit_open = False
                self.failure_count = 0
                logger.info("Circuit breaker reset")
                return True
            return False
        return True
    
    def _record_success(self):
        """Record successful request to reset circuit breaker."""
        self.failure_count = 0
        self.circuit_open = False
    
    def _record_failure(self):
        """Record failed request and potentially open circuit breaker."""
        self.failure_count += 1
        if self.failure_count >= self.circuit_threshold:
            self.circuit_open = True
            self.circuit_open_until = datetime.now() + timedelta(seconds=self.circuit_timeout)
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.HTTPError
        )),
        before=before_log(logger, logging.DEBUG),
        after=after_log(logger, logging.DEBUG)
    )
    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        **kwargs
    ) -> requests.Response:
        """
        Make HTTP request with retry logic.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            params: Query parameters
            data: Request body data
            headers: Additional headers
            **kwargs: Additional arguments for requests
            
        Returns:
            requests.Response: Response object
            
        Raises:
            Exception: Various exceptions based on response status
        """
        # Check circuit breaker
        if not self._check_circuit_breaker():
            raise Exception("Circuit breaker is open - service temporarily unavailable")
        
        url = urljoin(self.base_url, endpoint)
        
        # Merge headers
        request_headers = self.session.headers.copy()
        if headers:
            request_headers.update(headers)
        
        # Log request
        logger.debug(f"Making {method} request to {url}")
        
        start_time = time.time()
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=request_headers,
                timeout=self.timeout,
                **kwargs
            )
            
            # Log response time
            elapsed = time.time() - start_time
            logger.debug(f"Request completed in {elapsed:.2f}s with status {response.status_code}")
            
            # Raise exception for bad status codes
            response.raise_for_status()
            
            # Record success
            self._record_success()
            
            return response
            
        except requests.exceptions.RequestException as e:
            # Record failure
            self._record_failure()
            
            # Log error
            elapsed = time.time() - start_time
            logger.error(f"Request failed after {elapsed:.2f}s: {str(e)}")
            
            # Re-raise with more context
            raise Exception(f"API request failed: {str(e)}")
    
    def get(self, endpoint: str, **kwargs) -> Dict:
        """
        Make GET request.
        
        Args:
            endpoint: API endpoint
            **kwargs: Additional arguments
            
        Returns:
            Dict: JSON response
        """
        response = self._make_request('GET', endpoint, **kwargs)
        return response.json()
    
    def post(self, endpoint: str, data: Dict, **kwargs) -> Dict:
        """
        Make POST request.
        
        Args:
            endpoint: API endpoint
            data: Request data
            **kwargs: Additional arguments
            
        Returns:
            Dict: JSON response
        """
        response = self._make_request('POST', endpoint, data=data, **kwargs)
        return response.json()
    
    def close(self):
        """Close the session and clean up resources."""
        self.session.close()
        logger.info("API client session closed")


class BankingAPIClient(APIClientBase):
    """
    Client for core banking system APIs.
    
    This client connects to the bank's core transaction processing system
    to fetch real-time and historical transaction data. It handles:
    - Transaction queries by various criteria
    - Customer account information
    - Account balances and limits
    - Transaction history
    
    The client implements banking-specific security measures including
    HMAC signatures and audit logging.
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: str,
        api_secret: str,
        institution_id: str,
        **kwargs
    ):
        """
        Initialize banking API client.
        
        Args:
            base_url: Banking system API base URL
            api_key: API key for authentication
            api_secret: Secret key for HMAC signatures
            institution_id: Financial institution identifier
            **kwargs: Additional arguments for base client
        """
        super().__init__(base_url, api_key, **kwargs)
        self.api_secret = api_secret
        self.institution_id = institution_id
        
        # Add banking-specific headers
        self.session.headers.update({
            'X-Institution-ID': institution_id,
            'X-API-Version': 'v2'
        })
        
        logger.info(f"Banking API client initialized for institution {institution_id}")
    
    def _generate_signature(self, method: str, path: str, timestamp: str, body: str = '') -> str:
        """
        Generate HMAC signature for request authentication.
        
        Banking APIs require request signing for security. This method
        creates an HMAC-SHA256 signature using the API secret.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            path: Request path
            timestamp: ISO format timestamp
            body: Request body string (for POST/PUT)
            
        Returns:
            str: Base64 encoded signature
        """
        # Create signature string
        message = f"{method}\n{path}\n{timestamp}\n{body}"
        
        # Generate HMAC
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()
        
        return base64.b64encode(signature).decode('utf-8')
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """
        Override base _make_request to add banking-specific headers.
        
        Adds:
        - Request timestamp for audit
        - HMAC signature for authentication
        - Request ID for tracing
        """
        # Add timestamp
        timestamp = datetime.utcnow().isoformat() + 'Z'
        
        # Generate request ID for tracing
        request_id = hashlib.md5(f"{timestamp}{endpoint}".encode()).hexdigest()
        
        # Prepare body string for signature
        body = ''
        if 'data' in kwargs and kwargs['data']:
            body = json.dumps(kwargs['data'])
        
        # Generate signature
        signature = self._generate_signature(method, endpoint, timestamp, body)
        
        # Add banking headers
        if 'headers' not in kwargs:
            kwargs['headers'] = {}
        kwargs['headers'].update({
            'X-Request-Timestamp': timestamp,
            'X-Request-Signature': signature,
            'X-Request-ID': request_id
        })
        
        return super()._make_request(method, endpoint, **kwargs)
    
    def get_transaction(self, transaction_id: str) -> Dict:
        """
        Fetch a single transaction by ID.
        
        Args:
            transaction_id: Unique transaction identifier
            
        Returns:
            Dict: Transaction details including amount, timestamp, merchant, etc.
        """
        endpoint = f"/transactions/{transaction_id}"
        return self.get(endpoint)
    
    def get_transactions_batch(
        self,
        start_date: datetime,
        end_date: datetime,
        account_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        batch_size: int = 1000,
        max_batches: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch transactions in batches with pagination.
        
        This method handles large-scale data extraction with:
        - Automatic pagination
        - Rate limiting compliance
        - Progress tracking
        - Error recovery
        
        Args:
            start_date: Start date for transaction query
            end_date: End date for transaction query
            account_id: Optional account filter
            customer_id: Optional customer filter
            batch_size: Number of transactions per batch
            max_batches: Maximum number of batches to fetch
            
        Returns:
            pd.DataFrame: DataFrame with all transactions
        """
        all_transactions = []
        page = 1
        batch_count = 0
        
        logger.info(f"Fetching transactions from {start_date} to {end_date}")
        
        while True:
            # Check if we've reached max batches
            if max_batches and batch_count >= max_batches:
                logger.info(f"Reached maximum batches ({max_batches})")
                break
            
            # Prepare request parameters
            params = {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'page': page,
                'page_size': batch_size
            }
            
            if account_id:
                params['account_id'] = account_id
            if customer_id:
                params['customer_id'] = customer_id
            
            try:
                # Make request
                endpoint = "/transactions"
                response = self.get(endpoint, params=params)
                
                # Extract transactions
                transactions = response.get('transactions', [])
                if not transactions:
                    logger.info("No more transactions to fetch")
                    break
                
                all_transactions.extend(transactions)
                batch_count += 1
                
                logger.info(f"Fetched batch {batch_count}: {len(transactions)} transactions")
                
                # Check if this was the last page
                if not response.get('has_next', False):
                    logger.info("Reached last page")
                    break
                
                # Increment page for next batch
                page += 1
                
                # Small delay to respect rate limits
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error fetching batch {page}: {str(e)}")
                # Continue with next page? Or break?
                break
        
        # Convert to DataFrame
        if all_transactions:
            df = pd.DataFrame(all_transactions)
            
            # Convert timestamp columns
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            logger.info(f"Fetched {len(df)} transactions total")
            return df
        else:
            logger.warning("No transactions found")
            return pd.DataFrame()
    
    def get_customer_profile(self, customer_id: str) -> Dict:
        """
        Fetch customer profile information.
        
        Args:
            customer_id: Customer identifier
            
        Returns:
            Dict: Customer details including KYC status, risk rating, etc.
        """
        endpoint = f"/customers/{customer_id}"
        return self.get(endpoint)
    
    def get_account_balance(self, account_id: str) -> float:
        """
        Fetch current account balance.
        
        Args:
            account_id: Account identifier
            
        Returns:
            float: Current account balance
        """
        endpoint = f"/accounts/{account_id}/balance"
        response = self.get(endpoint)
        return response.get('balance', 0.0)
    
    def get_daily_summary(self, date: datetime) -> Dict:
        """
        Fetch daily transaction summary for fraud monitoring.
        
        Args:
            date: Date for summary
            
        Returns:
            Dict: Daily statistics including volume, value, anomalies
        """
        endpoint = "/reports/daily-summary"
        params = {'date': date.isoformat()}
        return self.get(endpoint, params=params)


class FraudIntelligenceClient(APIClientBase):
    """
    Client for external fraud intelligence feeds.
    
    This client connects to fraud intelligence services that provide:
    - Known fraudster databases
    - Suspicious IP addresses
    - Compromised card numbers
    - Fraud pattern indicators
    - Risk scores for various entities
    
    The client maintains local caches to reduce API calls and improve performance.
    """
    
    def __init__(self, base_url: str, api_key: str, cache_ttl: int = 3600, **kwargs):
        """
        Initialize fraud intelligence client.
        
        Args:
            base_url: Intelligence service API base URL
            api_key: API key for authentication
            cache_ttl: Cache time-to-live in seconds
            **kwargs: Additional arguments for base client
        """
        super().__init__(base_url, api_key, **kwargs)
        self.cache_ttl = cache_ttl
        self.cache = {}
        self.cache_timestamps = {}
        
        logger.info("Fraud intelligence client initialized")
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """
        Retrieve value from cache if not expired.
        
        Args:
            key: Cache key
            
        Returns:
            Optional[Any]: Cached value or None if expired/missing
        """
        if key in self.cache:
            timestamp = self.cache_timestamps.get(key)
            if timestamp and (datetime.now() - timestamp).total_seconds() < self.cache_ttl:
                logger.debug(f"Cache hit for {key}")
                return self.cache[key]
            else:
                # Remove expired entry
                del self.cache[key]
                del self.cache_timestamps[key]
                logger.debug(f"Cache expired for {key}")
        
        return None
    
    def _set_cache(self, key: str, value: Any):
        """
        Store value in cache with timestamp.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        self.cache[key] = value
        self.cache_timestamps[key] = datetime.now()
        logger.debug(f"Cached {key}")
    
    @sleep_and_retry
    @limits(calls=100, period=60)
    def check_ip_address(self, ip_address: str) -> Dict:
        """
        Check IP address against fraud databases.
        
        Args:
            ip_address: IP address to check
            
        Returns:
            Dict: Risk assessment including:
                - risk_score: 0-100 risk score
                - known_attacks: List of known attack types
                - proxy_detected: Whether IP is a proxy/VPN
                - country: Country of origin
                - isp: Internet service provider
        """
        # Check cache first
        cache_key = f"ip:{ip_address}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        # Make API request
        endpoint = "/v1/ip/check"
        params = {'ip': ip_address}
        
        try:
            response = self.get(endpoint, params=params)
            
            # Cache the result
            self._set_cache(cache_key, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error checking IP {ip_address}: {str(e)}")
            # Return default risk assessment on error
            return {
                'ip_address': ip_address,
                'risk_score': 50,  # Default medium risk
                'error': str(e),
                'cached': False
            }
    
    def check_email(self, email: str) -> Dict:
        """
        Check email address against fraud databases.
        
        Args:
            email: Email address to check
            
        Returns:
            Dict: Email risk assessment including:
                - risk_score: 0-100 risk score
                - domain_age: Age of email domain
                - disposable: Whether email is disposable
                - breaches: Known data breaches
        """
        cache_key = f"email:{email}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        endpoint = "/v1/email/check"
        params = {'email': email}
        
        try:
            response = self.get(endpoint, params=params)
            self._set_cache(cache_key, response)
            return response
        except Exception as e:
            logger.error(f"Error checking email {email}: {str(e)}")
            return {'email': email, 'risk_score': 50, 'error': str(e)}
    
    def check_phone(self, phone_number: str) -> Dict:
        """
        Check phone number against fraud databases.
        
        Args:
            phone_number: Phone number to check
            
        Returns:
            Dict: Phone risk assessment including:
                - risk_score: 0-100 risk score
                - carrier: Phone carrier
                - line_type: Mobile, landline, VoIP
                - ported: Whether number was ported
        """
        cache_key = f"phone:{phone_number}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        endpoint = "/v1/phone/check"
        params = {'phone': phone_number}
        
        try:
            response = self.get(endpoint, params=params)
            self._set_cache(cache_key, response)
            return response
        except Exception as e:
            logger.error(f"Error checking phone {phone_number}: {str(e)}")
            return {'phone': phone_number, 'risk_score': 50, 'error': str(e)}
    
    def get_fraud_patterns(self, pattern_type: Optional[str] = None) -> List[Dict]:
        """
        Get current fraud patterns and indicators.
        
        Args:
            pattern_type: Optional filter for specific pattern types
            
        Returns:
            List[Dict]: List of fraud patterns with:
                - pattern_id: Unique identifier
                - description: Pattern description
                - indicators: List of indicators
                - confidence: Pattern confidence score
        """
        endpoint = "/v1/patterns"
        params = {}
        if pattern_type:
            params['type'] = pattern_type
        
        try:
            response = self.get(endpoint, params=params)
            return response.get('patterns', [])
        except Exception as e:
            logger.error(f"Error fetching fraud patterns: {str(e)}")
            return []
    
    def submit_fraud_report(self, transaction_data: Dict) -> Dict:
        """
        Submit confirmed fraud to intelligence network.
        
        Args:
            transaction_data: Details of confirmed fraud transaction
            
        Returns:
            Dict: Submission confirmation
        """
        endpoint = "/v1/reports"
        data = {
            'timestamp': datetime.utcnow().isoformat(),
            'transaction': transaction_data,
            'reporter': 'VeritasFinancial'
        }
        
        try:
            response = self.post(endpoint, data=data)
            logger.info(f"Fraud report submitted: {response.get('report_id')}")
            return response
        except Exception as e:
            logger.error(f"Error submitting fraud report: {str(e)}")
            return {'status': 'error', 'message': str(e)}


class MerchantDataClient(APIClientBase):
    """
    Client for merchant information and risk scoring.
    
    This client provides:
    - Merchant category codes (MCC)
    - Merchant risk ratings
    - Historical chargeback rates
    - Geographic location validation
    - Business verification status
    
    Used to assess the risk level of merchants involved in transactions.
    """
    
    def __init__(self, base_url: str, api_key: str, **kwargs):
        """
        Initialize merchant data client.
        
        Args:
            base_url: Merchant data API base URL
            api_key: API key for authentication
            **kwargs: Additional arguments for base client
        """
        super().__init__(base_url, api_key, **kwargs)
        self.merchant_cache = {}
        
        logger.info("Merchant data client initialized")
    
    def get_merchant_details(self, merchant_id: str) -> Dict:
        """
        Get detailed information about a merchant.
        
        Args:
            merchant_id: Merchant identifier
            
        Returns:
            Dict: Merchant details including:
                - name: Merchant name
                - mcc: Merchant category code
                - country: Country of operation
                - risk_rating: A-F risk rating
                - verified: Whether merchant is verified
                - years_in_business: Operating history
        """
        # Check cache
        if merchant_id in self.merchant_cache:
            return self.merchant_cache[merchant_id]
        
        endpoint = f"/v1/merchants/{merchant_id}"
        
        try:
            response = self.get(endpoint)
            
            # Cache the result
            self.merchant_cache[merchant_id] = response
            
            return response
            
        except Exception as e:
            logger.error(f"Error fetching merchant {merchant_id}: {str(e)}")
            return {
                'merchant_id': merchant_id,
                'risk_rating': 'C',  # Default medium risk
                'error': str(e)
            }
    
    def get_mcc_risk_score(self, mcc_code: str) -> float:
        """
        Get risk score for a merchant category code.
        
        Different merchant categories have different inherent fraud risks.
        For example, electronics and travel have higher fraud rates.
        
        Args:
            mcc_code: Merchant category code
            
        Returns:
            float: Risk score between 0-1
        """
        # MCC risk mapping based on historical fraud rates
        mcc_risk_scores = {
            # High-risk categories
            '5944': 0.8,   # Jewelry stores
            '4814': 0.7,   # Telecommunication services
            '4722': 0.7,   # Travel agencies
            '5732': 0.6,   # Electronics stores
            
            # Medium-risk categories
            '5812': 0.4,   # Restaurants
            '5311': 0.4,   # Department stores
            '5411': 0.3,   # Grocery stores
            
            # Low-risk categories
            '4900': 0.1,   # Utilities
            '9311': 0.1,   # Tax payments
            '8090': 0.1,   # Health services
        }
        
        # Return mapped risk or default medium risk
        return mcc_risk_scores.get(mcc_code, 0.5)
    
    def get_chargeback_rate(self, merchant_id: str, days: int = 90) -> float:
        """
        Get historical chargeback rate for a merchant.
        
        Args:
            merchant_id: Merchant identifier
            days: Number of days to look back
            
        Returns:
            float: Chargeback rate as percentage of transactions
        """
        endpoint = f"/v1/merchants/{merchant_id}/chargebacks"
        params = {'days': days}
        
        try:
            response = self.get(endpoint, params=params)
            return response.get('chargeback_rate', 0.0)
        except Exception as e:
            logger.error(f"Error fetching chargeback rate: {str(e)}")
            return 0.0
    
    def validate_merchant_location(self, merchant_id: str, country: str, city: str) -> bool:
        """
        Validate if merchant location matches registered location.
        
        Used to detect location spoofing where fraudsters claim
        transactions from merchants in different locations.
        
        Args:
            merchant_id: Merchant identifier
            country: Claimed country
            city: Claimed city
            
        Returns:
            bool: True if location matches registered location
        """
        merchant = self.get_merchant_details(merchant_id)
        
        registered_country = merchant.get('country')
        registered_city = merchant.get('city')
        
        if not registered_country or not registered_city:
            logger.warning(f"Merchant {merchant_id} has no registered location")
            return False
        
        # Check country match (case-insensitive)
        country_match = country.upper() == registered_country.upper()
        
        # Check city match (case-insensitive)
        city_match = city.upper() == registered_city.upper()
        
        if not country_match:
            logger.warning(f"Merchant {merchant_id} country mismatch: {country} vs {registered_country}")
        
        return country_match and city_match


class GeoIPClient(APIClientBase):
    """
    Client for IP geolocation and risk assessment.
    
    This client provides:
    - Geographic location from IP address
    - ISP and connection type detection
    - Proxy/VPN/Tor detection
    - IP reputation scoring
    - Timezone and currency information
    
    Critical for detecting location-based fraud indicators.
    """
    
    def __init__(self, base_url: str, api_key: str, **kwargs):
        """
        Initialize GeoIP client.
        
        Args:
            base_url: GeoIP service API base URL
            api_key: API key for authentication
            **kwargs: Additional arguments for base client
        """
        super().__init__(base_url, api_key, **kwargs)
        self.ip_cache = {}
        
        logger.info("GeoIP client initialized")
    
    @lru_cache(maxsize=10000)
    def get_ip_info(self, ip_address: str) -> Dict:
        """
        Get geographic and network information for an IP address.
        
        Uses LRU cache to reduce API calls for frequently seen IPs.
        
        Args:
            ip_address: IP address to lookup
            
        Returns:
            Dict: IP information including:
                - country: Country code
                - city: City name
                - latitude: Latitude coordinate
                - longitude: Longitude coordinate
                - isp: Internet service provider
                - connection_type: Broadband, mobile, etc.
                - is_proxy: Whether IP is a proxy/VPN
                - is_tor: Whether IP is a Tor exit node
                - risk_score: IP reputation score (0-100)
        """
        endpoint = "/v1/ip"
        params = {'ip': ip_address}
        
        try:
            response = self.get(endpoint, params=params)
            
            # Add derived fields
            response['is_suspicious'] = (
                response.get('is_proxy', False) or
                response.get('is_tor', False) or
                response.get('risk_score', 0) > 70
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting IP info for {ip_address}: {str(e)}")
            return {
                'ip_address': ip_address,
                'country': 'Unknown',
                'risk_score': 50,
                'is_suspicious': False,
                'error': str(e)
            }
    
    def calculate_distance(
        self,
        ip_address: str,
        latitude: float,
        longitude: float
    ) -> Optional[float]:
        """
        Calculate distance between IP location and claimed location.
        
        Used to detect impossible travel scenarios where a transaction
        occurs far from the user's claimed location.
        
        Args:
            ip_address: IP address to check
            latitude: Claimed latitude
            longitude: Claimed longitude
            
        Returns:
            Optional[float]: Distance in kilometers, or None if location unknown
        """
        import math
        
        ip_info = self.get_ip_info(ip_address)
        
        ip_lat = ip_info.get('latitude')
        ip_lon = ip_info.get('longitude')
        
        if ip_lat is None or ip_lon is None:
            return None
        
        # Haversine formula for distance calculation
        R = 6371  # Earth's radius in kilometers
        
        lat1 = math.radians(ip_lat)
        lon1 = math.radians(ip_lon)
        lat2 = math.radians(latitude)
        lon2 = math.radians(longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        distance = R * c
        
        return distance
    
    def is_high_risk_country(self, ip_address: str) -> bool:
        """
        Check if IP is from a high-risk country for fraud.
        
        Args:
            ip_address: IP address to check
            
        Returns:
            bool: True if from high-risk country
        """
        # List of high-risk countries for fraud (simplified)
        high_risk_countries = {
            'NG', 'GH', 'CM', 'CI',  # Africa high-risk
            'RU', 'UA', 'BY',         # Eastern Europe
            'ID', 'PH', 'VN',         # Southeast Asia
            'BR', 'MX', 'CO'          # Latin America
        }
        
        ip_info = self.get_ip_info(ip_address)
        country = ip_info.get('country', '').upper()
        
        return country in high_risk_countries
    
    def get_timezone_mismatch(self, ip_address: str, local_timezone: str) -> bool:
        """
        Check if IP timezone matches user's local timezone.
        
        Mismatches can indicate account takeover or VPN use.
        
        Args:
            ip_address: IP address to check
            local_timezone: User's registered timezone
            
        Returns:
            bool: True if timezones don't match
        """
        ip_info = self.get_ip_info(ip_address)
        ip_timezone = ip_info.get('timezone')
        
        if not ip_timezone:
            return False
        
        return ip_timezone != local_timezone


class DeviceFingerprintClient:
    """
    Client for device fingerprinting and identification.
    
    This client creates and manages unique device identifiers based on
    various device characteristics. It helps track devices across sessions
    and detect device-based fraud patterns.
    
    Device fingerprinting considers:
    - Browser user agent
    - Screen resolution and color depth
    - Installed fonts and plugins
    - Timezone and language settings
    - Canvas fingerprinting
    - WebGL renderer
    - CPU cores and memory
    - Touch support
    """
    
    def __init__(self, redis_client=None):
        """
        Initialize device fingerprint client.
        
        Args:
            redis_client: Optional Redis client for caching fingerprints
        """
        self.redis_client = redis_client
        self.fingerprint_cache = {}
        
        logger.info("Device fingerprint client initialized")
    
    def generate_fingerprint(self, device_data: Dict) -> str:
        """
        Generate unique fingerprint from device characteristics.
        
        Combines multiple device attributes into a stable hash that
        uniquely identifies the device while respecting privacy.
        
        Args:
            device_data: Dictionary of device characteristics including:
                - user_agent: Browser user agent string
                - screen_resolution: Screen width x height
                - color_depth: Screen color depth
                - timezone: Browser timezone
                - language: Browser language
                - platform: Operating system platform
                - hardware_concurrency: CPU cores
                - device_memory: Available memory (GB)
                - touch_support: Whether touch is supported
                - canvas_hash: Canvas fingerprint hash
                - webgl_vendor: WebGL renderer info
                - fonts: List of installed fonts
                
        Returns:
            str: Unique device fingerprint hash
        """
        # Select stable components for fingerprinting
        components = []
        
        # Browser and OS (from user agent)
        if 'user_agent' in device_data:
            components.append(device_data['user_agent'])
        
        # Hardware characteristics
        if 'screen_resolution' in device_data:
            components.append(str(device_data['screen_resolution']))
        
        if 'color_depth' in device_data:
            components.append(str(device_data['color_depth']))
        
        if 'hardware_concurrency' in device_data:
            components.append(str(device_data['hardware_concurrency']))
        
        if 'device_memory' in device_data:
            components.append(str(device_data['device_memory']))
        
        # Software characteristics
        if 'platform' in device_data:
            components.append(device_data['platform'])
        
        if 'language' in device_data:
            components.append(device_data['language'])
        
        if 'timezone' in device_data:
            components.append(device_data['timezone'])
        
        # Advanced fingerprints
        if 'canvas_hash' in device_data:
            components.append(device_data['canvas_hash'])
        
        if 'webgl_vendor' in device_data:
            components.append(device_data['webgl_vendor'])
        
        # Sort for consistency
        components.sort()
        
        # Create hash
        fingerprint_string = '|'.join(components)
        fingerprint_hash = hashlib.sha256(
            fingerprint_string.encode('utf-8')
        ).hexdigest()
        
        logger.debug(f"Generated fingerprint {fingerprint_hash[:8]}... from {len(components)} components")
        
        return fingerprint_hash
    
    def get_device_risk_score(self, device_id: str, device_data: Dict) -> float:
        """
        Calculate risk score for a device based on its characteristics.
        
        Identifies potentially risky devices like:
        - Headless browsers
        - Automated scripts
        - Emulators and virtual machines
        - Devices with inconsistent attributes
        
        Args:
            device_id: Device identifier
            device_data: Device characteristics
            
        Returns:
            float: Risk score between 0-1
        """
        risk_factors = []
        
        # Check for headless browsers
        user_agent = device_data.get('user_agent', '').lower()
        if 'headless' in user_agent or 'phantom' in user_agent:
            risk_factors.append(0.8)
        
        # Check for automation tools
        automation_indicators = ['selenium', 'webdriver', 'puppeteer']
        for indicator in automation_indicators:
            if indicator in user_agent:
                risk_factors.append(0.7)
                break
        
        # Check for virtual machines
        vm_indicators = ['vmware', 'virtualbox', 'kvm', 'qemu']
        for indicator in vm_indicators:
            if indicator in user_agent:
                risk_factors.append(0.4)
                break
        
        # Check for inconsistent screen resolution
        resolution = device_data.get('screen_resolution', '')
        if resolution and 'x' in resolution:
            width, height = map(int, resolution.split('x'))
            # Unusual resolutions for real devices
            if width < 800 or height < 600:
                risk_factors.append(0.3)
        
        # Check for missing essential attributes
        essential_attrs = ['user_agent', 'platform', 'language']
        missing = sum(1 for attr in essential_attrs if attr not in device_data)
        if missing > 0:
            risk_factors.append(0.2 * missing)
        
        # Calculate final risk score
        if risk_factors:
            # Use weighted combination
            risk_score = min(1.0, sum(risk_factors) / len(risk_factors))
        else:
            risk_score = 0.1  # Default low risk
        
        logger.debug(f"Device {device_id} risk score: {risk_score:.2f}")
        
        return risk_score
    
    def track_device_activity(
        self,
        device_id: str,
        customer_id: str,
        timestamp: datetime
    ) -> Dict:
        """
        Track device usage patterns for fraud detection.
        
        Maintains history of which customers use which devices
        to detect account sharing or device farming.
        
        Args:
            device_id: Device identifier
            customer_id: Customer identifier
            timestamp: Activity timestamp
            
        Returns:
            Dict: Device activity statistics
        """
        activity_key = f"device:activity:{device_id}"
        
        # In production, this would store in Redis/DB
        # Here we'll simulate with in-memory cache
        if device_id not in self.fingerprint_cache:
            self.fingerprint_cache[device_id] = {
                'first_seen': timestamp,
                'last_seen': timestamp,
                'customers': set(),
                'activity_count': 0
            }
        
        cache = self.fingerprint_cache[device_id]
        cache['last_seen'] = timestamp
        cache['customers'].add(customer_id)
        cache['activity_count'] += 1
        
        return {
            'device_id': device_id,
            'first_seen': cache['first_seen'],
            'last_seen': cache['last_seen'],
            'unique_customers': len(cache['customers']),
            'total_activities': cache['activity_count'],
            'is_multi_customer': len(cache['customers']) > 1
        }
    
    def detect_device_anomalies(self, device_id: str, current_data: Dict) -> List[str]:
        """
        Detect anomalies in device behavior.
        
        Identifies suspicious patterns like:
        - Sudden changes in device characteristics
        - Device used from multiple locations
        - Excessive activity velocity
        
        Args:
            device_id: Device identifier
            current_data: Current device characteristics
            
        Returns:
            List[str]: List of detected anomalies
        """
        anomalies = []
        
        # Get historical data
        history = self.fingerprint_cache.get(device_id, {})
        historical_fingerprint = history.get('fingerprint')
        
        # Check for fingerprint changes
        current_fingerprint = self.generate_fingerprint(current_data)
        if historical_fingerprint and historical_fingerprint != current_fingerprint:
            anomalies.append('FINGERPRINT_CHANGE')
        
        # Check for location changes (if available)
        current_location = current_data.get('location')
        historical_location = history.get('last_location')
        
        if current_location and historical_location:
            if current_location != historical_location:
                anomalies.append('LOCATION_CHANGE')
        
        # Check activity velocity
        current_time = datetime.now()
        last_seen = history.get('last_seen')
        
        if last_seen:
            time_diff = (current_time - last_seen).total_seconds()
            if time_diff < 60:  # Less than a minute
                anomalies.append('HIGH_VELOCITY')
        
        # Check for inconsistent attributes
        if 'platform' in current_data and 'user_agent' in current_data:
            platform = current_data['platform'].lower()
            user_agent = current_data['user_agent'].lower()
            
            # Windows should have certain user agent patterns
            if 'windows' in platform and 'mac' in user_agent:
                anomalies.append('PLATFORM_MISMATCH')
        
        return anomalies


class APIClientFactory:
    """
    Factory class for creating and managing API clients.
    
    Provides centralized creation and configuration of all API clients
    with dependency injection and lifecycle management.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize factory with configuration.
        
        Args:
            config: Configuration dictionary with API settings
        """
        self.config = config
        self.clients = {}
        
        logger.info("API client factory initialized")
    
    def get_banking_client(self) -> BankingAPIClient:
        """
        Get or create banking API client.
        
        Returns:
            BankingAPIClient: Configured banking client
        """
        if 'banking' not in self.clients:
            banking_config = self.config.get('banking', {})
            self.clients['banking'] = BankingAPIClient(
                base_url=banking_config.get('base_url'),
                api_key=banking_config.get('api_key'),
                api_secret=banking_config.get('api_secret'),
                institution_id=banking_config.get('institution_id'),
                timeout=banking_config.get('timeout', 30)
            )
        
        return self.clients['banking']
    
    def get_fraud_intelligence_client(self) -> FraudIntelligenceClient:
        """
        Get or create fraud intelligence client.
        
        Returns:
            FraudIntelligenceClient: Configured fraud intelligence client
        """
        if 'fraud_intel' not in self.clients:
            fraud_config = self.config.get('fraud_intelligence', {})
            self.clients['fraud_intel'] = FraudIntelligenceClient(
                base_url=fraud_config.get('base_url'),
                api_key=fraud_config.get('api_key'),
                cache_ttl=fraud_config.get('cache_ttl', 3600)
            )
        
        return self.clients['fraud_intel']
    
    def get_merchant_client(self) -> MerchantDataClient:
        """
        Get or create merchant data client.
        
        Returns:
            MerchantDataClient: Configured merchant client
        """
        if 'merchant' not in self.clients:
            merchant_config = self.config.get('merchant', {})
            self.clients['merchant'] = MerchantDataClient(
                base_url=merchant_config.get('base_url'),
                api_key=merchant_config.get('api_key')
            )
        
        return self.clients['merchant']
    
    def get_geoip_client(self) -> GeoIPClient:
        """
        Get or create GeoIP client.
        
        Returns:
            GeoIPClient: Configured GeoIP client
        """
        if 'geoip' not in self.clients:
            geoip_config = self.config.get('geoip', {})
            self.clients['geoip'] = GeoIPClient(
                base_url=geoip_config.get('base_url'),
                api_key=geoip_config.get('api_key')
            )
        
        return self.clients['geoip']
    
    def get_device_fingerprint_client(self) -> DeviceFingerprintClient:
        """
        Get or create device fingerprint client.
        
        Returns:
            DeviceFingerprintClient: Configured device fingerprint client
        """
        if 'device' not in self.clients:
            self.clients['device'] = DeviceFingerprintClient()
        
        return self.clients['device']
    
    def close_all(self):
        """Close all client sessions."""
        for name, client in self.clients.items():
            try:
                if hasattr(client, 'close'):
                    client.close()
                logger.info(f"Closed client: {name}")
            except Exception as e:
                logger.error(f"Error closing client {name}: {str(e)}")