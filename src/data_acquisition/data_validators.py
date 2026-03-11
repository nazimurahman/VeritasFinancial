"""
Data Validators Module for VeritasFinancial Banking Fraud Detection System

This module provides comprehensive data validation, schema enforcement,
and quality checks for all incoming data streams. It ensures data integrity
before processing and identifies data quality issues early in the pipeline.

Author: VeritasFinancial Data Science Team
Version: 1.0.0
"""

import json
import logging
import re
from datetime import datetime, date, timezone
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import ipaddress
from collections import defaultdict
import pandas as pd
import numpy as np
from pydantic import BaseModel, ValidationError, validator, Field, root_validator
from jsonschema import validate, ValidationError as JsonValidationError
import cerberus
import voluptuous as vol

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """
    Severity levels for validation errors.
    
    Attributes:
        CRITICAL: Errors that prevent any processing
        HIGH: Serious issues requiring immediate attention
        MEDIUM: Important issues to monitor
        LOW: Minor issues, can be logged and ignored
        INFO: Informational messages
    """
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class ValidationCategory(Enum):
    """
    Categories of validation checks.
    
    Attributes:
        SCHEMA: Schema compliance validation
        DATA_TYPE: Data type validation
        FORMAT: Format validation (email, phone, etc.)
        RANGE: Value range validation
        BUSINESS: Business rule validation
        REFERENTIAL: Referential integrity validation
        CONSISTENCY: Cross-field consistency validation
        STATISTICAL: Statistical outlier detection
        TEMPORAL: Temporal sequence validation
    """
    SCHEMA = "SCHEMA"
    DATA_TYPE = "DATA_TYPE"
    FORMAT = "FORMAT"
    RANGE = "RANGE"
    BUSINESS = "BUSINESS"
    REFERENTIAL = "REFERENTIAL"
    CONSISTENCY = "CONSISTENCY"
    STATISTICAL = "STATISTICAL"
    TEMPORAL = "TEMPORAL"


@dataclass
class ValidationResult:
    """
    Comprehensive validation result object.
    
    Attributes:
        is_valid: Overall validation status
        errors: List of validation errors
        warnings: List of validation warnings
        metrics: Validation metrics and statistics
        field_stats: Per-field validation statistics
        validation_time: Time taken for validation (ms)
        validator_name: Name of validator that produced result
    """
    is_valid: bool
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    field_stats: Dict[str, Any] = field(default_factory=dict)
    validation_time: float = 0.0
    validator_name: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary."""
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'metrics': self.metrics,
            'field_stats': self.field_stats,
            'validation_time': self.validation_time,
            'validator_name': self.validator_name
        }
    
    def add_error(self, field: str, message: str, severity: ValidationSeverity = ValidationSeverity.HIGH):
        """Add an error to validation result."""
        self.errors.append({
            'field': field,
            'message': message,
            'severity': severity.value,
            'timestamp': datetime.utcnow().isoformat()
        })
        self.is_valid = False
    
    def add_warning(self, field: str, message: str, severity: ValidationSeverity = ValidationSeverity.MEDIUM):
        """Add a warning to validation result."""
        self.warnings.append({
            'field': field,
            'message': message,
            'severity': severity.value,
            'timestamp': datetime.utcnow().isoformat()
        })


class TransactionSchema(BaseModel):
    """
    Pydantic model for transaction data validation.
    
    This model defines the expected schema for transaction data with
    comprehensive validation rules and type checking.
    
    Attributes:
        transaction_id: Unique transaction identifier (required)
        account_id: Customer account identifier (required)
        amount: Transaction amount (required, positive/negative based on type)
        currency: Three-letter currency code (required)
        transaction_type: Type of transaction (required)
        merchant_id: Merchant identifier (optional)
        merchant_category: MCC code or category (optional)
        timestamp: Transaction timestamp (required, ISO format)
        location: Location data dictionary (optional)
        device_id: Device identifier (optional)
        ip_address: IP address (optional)
        card_present: Whether card was physically present (default: False)
        is_international: Whether transaction is cross-border (default: False)
        channel: Transaction channel (default: "UNKNOWN")
        metadata: Additional metadata (optional)
    """
    
    # Required fields with validation
    transaction_id: str = Field(..., min_length=5, max_length=50, description="Unique transaction ID")
    account_id: str = Field(..., min_length=5, max_length=50, description="Customer account ID")
    amount: float = Field(..., description="Transaction amount")
    currency: str = Field(..., min_length=3, max_length=3, description="ISO currency code")
    transaction_type: str = Field(..., description="Type of transaction")
    timestamp: datetime = Field(..., description="Transaction timestamp")
    
    # Optional fields with defaults
    merchant_id: Optional[str] = Field(None, min_length=3, max_length=50)
    merchant_category: Optional[str] = Field(None, min_length=2, max_length=10)
    location: Optional[Dict[str, Any]] = Field(None)
    device_id: Optional[str] = Field(None, min_length=5, max_length=100)
    ip_address: Optional[str] = Field(None)
    card_present: bool = False
    is_international: bool = False
    channel: str = "UNKNOWN"
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    # Validators for individual fields
    @validator('transaction_id')
    def validate_transaction_id(cls, v):
        """Validate transaction ID format."""
        # Transaction ID should be alphanumeric with possible hyphens
        if not re.match(r'^[A-Za-z0-9\-_]+$', v):
            raise ValueError('Transaction ID must be alphanumeric with hyphens/underscores only')
        return v
    
    @validator('account_id')
    def validate_account_id(cls, v):
        """Validate account ID format."""
        # Account ID should be alphanumeric
        if not re.match(r'^[A-Za-z0-9]+$', v):
            raise ValueError('Account ID must be alphanumeric only')
        return v
    
    @validator('amount')
    def validate_amount(cls, v):
        """Validate transaction amount."""
        # Amount should not be zero
        if abs(v) < 0.01:
            raise ValueError('Transaction amount too small (minimum 0.01)')
        
        # Check for reasonable maximum (configurable)
        if abs(v) > 1000000:
            logger.warning(f"Large transaction amount detected: {v}")
        
        # Check for precision (max 2 decimal places for most currencies)
        if abs(v - round(v, 2)) > 0.0001:
            logger.warning(f"Unusual amount precision: {v}")
        
        return v
    
    @validator('currency')
    def validate_currency(cls, v):
        """Validate currency code."""
        # List of common currencies (should come from config)
        valid_currencies = {'USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'CNY', 'INR', 'BRL'}
        
        v = v.upper()
        if v not in valid_currencies:
            logger.warning(f"Unusual currency code: {v}")
        
        return v
    
    @validator('transaction_type')
    def validate_transaction_type(cls, v):
        """Validate transaction type."""
        valid_types = {'PURCHASE', 'WITHDRAWAL', 'DEPOSIT', 'TRANSFER', 'PAYMENT', 'REFUND', 'CHARGEBACK'}
        
        v = v.upper()
        if v not in valid_types:
            logger.warning(f"Unknown transaction type: {v}")
        
        return v
    
    @validator('ip_address')
    def validate_ip_address(cls, v):
        """Validate IP address format."""
        if v:
            try:
                # Validate IPv4 or IPv6
                ipaddress.ip_address(v)
            except ValueError:
                raise ValueError(f'Invalid IP address format: {v}')
        return v
    
    @validator('channel')
    def validate_channel(cls, v):
        """Validate transaction channel."""
        valid_channels = {'ONLINE', 'POS', 'ATM', 'MOBILE', 'UNKNOWN', 'API', 'BATCH'}
        
        v = v.upper()
        if v not in valid_channels:
            logger.warning(f"Unknown channel: {v}")
        
        return v
    
    @root_validator
    def validate_transaction_logic(cls, values):
        """Cross-field validation for transaction logic."""
        amount = values.get('amount')
        transaction_type = values.get('transaction_type')
        
        # Debit transactions should have negative or positive based on convention
        if transaction_type in {'WITHDRAWAL', 'PURCHASE'} and amount > 0:
            # Some systems use positive for all transactions, just warn
            logger.warning(f"Debit transaction with positive amount: {transaction_type} {amount}")
        
        # International transaction validation
        is_international = values.get('is_international', False)
        location = values.get('location')
        
        if is_international and location and location.get('country'):
            # Could validate country against customer's home country
            pass
        
        return values
    
    class Config:
        """Pydantic model configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        validate_assignment = True
        use_enum_values = True


class CustomerSchema(BaseModel):
    """
    Pydantic model for customer profile data validation.
    
    Attributes:
        customer_id: Unique customer identifier
        account_id: Primary account identifier
        name: Customer name (may be hashed for privacy)
        email: Email address
        phone: Phone number
        date_of_birth: Date of birth
        country_of_residence: Country code
        risk_rating: Internal risk rating (1-10)
        account_age_days: Days since account opening
        total_balance: Total account balance
        average_balance: Average balance over 30 days
        transaction_volume_30d: Transaction count in 30 days
        kyc_status: KYC verification status
        kyc_completed_date: KYC completion date
        watchlist_flag: Whether on any watchlist
        fraud_history: List of fraud incidents
        device_history: List of device IDs
        last_updated: Profile last updated timestamp
    """
    
    customer_id: str = Field(..., min_length=5, max_length=50)
    account_id: str = Field(..., min_length=5, max_length=50)
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    date_of_birth: Optional[date] = None
    country_of_residence: Optional[str] = Field(None, min_length=2, max_length=2)
    risk_rating: int = Field(5, ge=1, le=10)
    account_age_days: int = Field(0, ge=0)
    total_balance: float = Field(0.0, ge=0)
    average_balance: float = Field(0.0, ge=0)
    transaction_volume_30d: int = Field(0, ge=0)
    kyc_status: str = "PENDING"
    kyc_completed_date: Optional[datetime] = None
    watchlist_flag: bool = False
    fraud_history: Optional[List[Dict[str, Any]]] = None
    device_history: Optional[List[str]] = None
    last_updated: Optional[datetime] = None
    
    @validator('email')
    def validate_email(cls, v):
        """Validate email format if provided."""
        if v:
            # Basic email validation
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, v):
                raise ValueError(f'Invalid email format: {v}')
        return v
    
    @validator('phone')
    def validate_phone(cls, v):
        """Validate phone number format if provided."""
        if v:
            # Basic phone validation (can be enhanced per country)
            phone_pattern = r'^\+?[1-9][0-9]{7,14}$'
            if not re.match(phone_pattern, v.replace(' ', '').replace('-', '')):
                raise ValueError(f'Invalid phone format: {v}')
        return v
    
    @validator('country_of_residence')
    def validate_country(cls, v):
        """Validate country code."""
        if v:
            v = v.upper()
            # ISO 3166-1 alpha-2 country codes
            valid_countries = {'US', 'GB', 'CA', 'AU', 'DE', 'FR', 'JP', 'CN', 'IN', 'BR'}
            if v not in valid_countries:
                logger.warning(f"Unusual country code: {v}")
        return v


class DeviceSchema(BaseModel):
    """
    Pydantic model for device fingerprint data validation.
    
    Attributes:
        device_id: Unique device identifier
        device_type: Type of device
        os_name: Operating system name
        os_version: Operating system version
        browser_name: Browser name
        browser_version: Browser version
        screen_resolution: Screen resolution
        timezone: Device timezone
        language: Device language
        ip_address: IP address
        location_history: Historical locations
        first_seen: First appearance timestamp
        last_seen: Last appearance timestamp
        fraud_score: Device risk score (0-1)
        associated_accounts: Accounts using this device
    """
    
    device_id: str = Field(..., min_length=5, max_length=100)
    device_type: str = Field(..., description="MOBILE, TABLET, DESKTOP, UNKNOWN")
    os_name: Optional[str] = None
    os_version: Optional[str] = None
    browser_name: Optional[str] = None
    browser_version: Optional[str] = None
    screen_resolution: Optional[str] = None
    timezone: Optional[str] = None
    language: Optional[str] = Field(None, min_length=2, max_length=5)
    ip_address: Optional[str] = None
    location_history: Optional[List[Dict[str, Any]]] = None
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    fraud_score: float = Field(0.0, ge=0.0, le=1.0)
    associated_accounts: Optional[List[str]] = None
    
    @validator('device_type')
    def validate_device_type(cls, v):
        """Validate device type."""
        valid_types = {'MOBILE', 'TABLET', 'DESKTOP', 'UNKNOWN', 'SERVER', 'IOT'}
        v = v.upper()
        if v not in valid_types:
            raise ValueError(f'Invalid device type: {v}')
        return v


class DataValidator:
    """
    Comprehensive data validator for all incoming data streams.
    
    This class provides a unified interface for validating different
    types of data (transactions, customers, devices) with configurable
    validation rules and severity levels.
    
    Features:
    - Multiple validation backends (Pydantic, JSON Schema, Cerberus)
    - Configurable validation rules per data type
    - Statistical anomaly detection
    - Referential integrity checks
    - Cross-field consistency validation
    - Detailed validation metrics
    - Batch validation support
    - Custom validation rules
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data validator with configuration.
        
        Args:
            config: Configuration dictionary with validation rules
        """
        self.config = config or self._default_config()
        self.validation_stats = defaultdict(lambda: {'total': 0, 'passed': 0, 'failed': 0})
        self.schemas = self._load_schemas()
        
    def _default_config(self) -> Dict[str, Any]:
        """Provide default validation configuration."""
        return {
            'transaction': {
                'schema_version': '1.0',
                'required_fields': ['transaction_id', 'account_id', 'amount', 'timestamp'],
                'field_validation': {
                    'amount': {'min': 0.01, 'max': 1000000, 'precision': 2},
                    'currency': {'allowed': ['USD', 'EUR', 'GBP', 'JPY', 'CAD']}
                },
                'severity_levels': {
                    'missing_field': 'HIGH',
                    'invalid_format': 'MEDIUM',
                    'out_of_range': 'MEDIUM'
                }
            },
            'customer': {
                'schema_version': '1.0',
                'required_fields': ['customer_id', 'account_id'],
                'field_validation': {
                    'risk_rating': {'min': 1, 'max': 10},
                    'account_age_days': {'min': 0, 'max': 36500}
                }
            },
            'device': {
                'schema_version': '1.0',
                'required_fields': ['device_id', 'device_type']
            }
        }
    
    def _load_schemas(self) -> Dict[str, Any]:
        """Load validation schemas from configuration."""
        return {
            'transaction': self._create_transaction_schema(),
            'customer': self._create_customer_schema(),
            'device': self._create_device_schema()
        }
    
    def _create_transaction_schema(self) -> Dict[str, Any]:
        """Create JSON schema for transaction validation."""
        return {
            'type': 'object',
            'properties': {
                'transaction_id': {'type': 'string', 'pattern': '^[A-Za-z0-9\\-_]+$'},
                'account_id': {'type': 'string', 'pattern': '^[A-Za-z0-9]+$'},
                'amount': {'type': 'number', 'minimum': 0.01, 'maximum': 1000000},
                'currency': {'type': 'string', 'pattern': '^[A-Z]{3}$'},
                'transaction_type': {'type': 'string', 'enum': ['PURCHASE', 'WITHDRAWAL', 'DEPOSIT', 'TRANSFER']},
                'timestamp': {'type': 'string', 'format': 'date-time'},
                'card_present': {'type': 'boolean'},
                'is_international': {'type': 'boolean'},
                'channel': {'type': 'string', 'enum': ['ONLINE', 'POS', 'ATM', 'MOBILE']}
            },
            'required': ['transaction_id', 'account_id', 'amount', 'timestamp']
        }
    
    def _create_customer_schema(self) -> Dict[str, Any]:
        """Create JSON schema for customer validation."""
        return {
            'type': 'object',
            'properties': {
                'customer_id': {'type': 'string', 'pattern': '^[A-Za-z0-9]+$'},
                'account_id': {'type': 'string', 'pattern': '^[A-Za-z0-9]+$'},
                'email': {'type': 'string', 'format': 'email'},
                'phone': {'type': 'string', 'pattern': '^\\+?[0-9]{7,15}$'},
                'risk_rating': {'type': 'integer', 'minimum': 1, 'maximum': 10},
                'kyc_status': {'type': 'string', 'enum': ['PENDING', 'VERIFIED', 'REJECTED']}
            },
            'required': ['customer_id', 'account_id']
        }
    
    def _create_device_schema(self) -> Dict[str, Any]:
        """Create JSON schema for device validation."""
        return {
            'type': 'object',
            'properties': {
                'device_id': {'type': 'string', 'minLength': 5},
                'device_type': {'type': 'string', 'enum': ['MOBILE', 'TABLET', 'DESKTOP', 'UNKNOWN']},
                'fraud_score': {'type': 'number', 'minimum': 0, 'maximum': 1}
            },
            'required': ['device_id', 'device_type']
        }
    
    def validate_transaction(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        strict: bool = True,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate transaction data against schema and business rules.
        
        Args:
            data: Single transaction or list of transactions
            strict: Whether to perform strict validation (fail on warnings)
            context: Additional context for validation (e.g., customer history)
            
        Returns:
            ValidationResult with detailed validation information
        """
        start_time = datetime.utcnow()
        result = ValidationResult(is_valid=True, validator_name='transaction_validator')
        
        # Handle both single and batch validation
        if isinstance(data, list):
            return self._validate_transaction_batch(data, strict, context)
        
        try:
            # 1. Schema validation using Pydantic
            try:
                transaction = TransactionSchema(**data)
                result.metrics['schema_validation'] = 'passed'
            except ValidationError as e:
                result.is_valid = False
                for error in e.errors():
                    result.add_error(
                        field=error['loc'][0] if error['loc'] else 'unknown',
                        message=error['msg'],
                        severity=ValidationSeverity.HIGH
                    )
                result.metrics['schema_validation'] = 'failed'
                return result
            
            # 2. Business rule validation
            business_validation = self._validate_transaction_business_rules(transaction, context)
            if not business_validation['is_valid']:
                result.is_valid = False
                for error in business_validation['errors']:
                    result.add_error(**error)
            
            # 3. Statistical outlier detection
            if context and context.get('customer_history'):
                outlier_check = self._detect_statistical_outliers(transaction, context['customer_history'])
                if outlier_check['is_outlier']:
                    result.add_warning(
                        field='amount',
                        message=f"Statistical outlier detected: {outlier_check['reason']}",
                        severity=ValidationSeverity.MEDIUM
                    )
            
            # 4. Temporal consistency check
            if context and context.get('last_transaction_time'):
                temporal_check = self._check_temporal_consistency(
                    transaction.timestamp,
                    context['last_transaction_time']
                )
                if not temporal_check['is_valid']:
                    result.add_warning(
                        field='timestamp',
                        message=temporal_check['message'],
                        severity=ValidationSeverity.MEDIUM
                    )
            
            # 5. Referential integrity checks
            if context and context.get('known_devices'):
                if transaction.device_id and transaction.device_id not in context['known_devices']:
                    result.add_warning(
                        field='device_id',
                        message="Unknown device for this customer",
                        severity=ValidationSeverity.MEDIUM
                    )
            
            # Update statistics
            self._update_validation_stats('transaction', result.is_valid)
            
        except Exception as e:
            logger.error(f"Unexpected error during transaction validation: {e}")
            result.is_valid = False
            result.add_error(
                field='_system',
                message=f"Validation system error: {str(e)}",
                severity=ValidationSeverity.CRITICAL
            )
        
        # Calculate validation time
        result.validation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return result
    
    def _validate_transaction_batch(
        self,
        transactions: List[Dict[str, Any]],
        strict: bool,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate a batch of transactions.
        
        This method performs batch validation with aggregate statistics
        and identifies systematic issues across transactions.
        """
        start_time = datetime.utcnow()
        result = ValidationResult(is_valid=True, validator_name='transaction_batch_validator')
        
        batch_results = []
        field_errors = defaultdict(int)
        
        for idx, transaction in enumerate(transactions):
            tx_result = self.validate_transaction(transaction, strict, context)
            batch_results.append(tx_result)
            
            # Aggregate errors
            for error in tx_result.errors:
                field_errors[error['field']] += 1
            
            if not tx_result.is_valid:
                result.is_valid = False
        
        # Calculate batch metrics
        result.metrics = {
            'total_transactions': len(transactions),
            'valid_transactions': sum(1 for r in batch_results if r.is_valid),
            'invalid_transactions': sum(1 for r in batch_results if not r.is_valid),
            'field_error_counts': dict(field_errors),
            'error_rate': len([r for r in batch_results if not r.is_valid]) / len(transactions) if transactions else 0
        }
        
        # Add warnings for systematic issues
        for field, count in field_errors.items():
            if count > len(transactions) * 0.1:  # More than 10% have same field error
                result.add_warning(
                    field=field,
                    message=f"Systematic validation issue: {count} transactions affected",
                    severity=ValidationSeverity.HIGH
                )
        
        result.validation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return result
    
    def _validate_transaction_business_rules(
        self,
        transaction: TransactionSchema,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Apply business rules to transaction validation.
        
        Args:
            transaction: Validated transaction object
            context: Business context (customer profile, limits, etc.)
            
        Returns:
            Dictionary with validation results
        """
        result = {'is_valid': True, 'errors': []}
        
        # Rule 1: Check against daily limits
        if context and context.get('daily_limit'):
            if abs(transaction.amount) > context['daily_limit']:
                result['is_valid'] = False
                result['errors'].append({
                    'field': 'amount',
                    'message': f"Amount exceeds daily limit of {context['daily_limit']}",
                    'severity': ValidationSeverity.HIGH
                })
        
        # Rule 2: Velocity check (using context counters)
        if context and context.get('hourly_transactions', 0) > 10:
            result['is_valid'] = False
            result['errors'].append({
                'field': '_velocity',
                'message': "Transaction velocity exceeds threshold",
                'severity': ValidationSeverity.HIGH
            })
        
        # Rule 3: Card-present for high-value transactions
        if abs(transaction.amount) > 10000 and not transaction.card_present and transaction.channel != 'ATM':
            result['errors'].append({
                'field': 'card_present',
                'message': "High-value transaction without physical card",
                'severity': ValidationSeverity.MEDIUM
            })
        
        # Rule 4: Weekend/holiday patterns
        if transaction.timestamp.weekday() >= 5 and abs(transaction.amount) > 5000:
            result['errors'].append({
                'field': 'timestamp',
                'message': "Large weekend transaction",
                'severity': ValidationSeverity.LOW
            })
        
        return result
    
    def _detect_statistical_outliers(
        self,
        transaction: TransactionSchema,
        history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Detect statistical outliers in transaction data.
        
        Args:
            transaction: Current transaction
            history: Historical transaction data for this customer
            
        Returns:
            Dictionary with outlier detection results
        """
        result = {'is_outlier': False, 'reason': None, 'z_score': 0}
        
        if not history:
            return result
        
        # Extract amounts from history
        historical_amounts = [abs(t.get('amount', 0)) for t in history if t.get('amount')]
        
        if len(historical_amounts) < 5:
            return result
        
        # Calculate statistics
        mean_amount = np.mean(historical_amounts)
        std_amount = np.std(historical_amounts)
        
        if std_amount == 0:
            return result
        
        # Calculate z-score for current transaction
        current_amount = abs(transaction.amount)
        z_score = (current_amount - mean_amount) / std_amount
        
        # Check if outlier (z-score > 3)
        if abs(z_score) > 3:
            result['is_outlier'] = True
            result['reason'] = f"Transaction amount is {z_score:.2f} standard deviations from mean"
            result['z_score'] = z_score
        
        return result
    
    def _check_temporal_consistency(
        self,
        current_time: datetime,
        last_time: Optional[datetime]
    ) -> Dict[str, Any]:
        """
        Check temporal consistency of transactions.
        
        Args:
            current_time: Current transaction timestamp
            last_time: Last transaction timestamp
            
        Returns:
            Dictionary with temporal consistency check results
        """
        result = {'is_valid': True, 'message': ''}
        
        if last_time:
            # Check if transaction is in the future
            if current_time > datetime.now(timezone.utc):
                result['is_valid'] = False
                result['message'] = "Transaction timestamp is in the future"
            
            # Check for extremely fast consecutive transactions
            time_diff = (current_time - last_time).total_seconds()
            if time_diff < 0:
                result['is_valid'] = False
                result['message'] = "Transaction timestamps out of order"
            elif time_diff < 1:  # Less than 1 second between transactions
                result['is_valid'] = False
                result['message'] = "Impossibly fast consecutive transactions"
        
        return result
    
    def validate_customer(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        strict: bool = True
    ) -> ValidationResult:
        """
        Validate customer profile data.
        
        Args:
            data: Single customer or list of customers
            strict: Whether to perform strict validation
            
        Returns:
            ValidationResult with detailed validation information
        """
        start_time = datetime.utcnow()
        result = ValidationResult(is_valid=True, validator_name='customer_validator')
        
        # Handle batch validation
        if isinstance(data, list):
            return self._validate_customer_batch(data, strict)
        
        try:
            # Validate using Pydantic
            customer = CustomerSchema(**data)
            
            # Additional business rule validation
            if customer.account_age_days < 0:
                result.add_error('account_age_days', 'Account age cannot be negative')
            
            if customer.kyc_status == 'VERIFIED' and not customer.kyc_completed_date:
                result.add_error('kyc_completed_date', 'KYC verified but completion date missing')
            
            # Update statistics
            self._update_validation_stats('customer', result.is_valid)
            
        except ValidationError as e:
            result.is_valid = False
            for error in e.errors():
                result.add_error(
                    field=error['loc'][0] if error['loc'] else 'unknown',
                    message=error['msg']
                )
        except Exception as e:
            result.is_valid = False
            result.add_error('_system', f"Validation error: {str(e)}")
        
        result.validation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return result
    
    def _validate_customer_batch(
        self,
        customers: List[Dict[str, Any]],
        strict: bool
    ) -> ValidationResult:
        """Validate a batch of customer records."""
        result = ValidationResult(is_valid=True, validator_name='customer_batch_validator')
        
        valid_count = 0
        for customer in customers:
            customer_result = self.validate_customer(customer, strict)
            if customer_result.is_valid:
                valid_count += 1
            else:
                result.is_valid = False
                result.errors.extend(customer_result.errors)
        
        result.metrics = {
            'total': len(customers),
            'valid': valid_count,
            'invalid': len(customers) - valid_count,
            'validity_rate': valid_count / len(customers) if customers else 0
        }
        
        return result
    
    def validate_device(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        strict: bool = True
    ) -> ValidationResult:
        """
        Validate device fingerprint data.
        
        Args:
            data: Single device or list of devices
            strict: Whether to perform strict validation
            
        Returns:
            ValidationResult with detailed validation information
        """
        start_time = datetime.utcnow()
        result = ValidationResult(is_valid=True, validator_name='device_validator')
        
        # Handle batch validation
        if isinstance(data, list):
            return self._validate_device_batch(data, strict)
        
        try:
            # Validate using Pydantic
            device = DeviceSchema(**data)
            
            # Additional validation
            if device.fraud_score < 0 or device.fraud_score > 1:
                result.add_error('fraud_score', 'Fraud score must be between 0 and 1')
            
            # Check IP address if provided
            if device.ip_address:
                try:
                    ipaddress.ip_address(device.ip_address)
                except ValueError:
                    result.add_error('ip_address', f'Invalid IP address: {device.ip_address}')
            
            # Update statistics
            self._update_validation_stats('device', result.is_valid)
            
        except ValidationError as e:
            result.is_valid = False
            for error in e.errors():
                result.add_error(
                    field=error['loc'][0] if error['loc'] else 'unknown',
                    message=error['msg']
                )
        except Exception as e:
            result.is_valid = False
            result.add_error('_system', f"Validation error: {str(e)}")
        
        result.validation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return result
    
    def _validate_device_batch(
        self,
        devices: List[Dict[str, Any]],
        strict: bool
    ) -> ValidationResult:
        """Validate a batch of device records."""
        result = ValidationResult(is_valid=True, validator_name='device_batch_validator')
        
        valid_count = 0
        for device in devices:
            device_result = self.validate_device(device, strict)
            if device_result.is_valid:
                valid_count += 1
            else:
                result.is_valid = False
                result.errors.extend(device_result.errors)
        
        result.metrics = {
            'total': len(devices),
            'valid': valid_count,
            'invalid': len(devices) - valid_count,
            'validity_rate': valid_count / len(devices) if devices else 0
        }
        
        return result
    
    def _update_validation_stats(self, data_type: str, is_valid: bool):
        """Update validation statistics."""
        self.validation_stats[data_type]['total'] += 1
        if is_valid:
            self.validation_stats[data_type]['passed'] += 1
        else:
            self.validation_stats[data_type]['failed'] += 1
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        stats = {}
        for data_type, counts in self.validation_stats.items():
            total = counts['total']
            if total > 0:
                stats[data_type] = {
                    **counts,
                    'pass_rate': counts['passed'] / total,
                    'fail_rate': counts['failed'] / total
                }
            else:
                stats[data_type] = counts
        return stats
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        return {
            'statistics': self.get_validation_stats(),
            'config': self.config,
            'schemas': self.schemas,
            'timestamp': datetime.utcnow().isoformat()
        }


class SchemaValidator:
    """
    JSON Schema validator for data validation.
    
    Provides schema-based validation using JSON Schema standard.
    """
    
    def __init__(self, schema: Dict[str, Any]):
        """
        Initialize schema validator.
        
        Args:
            schema: JSON Schema dictionary
        """
        self.schema = schema
        
    def validate(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate data against schema.
        
        Args:
            data: Data to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        try:
            validate(instance=data, schema=self.schema)
            return True, []
        except JsonValidationError as e:
            return False, [str(e)]


class CerberusValidator:
    """
    Cerberus-based validator for flexible validation rules.
    
    Provides more flexible validation than JSON Schema with custom rules.
    """
    
    def __init__(self, schema: Dict[str, Any]):
        """
        Initialize Cerberus validator.
        
        Args:
            schema: Cerberus schema dictionary
        """
        self.validator = cerberus.Validator(schema)
        
    def validate(self, data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate data using Cerberus.
        
        Args:
            data: Data to validate
            
        Returns:
            Tuple of (is_valid, errors_dict)
        """
        is_valid = self.validator.validate(data)
        return is_valid, self.validator.errors


class VoluptuousValidator:
    """
    Voluptuous-based validator for strict validation.
    
    Provides strict validation with custom validation functions.
    """
    
    def __init__(self, schema: vol.Schema):
        """
        Initialize Voluptuous validator.
        
        Args:
            schema: Voluptuous schema
        """
        self.schema = schema
        
    def validate(self, data: Dict[str, Any]) -> Tuple[bool, Optional[Exception]]:
        """
        Validate data using Voluptuous.
        
        Args:
            data: Data to validate
            
        Returns:
            Tuple of (is_valid, error_if_any)
        """
        try:
            self.schema(data)
            return True, None
        except vol.Invalid as e:
            return False, e


# Example usage and testing
if __name__ == "__main__":
    # Example transaction data
    sample_transaction = {
        'transaction_id': 'TXN123456789',
        'account_id': 'ACC987654321',
        'amount': 1500.00,
        'currency': 'USD',
        'transaction_type': 'PURCHASE',
        'timestamp': datetime.utcnow().isoformat(),
        'merchant_id': 'MERCHANT001',
        'merchant_category': 'ELECTRONICS',
        'device_id': 'DEVICE123',
        'ip_address': '192.168.1.1',
        'card_present': False,
        'is_international': False,
        'channel': 'ONLINE'
    }
    
    # Initialize validator
    validator = DataValidator()
    
    # Validate transaction
    result = validator.validate_transaction(sample_transaction)
    
    print(f"Validation result: {'VALID' if result.is_valid else 'INVALID'}")
    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"  - {error['field']}: {error['message']}")
    
    # Get validation statistics
    stats = validator.get_validation_stats()
    print(f"\nValidation stats: {stats}")