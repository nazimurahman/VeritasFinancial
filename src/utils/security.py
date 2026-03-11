"""
Security Module for VeritasFinancial Banking Fraud Detection System

This module provides comprehensive security utilities for:
- Data encryption and decryption (AES-256, RSA)
- Secure key management
- PII (Personally Identifiable Information) masking
- Access control and authentication
- Audit logging
- Secure configuration handling
- Input validation and sanitization
- Compliance with banking regulations (PCI-DSS, GDPR)

Author: VeritasFinancial Data Science Team
Version: 1.0.0
"""

import hashlib
import hmac
import secrets
import base64
import os
import re
from typing import Optional, Union, Dict, List, Any, Tuple
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import bcrypt
import argon2
from argon2 import PasswordHasher
import pandas as pd
import numpy as np
import ipaddress
import uuid

# Configure logging
logger = logging.getLogger(__name__)


class SecurityManager:
    """
    Main security manager for VeritasFinancial.
    
    Handles all security-related operations including:
    - Encryption/Decryption
    - Key management
    - PII masking
    - Access control
    - Audit logging
    - Compliance checks
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 encryption_key: Optional[bytes] = None,
                 enable_audit: bool = True,
                 compliance_mode: str = 'strict'):
        """
        Initialize security manager.
        
        Args:
            config_path: Path to security configuration file
            encryption_key: Master encryption key (if None, generate or load)
            enable_audit: Enable audit logging
            compliance_mode: 'strict' or 'relaxed' for compliance checks
        """
        self.enable_audit = enable_audit
        self.compliance_mode = compliance_mode
        
        # Load configuration
        self.config = self._load_config(config_path) if config_path else {}
        
        # Initialize encryption
        self.master_key = encryption_key or self._load_or_create_master_key()
        self.fernet = Fernet(base64.urlsafe_b64encode(self.master_key[:32]))
        
        # Initialize password hasher
        self.ph = PasswordHasher(
            time_cost=2,      # Number of iterations
            memory_cost=102400, # Memory usage in KB
            parallelism=8,     # Number of parallel threads
            hash_len=32,       # Length of hash in bytes
            salt_len=16        # Length of salt in bytes
        )
        
        # PII patterns for masking
        self.pii_patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'),
            'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            'credit_card': re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
            'ip_address': re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
            'account_number': re.compile(r'\b\d{10,12}\b')
        }
        
        # Audit log
        self.audit_log = []
        
        logger.info("SecurityManager initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load security configuration from file."""
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.json'):
                    return json.load(f)
                elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    import yaml
                    return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            return {}
    
    def _load_or_create_master_key(self) -> bytes:
        """Load existing master key or create new one."""
        key_path = Path.home() / '.veritasfinancial' / 'master.key'
        
        if key_path.exists():
            with open(key_path, 'rb') as f:
                return f.read()
        else:
            # Generate new key
            key = secrets.token_bytes(32)
            key_path.parent.mkdir(parents=True, exist_ok=True)
            key_path.write_bytes(key)
            logger.info(f"New master key created at {key_path}")
            return key
    
    def encrypt_data(self, data: Union[str, bytes, Dict, Any]) -> bytes:
        """
        Encrypt sensitive data.
        
        Args:
            data: Data to encrypt (string, bytes, or JSON-serializable object)
            
        Returns:
            Encrypted bytes
        """
        # Convert to bytes if needed
        if isinstance(data, dict):
            data = json.dumps(data).encode('utf-8')
        elif isinstance(data, str):
            data = data.encode('utf-8')
        elif not isinstance(data, bytes):
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Encrypt with Fernet (symmetric encryption)
        encrypted = self.fernet.encrypt(data)
        
        # Log audit
        self._audit_log('ENCRYPT', f"Encrypted {len(data)} bytes")
        
        return encrypted
    
    def decrypt_data(self, encrypted_data: bytes) -> Union[bytes, Dict]:
        """
        Decrypt encrypted data.
        
        Args:
            encrypted_data: Encrypted bytes
            
        Returns:
            Decrypted data (bytes or dict if JSON)
        """
        try:
            decrypted = self.fernet.decrypt(encrypted_data)
            
            # Try to parse as JSON
            try:
                return json.loads(decrypted.decode('utf-8'))
            except:
                return decrypted
                
        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}")
            raise SecurityError(f"Decryption failed: {str(e)}")
    
    def hash_password(self, password: str) -> str:
        """
        Hash password using Argon2id (recommended for password storage).
        
        Argon2id is the winner of the Password Hashing Competition
        and is resistant to GPU cracking attacks.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password string
        """
        try:
            # Add pepper (secret key) from config
            pepper = self.config.get('password_pepper', '')
            password_with_pepper = password + pepper
            
            # Hash with Argon2id
            hash_str = self.ph.hash(password_with_pepper)
            
            self._audit_log('PASSWORD_HASH', "Password hashed")
            return hash_str
            
        except Exception as e:
            logger.error(f"Password hashing failed: {str(e)}")
            raise SecurityError(f"Password hashing failed: {str(e)}")
    
    def verify_password(self, password: str, hash_str: str) -> bool:
        """
        Verify password against hash.
        
        Args:
            password: Plain text password to verify
            hash_str: Stored hash string
            
        Returns:
            True if password matches
        """
        try:
            pepper = self.config.get('password_pepper', '')
            password_with_pepper = password + pepper
            
            # Verify with Argon2id
            self.ph.verify(hash_str, password_with_pepper)
            
            self._audit_log('PASSWORD_VERIFY', "Password verified successfully")
            return True
            
        except argon2.exceptions.VerifyMismatchError:
            self._audit_log('PASSWORD_VERIFY_FAIL', "Password verification failed")
            return False
        except Exception as e:
            logger.error(f"Password verification error: {str(e)}")
            return False
    
    def generate_jwt_token(self, 
                           payload: Dict,
                           expiry_hours: int = 24) -> str:
        """
        Generate JWT token for authentication.
        
        Args:
            payload: Token payload (user_id, roles, etc.)
            expiry_hours: Token expiry time in hours
            
        Returns:
            JWT token string
        """
        # Add standard claims
        payload.update({
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(hours=expiry_hours),
            'jti': str(uuid.uuid4())
        })
        
        # Sign token
        secret = self.config.get('jwt_secret', self.master_key.hex())
        token = jwt.encode(payload, secret, algorithm='HS256')
        
        self._audit_log('JWT_GENERATE', f"JWT token generated for user {payload.get('user_id')}")
        return token
    
    def verify_jwt_token(self, token: str) -> Dict:
        """
        Verify and decode JWT token.
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded payload
        """
        try:
            secret = self.config.get('jwt_secret', self.master_key.hex())
            payload = jwt.decode(token, secret, algorithms=['HS256'])
            
            self._audit_log('JWT_VERIFY', f"JWT token verified for user {payload.get('user_id')}")
            return payload
            
        except jwt.ExpiredSignatureError:
            raise SecurityError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise SecurityError(f"Invalid token: {str(e)}")
    
    def mask_pii(self, 
                 data: Union[str, Dict, pd.DataFrame],
                 mask_char: str = '*',
                 preserve_length: bool = True) -> Union[str, Dict, pd.DataFrame]:
        """
        Mask Personally Identifiable Information (PII).
        
        Supports:
        - Email addresses
        - Phone numbers
        - Social Security Numbers
        - Credit card numbers
        - IP addresses
        - Account numbers
        
        Args:
            data: Data containing PII
            mask_char: Character to use for masking
            preserve_length: Keep original length when masking
            
        Returns:
            Data with PII masked
        """
        if isinstance(data, str):
            return self._mask_string_pii(data, mask_char, preserve_length)
        
        elif isinstance(data, dict):
            return self._mask_dict_pii(data, mask_char, preserve_length)
        
        elif isinstance(data, pd.DataFrame):
            return self._mask_dataframe_pii(data, mask_char, preserve_length)
        
        else:
            return data
    
    def _mask_string_pii(self, text: str, mask_char: str, preserve_length: bool) -> str:
        """Mask PII in string."""
        masked_text = text
        
        for pii_type, pattern in self.pii_patterns.items():
            def mask_match(match):
                value = match.group()
                if preserve_length:
                    return mask_char * len(value)
                else:
                    # Show first/last characters
                    if pii_type == 'email':
                        parts = value.split('@')
                        if len(parts) == 2:
                            return f"{mask_char * len(parts[0])}@{mask_char * len(parts[1])}"
                    elif pii_type == 'credit_card':
                        return value[:4] + mask_char * (len(value) - 8) + value[-4:]
                    elif pii_type == 'ssn':
                        return '***-**-' + value[-4:]
                    else:
                        return mask_char * len(value)
            
            masked_text = pattern.sub(mask_match, masked_text)
        
        return masked_text
    
    def _mask_dict_pii(self, data: Dict, mask_char: str, preserve_length: bool) -> Dict:
        """Mask PII in dictionary."""
        masked = {}
        
        for key, value in data.items():
            # Check if key indicates PII
            if any(pii_key in key.lower() for pii_key in 
                   ['email', 'phone', 'ssn', 'credit', 'account', 'ip']):
                if isinstance(value, str):
                    masked[key] = self._mask_string_pii(value, mask_char, preserve_length)
                else:
                    masked[key] = mask_char * 8
            else:
                if isinstance(value, dict):
                    masked[key] = self._mask_dict_pii(value, mask_char, preserve_length)
                elif isinstance(value, (list, tuple)):
                    masked[key] = [self._mask_string_pii(str(v), mask_char, preserve_length) 
                                  for v in value]
                else:
                    masked[key] = value
        
        return masked
    
    def _mask_dataframe_pii(self, df: pd.DataFrame, mask_char: str, preserve_length: bool) -> pd.DataFrame:
        """Mask PII in DataFrame."""
        masked_df = df.copy()
        
        for col in df.columns:
            # Check if column contains PII
            if any(pii_key in col.lower() for pii_key in 
                   ['email', 'phone', 'ssn', 'credit', 'account', 'ip']):
                masked_df[col] = df[col].astype(str).apply(
                    lambda x: self._mask_string_pii(x, mask_char, preserve_length)
                )
        
        return masked_df
    
    def generate_api_key(self) -> Tuple[str, str]:
        """
        Generate API key and secret.
        
        Returns:
            Tuple of (api_key, api_secret)
        """
        api_key = secrets.token_urlsafe(32)
        api_secret = secrets.token_urlsafe(64)
        
        # Store hash of secret for verification
        secret_hash = self.hash_password(api_secret)
        
        self._audit_log('API_KEY_GENERATE', f"API key generated: {api_key[:8]}...")
        
        return api_key, api_secret
    
    def verify_api_key(self, api_key: str, api_secret: str, stored_hash: str) -> bool:
        """
        Verify API key and secret.
        
        Args:
            api_key: API key to verify
            api_secret: API secret to verify
            stored_hash: Stored hash of secret
            
        Returns:
            True if valid
        """
        return self.verify_password(api_secret, stored_hash)
    
    def sanitize_input(self, 
                       input_str: str,
                       allowed_chars: Optional[str] = None,
                       max_length: int = 1000) -> str:
        """
        Sanitize user input to prevent injection attacks.
        
        Args:
            input_str: Input string to sanitize
            allowed_chars: Regex pattern of allowed characters
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
        """
        if not input_str:
            return input_str
        
        # Truncate
        if len(input_str) > max_length:
            input_str = input_str[:max_length]
        
        # Remove control characters
        input_str = ''.join(char for char in input_str if ord(char) >= 32)
        
        # Apply allowed characters filter
        if allowed_chars:
            pattern = f'[^{re.escape(allowed_chars)}]'
            input_str = re.sub(pattern, '', input_str)
        
        # Escape HTML
        import html
        input_str = html.escape(input_str)
        
        return input_str
    
    def validate_ip(self, ip_address: str) -> bool:
        """Validate IP address format."""
        try:
            ipaddress.ip_address(ip_address)
            return True
        except:
            return False
    
    def validate_email(self, email: str) -> bool:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def generate_secure_random(self, length: int = 32) -> str:
        """
        Generate cryptographically secure random string.
        
        Args:
            length: Length of random string
            
        Returns:
            Random string
        """
        return secrets.token_urlsafe(length)
    
    def compute_hash(self, 
                     data: Union[str, bytes], 
                     algorithm: str = 'sha256') -> str:
        """
        Compute cryptographic hash of data.
        
        Args:
            data: Data to hash
            algorithm: Hash algorithm (sha256, sha512, md5)
            
        Returns:
            Hex digest of hash
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if algorithm == 'sha256':
            return hashlib.sha256(data).hexdigest()
        elif algorithm == 'sha512':
            return hashlib.sha512(data).hexdigest()
        elif algorithm == 'md5':
            return hashlib.md5(data).hexdigest()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def compute_hmac(self, 
                     data: Union[str, bytes], 
                     key: Optional[bytes] = None) -> str:
        """
        Compute HMAC for data integrity.
        
        Args:
            data: Data to compute HMAC for
            key: HMAC key (uses master key if not provided)
            
        Returns:
            HMAC hex digest
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        key = key or self.master_key
        return hmac.new(key, data, hashlib.sha256).hexdigest()
    
    def check_compliance(self, 
                         data: Dict,
                         regulation: str = 'gdpr') -> Dict[str, Any]:
        """
        Check data compliance with regulations.
        
        Args:
            data: Data to check
            regulation: Regulation to check ('gdpr', 'pci-dss', 'hipaa')
            
        Returns:
            Compliance report
        """
        report = {
            'regulation': regulation,
            'compliant': True,
            'violations': [],
            'warnings': []
        }
        
        if regulation == 'gdpr':
            # Check for consent
            if 'consent' not in data:
                report['violations'].append("Missing consent field")
                report['compliant'] = False
            
            # Check for data minimization
            if len(data) > 20:
                report['warnings'].append("Excessive data collection")
            
            # Check for PII
            pii_found = any(key in data for key in 
                           ['email', 'phone', 'address', 'ssn'])
            if pii_found and 'pii_processing_agreement' not in data:
                report['violations'].append("PII processing without agreement")
                report['compliant'] = False
        
        elif regulation == 'pci-dss':
            # Check for credit card data
            if 'credit_card' in data or 'cvv' in data:
                if 'encryption_method' not in data:
                    report['violations'].append("Credit card data not encrypted")
                    report['compliant'] = False
                
                if 'cvv' in data:
                    report['violations'].append("CVV should not be stored")
                    report['compliant'] = False
        
        self._audit_log('COMPLIANCE_CHECK', 
                       f"Compliance check for {regulation}: {report['compliant']}")
        
        return report
    
    def _audit_log(self, action: str, details: str):
        """
        Internal audit logging.
        
        Args:
            action: Action being logged
            details: Additional details
        """
        if not self.enable_audit:
            return
        
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'action': action,
            'details': details,
            'user': os.environ.get('USER', 'system')
        }
        
        self.audit_log.append(log_entry)
        
        # Also log to file
        logger.info(f"AUDIT: {action} - {details}")
        
        # Keep only last 10000 entries
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-10000:]
    
    def get_audit_log(self, 
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None,
                      action: Optional[str] = None) -> List[Dict]:
        """
        Retrieve audit log entries.
        
        Args:
            start_time: Filter by start time
            end_time: Filter by end time
            action: Filter by action
            
        Returns:
            Filtered audit log
        """
        filtered_log = self.audit_log
        
        if start_time:
            filtered_log = [e for e in filtered_log 
                          if datetime.fromisoformat(e['timestamp']) >= start_time]
        
        if end_time:
            filtered_log = [e for e in filtered_log 
                          if datetime.fromisoformat(e['timestamp']) <= end_time]
        
        if action:
            filtered_log = [e for e in filtered_log if e['action'] == action]
        
        return filtered_log


class DataEncryptor:
    """
    Specialized data encryptor for different data types.
    
    Provides encryption methods optimized for:
    - Large files (chunked encryption)
    - Database fields
    - Model parameters
    - Batch data processing
    """
    
    def __init__(self, security_manager: SecurityManager):
        """
        Initialize data encryptor.
        
        Args:
            security_manager: SecurityManager instance
        """
        self.security = security_manager
    
    def encrypt_file(self, 
                     input_path: str, 
                     output_path: str,
                     chunk_size: int = 64 * 1024 * 1024):  # 64MB chunks
        """
        Encrypt large file in chunks.
        
        Args:
            input_path: Path to input file
            output_path: Path to output encrypted file
            chunk_size: Size of chunks to process
        """
        # Generate random key for this file
        file_key = secrets.token_bytes(32)
        iv = secrets.token_bytes(16)
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(file_key),
            modes.CBC(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        
        with open(input_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
            # Write header (encrypted key and IV)
            encrypted_key = self.security.encrypt_data(file_key)
            f_out.write(len(encrypted_key).to_bytes(4, 'big'))
            f_out.write(encrypted_key)
            f_out.write(iv)
            
            # Encrypt file in chunks
            while True:
                chunk = f_in.read(chunk_size)
                if not chunk:
                    break
                
                # Pad last chunk
                if len(chunk) % 16 != 0:
                    padding_length = 16 - (len(chunk) % 16)
                    chunk += bytes([padding_length] * padding_length)
                
                encrypted_chunk = encryptor.update(chunk)
                f_out.write(encrypted_chunk)
            
            # Finalize
            f_out.write(encryptor.finalize())
    
    def decrypt_file(self, 
                     input_path: str, 
                     output_path: str):
        """
        Decrypt file encrypted with encrypt_file.
        
        Args:
            input_path: Path to encrypted file
            output_path: Path to output decrypted file
        """
        with open(input_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
            # Read header
            key_len = int.from_bytes(f_in.read(4), 'big')
            encrypted_key = f_in.read(key_len)
            iv = f_in.read(16)
            
            # Decrypt key
            file_key = self.security.decrypt_data(encrypted_key)
            if isinstance(file_key, dict):
                file_key = bytes(file_key)
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(file_key),
                modes.CBC(iv),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            
            # Decrypt remaining data
            remaining = f_in.read()
            decrypted = decryptor.update(remaining) + decryptor.finalize()
            
            # Remove padding
            padding_length = decrypted[-1]
            decrypted = decrypted[:-padding_length]
            
            f_out.write(decrypted)


class SecurityError(Exception):
    """Custom exception for security errors."""
    pass


# Convenience functions
def mask_pii(data: Any, **kwargs) -> Any:
    """Convenience function for PII masking."""
    security = SecurityManager()
    return security.mask_pii(data, **kwargs)


def encrypt_sensitive(data: Union[str, bytes, Dict]) -> bytes:
    """Convenience function for encryption."""
    security = SecurityManager()
    return security.encrypt_data(data)


def decrypt_sensitive(encrypted_data: bytes) -> Union[bytes, Dict]:
    """Convenience function for decryption."""
    security = SecurityManager()
    return security.decrypt_data(encrypted_data)


def hash_password(password: str) -> str:
    """Convenience function for password hashing."""
    security = SecurityManager()
    return security.hash_password(password)


def verify_password(password: str, hash_str: str) -> bool:
    """Convenience function for password verification."""
    security = SecurityManager()
    return security.verify_password(password, hash_str)