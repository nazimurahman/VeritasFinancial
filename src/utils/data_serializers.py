"""
Data Serializers Module for VeritasFinancial Banking Fraud Detection System

This module provides comprehensive serialization utilities for:
- Model persistence (save/load ML models)
- Data compression and decompression
- Format conversion (JSON, Parquet, Pickle, etc.)
- Secure serialization with encryption support
- Version control for serialized artifacts
- Memory-efficient serialization for large datasets

Author: VeritasFinancial Data Science Team
Version: 1.0.0
"""

import pickle
import json
import yaml
import joblib
import pandas as pd
import numpy as np
import torch
import cloudpickle
import base64
import zlib
import gzip
import bz2
import lzma
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import logging
from datetime import datetime
import hashlib
import os
import io
import msgpack
import pyarrow as pa
import pyarrow.parquet as pq
import pickle5  # Python 3.8+ compatibility
import dill  # Extended pickling support

# Configure logging
logger = logging.getLogger(__name__)


class DataSerializer:
    """
    Main serializer class for handling all data serialization needs in the fraud detection system.
    
    Features:
    - Multiple serialization formats (pickle, joblib, torch, json, parquet)
    - Compression support (gzip, bz2, lzma, zlib)
    - Encryption integration
    - Version control
    - Memory-efficient streaming for large datasets
    - Cross-version compatibility
    """
    
    # Supported serialization formats
    SUPPORTED_FORMATS = ['pickle', 'joblib', 'torch', 'json', 'yaml', 'parquet', 'msgpack']
    
    # Supported compression algorithms
    SUPPORTED_COMPRESSION = ['gzip', 'bz2', 'lzma', 'zlib', None]
    
    def __init__(self, 
                 default_format: str = 'joblib',
                 default_compression: Optional[str] = 'gzip',
                 compress_level: int = 9,
                 encrypt_by_default: bool = False,
                 encryption_key: Optional[bytes] = None,
                 version_control: bool = True):
        """
        Initialize the DataSerializer with configuration.
        
        Args:
            default_format: Default serialization format ('pickle', 'joblib', 'torch', 'json', etc.)
            default_compression: Default compression algorithm (None for no compression)
            compress_level: Compression level (1-9, higher = more compression but slower)
            encrypt_by_default: Whether to encrypt serialized data by default
            encryption_key: Encryption key for secure serialization (32 bytes for AES-256)
            version_control: Enable version tracking for serialized artifacts
        """
        self.default_format = default_format
        self.default_compression = default_compression
        self.compress_level = compress_level
        self.encrypt_by_default = encrypt_by_default
        self.encryption_key = encryption_key
        self.version_control = version_control
        
        # Initialize version registry
        self._version_registry = {}
        
        # Validate initialization parameters
        self._validate_config()
        
        logger.info(f"DataSerializer initialized with format={default_format}, "
                   f"compression={default_compression}, encrypt={encrypt_by_default}")
    
    def _validate_config(self):
        """Validate serializer configuration."""
        if self.default_format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {self.default_format}. "
                           f"Supported formats: {self.SUPPORTED_FORMATS}")
        
        if self.default_compression not in self.SUPPORTED_COMPRESSION:
            raise ValueError(f"Unsupported compression: {self.default_compression}. "
                           f"Supported: {self.SUPPORTED_COMPRESSION}")
        
        if self.encrypt_by_default and not self.encryption_key:
            raise ValueError("Encryption key required when encrypt_by_default=True")
    
    def serialize(self, 
                  obj: Any, 
                  path: Optional[Union[str, Path]] = None,
                  format: Optional[str] = None,
                  compression: Optional[str] = None,
                  encrypt: Optional[bool] = None,
                  metadata: Optional[Dict] = None) -> Union[bytes, None]:
        """
        Serialize an object to bytes or file.
        
        This is the main serialization method that handles:
        - Automatic format detection based on file extension
        - Memory-efficient serialization for large objects
        - Metadata embedding for version control
        - Encryption support for sensitive data
        
        Args:
            obj: Object to serialize
            path: Optional file path to save serialized data
            format: Serialization format (overrides default)
            compression: Compression algorithm (overrides default)
            encrypt: Whether to encrypt (overrides default)
            metadata: Additional metadata to embed with the serialized data
            
        Returns:
            Bytes if no path provided, None if saved to file
        """
        # Use defaults if not specified
        format = format or self.default_format
        compression = compression or self.default_compression
        encrypt = encrypt if encrypt is not None else self.encrypt_by_default
        
        # Add metadata for version control
        if self.version_control and metadata is None:
            metadata = self._create_metadata(obj)
        
        try:
            # Select serialization method based on format
            if format == 'pickle':
                serialized_data = self._serialize_pickle(obj, compression)
            elif format == 'joblib':
                serialized_data = self._serialize_joblib(obj, compression)
            elif format == 'torch':
                serialized_data = self._serialize_torch(obj, compression)
            elif format == 'json':
                serialized_data = self._serialize_json(obj, compression)
            elif format == 'yaml':
                serialized_data = self._serialize_yaml(obj, compression)
            elif format == 'parquet':
                serialized_data = self._serialize_parquet(obj, compression)
            elif format == 'msgpack':
                serialized_data = self._serialize_msgpack(obj, compression)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Apply encryption if requested
            if encrypt:
                serialized_data = self._encrypt_data(serialized_data)
            
            # Add metadata wrapper for version control
            if self.version_control:
                serialized_data = self._wrap_with_metadata(serialized_data, metadata)
            
            # Save to file or return bytes
            if path:
                self._save_to_file(path, serialized_data)
                logger.info(f"Object serialized and saved to {path}")
                return None
            else:
                logger.debug(f"Object serialized to bytes ({len(serialized_data)} bytes)")
                return serialized_data
                
        except Exception as e:
            logger.error(f"Serialization failed: {str(e)}")
            raise SerializationError(f"Failed to serialize object: {str(e)}")
    
    def deserialize(self, 
                    data: Union[bytes, str, Path],
                    format: Optional[str] = None) -> Any:
        """
        Deserialize data from bytes or file.
        
        Handles:
        - Automatic format detection
        - Decompression
        - Decryption
        - Metadata extraction
        - Version compatibility checks
        
        Args:
            data: Bytes or file path to deserialize
            format: Expected format (auto-detected if not provided)
            
        Returns:
            Deserialized object
        """
        try:
            # Load data from file if path provided
            if isinstance(data, (str, Path)):
                data = self._load_from_file(data)
                # Auto-detect format from file extension if not specified
                if not format:
                    format = self._detect_format_from_path(data)
            
            # Extract metadata wrapper if present
            if self.version_control:
                data, metadata = self._extract_metadata(data)
                self._validate_version_compatibility(metadata)
            
            # Decrypt if needed
            data = self._decrypt_data(data)
            
            # Decompress based on magic bytes
            compression = self._detect_compression(data)
            if compression:
                data = self._decompress_data(data, compression)
            
            # Detect format from data structure
            if not format:
                format = self._detect_format_from_data(data)
            
            # Deserialize based on format
            if format == 'pickle':
                return self._deserialize_pickle(data)
            elif format == 'joblib':
                return self._deserialize_joblib(data)
            elif format == 'torch':
                return self._deserialize_torch(data)
            elif format == 'json':
                return self._deserialize_json(data)
            elif format == 'yaml':
                return self._deserialize_yaml(data)
            elif format == 'parquet':
                return self._deserialize_parquet(data)
            elif format == 'msgpack':
                return self._deserialize_msgpack(data)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Deserialization failed: {str(e)}")
            raise DeserializationError(f"Failed to deserialize data: {str(e)}")
    
    def _serialize_pickle(self, obj: Any, compression: Optional[str]) -> bytes:
        """
        Serialize using pickle with compression.
        
        Pickle is Python's native serialization format but can be insecure.
        We use it for trusted data only and add compression for efficiency.
        """
        try:
            # Use highest protocol for efficiency
            serialized = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Apply compression if requested
            if compression:
                serialized = self._compress_data(serialized, compression)
            
            return serialized
            
        except Exception as e:
            raise SerializationError(f"Pickle serialization failed: {str(e)}")
    
    def _serialize_joblib(self, obj: Any, compression: Optional[str]) -> bytes:
        """
        Serialize using joblib (optimized for NumPy arrays).
        
        Joblib is excellent for numerical data and scikit-learn models.
        It's more efficient than pickle for large numpy arrays.
        """
        try:
            # Joblib handles compression internally
            with io.BytesIO() as buffer:
                joblib.dump(obj, buffer, compress=compression)
                serialized = buffer.getvalue()
            
            return serialized
            
        except Exception as e:
            raise SerializationError(f"Joblib serialization failed: {str(e)}")
    
    def _serialize_torch(self, obj: Any, compression: Optional[str]) -> bytes:
        """
        Serialize PyTorch models and tensors.
        
        Handles both state_dict and full models with custom compression.
        """
        try:
            # Handle PyTorch models specially
            if isinstance(obj, torch.nn.Module):
                # Save only state dict for models (more portable)
                obj = obj.state_dict()
            
            # Serialize with torch
            with io.BytesIO() as buffer:
                torch.save(obj, buffer, pickle_protocol=pickle.HIGHEST_PROTOCOL)
                serialized = buffer.getvalue()
            
            # Apply compression if requested
            if compression:
                serialized = self._compress_data(serialized, compression)
            
            return serialized
            
        except Exception as e:
            raise SerializationError(f"PyTorch serialization failed: {str(e)}")
    
    def _serialize_json(self, obj: Any, compression: Optional[str]) -> bytes:
        """
        Serialize to JSON with custom encoders for numpy/pandas types.
        
        JSON is human-readable and cross-platform but less efficient.
        """
        try:
            # Custom JSON encoder for numpy and pandas types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, pd.Series):
                        return obj.to_dict()
                    elif isinstance(obj, pd.DataFrame):
                        return obj.to_dict(orient='records')
                    elif isinstance(obj, datetime):
                        return obj.isoformat()
                    return super().default(obj)
            
            # Convert to JSON string and encode to bytes
            json_str = json.dumps(obj, cls=NumpyEncoder, indent=2)
            serialized = json_str.encode('utf-8')
            
            # Apply compression if requested
            if compression:
                serialized = self._compress_data(serialized, compression)
            
            return serialized
            
        except Exception as e:
            raise SerializationError(f"JSON serialization failed: {str(e)}")
    
    def _serialize_parquet(self, obj: Any, compression: Optional[str]) -> bytes:
        """
        Serialize pandas DataFrames to Parquet format.
        
        Parquet is columnar storage, excellent for tabular data.
        """
        try:
            if not isinstance(obj, (pd.DataFrame, pd.Series)):
                raise ValueError("Parquet serialization only supports pandas objects")
            
            # Convert Series to DataFrame
            if isinstance(obj, pd.Series):
                obj = obj.to_frame()
            
            # Serialize to Parquet
            with io.BytesIO() as buffer:
                obj.to_parquet(buffer, compression=compression or 'snappy')
                serialized = buffer.getvalue()
            
            return serialized
            
        except Exception as e:
            raise SerializationError(f"Parquet serialization failed: {str(e)}")
    
    def _compress_data(self, data: bytes, algorithm: str) -> bytes:
        """
        Compress data using specified algorithm.
        
        Different compression algorithms offer trade-offs:
        - gzip: Good compression, widely supported
        - bz2: Better compression, slower
        - lzma: Best compression, slowest
        - zlib: Fast, moderate compression
        """
        try:
            if algorithm == 'gzip':
                return gzip.compress(data, compresslevel=self.compress_level)
            elif algorithm == 'bz2':
                return bz2.compress(data, compresslevel=self.compress_level)
            elif algorithm == 'lzma':
                return lzma.compress(data, preset=self.compress_level)
            elif algorithm == 'zlib':
                return zlib.compress(data, level=self.compress_level)
            else:
                return data
                
        except Exception as e:
            raise CompressionError(f"Compression failed: {str(e)}")
    
    def _decompress_data(self, data: bytes, algorithm: str) -> bytes:
        """Decompress data using specified algorithm."""
        try:
            if algorithm == 'gzip':
                return gzip.decompress(data)
            elif algorithm == 'bz2':
                return bz2.decompress(data)
            elif algorithm == 'lzma':
                return lzma.decompress(data)
            elif algorithm == 'zlib':
                return zlib.decompress(data)
            else:
                return data
                
        except Exception as e:
            raise DecompressionError(f"Decompression failed: {str(e)}")
    
    def _detect_compression(self, data: bytes) -> Optional[str]:
        """
        Detect compression algorithm from magic bytes.
        
        Magic bytes are the first few bytes that identify file formats.
        """
        if data.startswith(b'\x1f\x8b'):  # gzip magic bytes
            return 'gzip'
        elif data.startswith(b'BZh'):  # bz2 magic bytes
            return 'bz2'
        elif data.startswith(b'\xfd\x37\x7a\x58\x5a\x00'):  # lzma magic bytes
            return 'lzma'
        elif data.startswith(b'\x78\x9c') or data.startswith(b'\x78\xda'):  # zlib magic bytes
            return 'zlib'
        return None
    
    def _create_metadata(self, obj: Any) -> Dict:
        """
        Create metadata for version control.
        
        Includes:
        - Timestamp
        - Object type and size
        - Hash for integrity
        - Version information
        """
        metadata = {
            'serialized_at': datetime.utcnow().isoformat(),
            'object_type': type(obj).__name__,
            'object_size': self._estimate_size(obj),
            'serializer_version': '1.0.0',
            'python_version': os.sys.version,
            'hash': self._compute_hash(obj)
        }
        return metadata
    
    def _compute_hash(self, obj: Any) -> str:
        """Compute SHA-256 hash of object for integrity verification."""
        try:
            # Quick serialization for hash computation
            temp_data = pickle.dumps(obj, protocol=4)
            return hashlib.sha256(temp_data).hexdigest()
        except:
            return "hash_unavailable"
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            return len(pickle.dumps(obj, protocol=4))
        except:
            return -1
    
    def _wrap_with_metadata(self, data: bytes, metadata: Dict) -> bytes:
        """
        Wrap serialized data with metadata for version control.
        
        Format: [META][DATA] where META is a JSON header
        """
        meta_json = json.dumps(metadata).encode('utf-8')
        meta_len = len(meta_json).to_bytes(4, byteorder='big')
        return meta_len + meta_json + data
    
    def _extract_metadata(self, data: bytes) -> Tuple[bytes, Dict]:
        """Extract metadata from wrapped data."""
        meta_len = int.from_bytes(data[:4], byteorder='big')
        meta_json = data[4:4+meta_len].decode('utf-8')
        metadata = json.loads(meta_json)
        actual_data = data[4+meta_len:]
        return actual_data, metadata
    
    def _encrypt_data(self, data: bytes) -> bytes:
        """
        Encrypt data using AES-256-GCM.
        
        Provides confidentiality and integrity protection.
        """
        if not self.encryption_key:
            return data
        
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.primitives import padding
            from cryptography.hazmat.backends import default_backend
            import os
            
            # Generate random nonce
            nonce = os.urandom(12)
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(self.encryption_key),
                modes.GCM(nonce),
                backend=default_backend()
            )
            
            # Encrypt
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(data) + encryptor.finalize()
            
            # Return nonce + ciphertext + tag
            return nonce + ciphertext + encryptor.tag
            
        except ImportError:
            logger.warning("cryptography not installed, skipping encryption")
            return data
    
    def _decrypt_data(self, data: bytes) -> bytes:
        """Decrypt AES-256-GCM encrypted data."""
        if not self.encryption_key:
            return data
        
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.backends import default_backend
            
            # Extract components (nonce:12, tag:16)
            nonce = data[:12]
            tag = data[-16:]
            ciphertext = data[12:-16]
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(self.encryption_key),
                modes.GCM(nonce, tag),
                backend=default_backend()
            )
            
            # Decrypt
            decryptor = cipher.decryptor()
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            
            return plaintext
            
        except ImportError:
            return data


class SerializationError(Exception):
    """Custom exception for serialization errors."""
    pass


class DeserializationError(Exception):
    """Custom exception for deserialization errors."""
    pass


class CompressionError(Exception):
    """Custom exception for compression errors."""
    pass


class DecompressionError(Exception):
    """Custom exception for decompression errors."""
    pass


# Convenience functions
def save_model(model: Any, path: Union[str, Path], **kwargs):
    """Convenience function to save a model."""
    serializer = DataSerializer(**kwargs)
    return serializer.serialize(model, path)


def load_model(path: Union[str, Path], **kwargs):
    """Convenience function to load a model."""
    serializer = DataSerializer(**kwargs)
    return serializer.deserialize(path)


def save_dataframe(df: pd.DataFrame, path: Union[str, Path], **kwargs):
    """Convenience function to save a DataFrame."""
    kwargs.setdefault('format', 'parquet')
    serializer = DataSerializer(**kwargs)
    return serializer.serialize(df, path)


def load_dataframe(path: Union[str, Path], **kwargs):
    """Convenience function to load a DataFrame."""
    serializer = DataSerializer(**kwargs)
    return serializer.deserialize(path)