"""
Configuration Management for VeritasFinancial
=============================================
Manages all configuration aspects of the fraud detection system including:
- YAML/JSON configuration loading
- Environment variable integration
- Configuration validation
- Dynamic configuration updates
- Secret management
- Feature flags

Author: VeritasFinancial Team
"""

import os
import yaml
import json
from typing import Any, Dict, Optional, List, Union
from pathlib import Path
from dotenv import load_dotenv
import jsonschema
from datetime import datetime
import copy
import hashlib
import re


class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass


class EnvironmentManager:
    """
    Manages environment variables and their integration with configuration.
    
    This class handles loading, validating, and accessing environment variables
    with support for default values and type conversion.
    """
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize environment manager.
        
        Args:
            env_file (str, optional): Path to .env file
        """
        # Load .env file if provided
        if env_file and os.path.exists(env_file):
            load_dotenv(env_file)
        
        # Store environment variables
        self._env_vars = dict(os.environ)
        self._cache = {}
        
    def get(self, key: str, default: Any = None, required: bool = False,
            var_type: type = str) -> Any:
        """
        Get environment variable with type conversion.
        
        Args:
            key (str): Environment variable name
            default (Any): Default value if not found
            required (bool): Whether variable is required
            var_type (type): Expected type for conversion
            
        Returns:
            Any: Environment variable value
            
        Raises:
            ConfigError: If required variable is missing or type conversion fails
        """
        # Check cache first
        cache_key = f"{key}_{var_type.__name__}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Get value from environment
        value = self._env_vars.get(key)
        
        # Handle required but missing
        if value is None:
            if required:
                raise ConfigError(f"Required environment variable '{key}' not found")
            return default
        
        # Convert type
        try:
            if var_type == bool:
                converted = value.lower() in ('true', '1', 'yes', 'on')
            elif var_type == int:
                converted = int(value)
            elif var_type == float:
                converted = float(value)
            elif var_type == list:
                converted = self._parse_list(value)
            elif var_type == dict:
                converted = self._parse_dict(value)
            else:
                converted = str(value)
        except (ValueError, TypeError) as e:
            raise ConfigError(
                f"Failed to convert '{key}={value}' to {var_type.__name__}: {e}"
            )
        
        # Cache and return
        self._cache[cache_key] = converted
        return converted
    
    def _parse_list(self, value: str) -> List[str]:
        """
        Parse string as list.
        
        Supports comma-separated values and JSON arrays.
        
        Args:
            value (str): String to parse
            
        Returns:
            List[str]: Parsed list
        """
        value = value.strip()
        
        # Try JSON parsing first
        if value.startswith('[') and value.endswith(']'):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # Fall back to comma-separated
        return [item.strip() for item in value.split(',') if item.strip()]
    
    def _parse_dict(self, value: str) -> Dict:
        """
        Parse string as dictionary.
        
        Supports JSON objects and key=value pairs.
        
        Args:
            value (str): String to parse
            
        Returns:
            Dict: Parsed dictionary
        """
        value = value.strip()
        
        # Try JSON parsing
        if value.startswith('{') and value.endswith('}'):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # Parse as key=value pairs
        result = {}
        pairs = value.split(',')
        for pair in pairs:
            if '=' in pair:
                key, val = pair.split('=', 1)
                result[key.strip()] = val.strip()
        
        return result
    
    def set(self, key: str, value: Any):
        """
        Set environment variable (for testing).
        
        Args:
            key (str): Variable name
            value (Any): Variable value
        """
        self._env_vars[key] = str(value)
        # Clear cache for this key
        self._cache = {k: v for k, v in self._cache.items() 
                      if not k.startswith(f"{key}_")}
    
    def get_all(self, prefix: Optional[str] = None) -> Dict[str, str]:
        """
        Get all environment variables, optionally filtered by prefix.
        
        Args:
            prefix (str, optional): Filter by this prefix
            
        Returns:
            Dict[str, str]: Environment variables
        """
        if prefix:
            return {k: v for k, v in self._env_vars.items() 
                   if k.startswith(prefix)}
        return self._env_vars.copy()


class ConfigManager:
    """
    Central configuration manager for the entire system.
    
    This class handles loading, validating, and accessing configuration from
    multiple sources with support for:
    - YAML/JSON configuration files
    - Environment variables
    - Default values
    - Configuration validation
    - Dynamic updates
    - Secrets management
    
    Attributes:
        config (Dict): Loaded configuration
        env_manager (EnvironmentManager): Environment variable manager
    """
    
    # Default configuration values
    DEFAULTS = {
        'system': {
            'name': 'VeritasFinancial',
            'version': '1.0.0',
            'environment': 'development'
        },
        'logging': {
            'level': 'INFO',
            'format': 'structured',
            'rotation_size_mb': 10,
            'backup_count': 5
        },
        'data_pipeline': {
            'batch_size': 10000,
            'max_workers': 4,
            'use_gpu': False,
            'cache_enabled': True
        },
        'models': {
            'ensemble_method': 'weighted_average',
            'threshold': 0.5,
            'feature_version': 'v1'
        },
        'security': {
            'encryption_enabled': True,
            'token_expiry_hours': 24,
            'max_login_attempts': 5
        }
    }
    
    # Configuration schema for validation
    SCHEMA = {
        "type": "object",
        "properties": {
            "system": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "version": {"type": "string", "pattern": r"^\d+\.\d+\.\d+$"},
                    "environment": {"type": "string", "enum": ["development", "staging", "production"]}
                },
                "required": ["name", "version", "environment"]
            },
            "logging": {
                "type": "object",
                "properties": {
                    "level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR"]},
                    "format": {"type": "string", "enum": ["structured", "simple"]},
                    "rotation_size_mb": {"type": "integer", "minimum": 1},
                    "backup_count": {"type": "integer", "minimum": 0}
                }
            },
            "data_pipeline": {
                "type": "object",
                "properties": {
                    "batch_size": {"type": "integer", "minimum": 1},
                    "max_workers": {"type": "integer", "minimum": 1},
                    "use_gpu": {"type": "boolean"},
                    "cache_enabled": {"type": "boolean"}
                }
            },
            "models": {
                "type": "object",
                "properties": {
                    "ensemble_method": {"type": "string"},
                    "threshold": {"type": "number", "minimum": 0, "maximum": 1},
                    "feature_version": {"type": "string"}
                }
            },
            "security": {
                "type": "object",
                "properties": {
                    "encryption_enabled": {"type": "boolean"},
                    "token_expiry_hours": {"type": "integer", "minimum": 1},
                    "max_login_attempts": {"type": "integer", "minimum": 1}
                }
            }
        },
        "required": ["system", "logging"]
    }
    
    def __init__(self, config_path: Optional[str] = None, 
                 env_file: Optional[str] = None,
                 auto_load_env: bool = True):
        """
        Initialize configuration manager.
        
        Args:
            config_path (str, optional): Path to configuration file
            env_file (str, optional): Path to .env file
            auto_load_env (bool): Automatically load environment variables
            
        Raises:
            ConfigError: If configuration loading fails
        """
        self.env_manager = EnvironmentManager(env_file) if auto_load_env else None
        self.config = self._load_initial_config(config_path)
        self._config_history = []
        self._watchers = {}
        
    def _load_initial_config(self, config_path: Optional[str]) -> Dict:
        """
        Load initial configuration from file and defaults.
        
        Args:
            config_path (str, optional): Path to configuration file
            
        Returns:
            Dict: Loaded configuration
        """
        # Start with defaults
        config = copy.deepcopy(self.DEFAULTS)
        
        # Load from file if provided
        if config_path:
            file_config = self._load_from_file(config_path)
            config = self._deep_merge(config, file_config)
        
        # Override with environment variables if available
        if self.env_manager:
            env_config = self._load_from_env()
            config = self._deep_merge(config, env_config)
        
        return config
    
    def _load_from_file(self, config_path: str) -> Dict:
        """
        Load configuration from file.
        
        Supports YAML and JSON formats.
        
        Args:
            config_path (str): Path to configuration file
            
        Returns:
            Dict: Loaded configuration
            
        Raises:
            ConfigError: If file loading fails
        """
        path = Path(config_path)
        
        if not path.exists():
            raise ConfigError(f"Configuration file not found: {config_path}")
        
        try:
            with open(path, 'r') as f:
                if path.suffix in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif path.suffix == '.json':
                    return json.load(f) or {}
                else:
                    raise ConfigError(f"Unsupported file format: {path.suffix}")
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ConfigError(f"Failed to parse configuration file: {e}")
    
    def _load_from_env(self) -> Dict:
        """
        Load configuration from environment variables.
        
        Environment variables with prefix 'VF_' are automatically loaded
        and nested based on double underscores.
        
        Example:
            VF_SYSTEM__ENVIRONMENT=production -> {'system': {'environment': 'production'}}
            
        Returns:
            Dict: Configuration from environment
        """
        if not self.env_manager:
            return {}
        
        env_vars = self.env_manager.get_all(prefix='VF_')
        config = {}
        
        for key, value in env_vars.items():
            # Remove prefix and split by double underscore
            key = key[3:]  # Remove 'VF_'
            parts = key.split('__')
            
            # Build nested dictionary
            current = config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Try to parse value
            current[parts[-1].lower()] = self._parse_env_value(value)
        
        return config
    
    def _parse_env_value(self, value: str) -> Any:
        """
        Parse environment variable value to appropriate type.
        
        Args:
            value (str): Raw environment variable value
            
        Returns:
            Any: Parsed value
        """
        # Try boolean
        if value.lower() in ['true', 'false', 'yes', 'no', 'on', 'off']:
            return value.lower() in ['true', 'yes', 'on']
        
        # Try number
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Try JSON
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        # Default to string
        return value
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """
        Deep merge two dictionaries.
        
        Args:
            base (Dict): Base dictionary
            override (Dict): Override dictionary
            
        Returns:
            Dict: Merged dictionary
        """
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    def get_config(self, *keys: str, default: Any = None) -> Any:
        """
        Get configuration value by key path.
        
        Args:
            *keys: Key path (e.g., 'system', 'environment')
            default: Default value if key not found
            
        Returns:
            Any: Configuration value
            
        Example:
            >>> env = config.get_config('system', 'environment')
            >>> threshold = config.get_config('models', 'threshold', default=0.5)
        """
        current = self.config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def set_config(self, value: Any, *keys: str, validate: bool = True):
        """
        Set configuration value.
        
        Args:
            value: Value to set
            *keys: Key path
            validate (bool): Whether to validate after setting
            
        Raises:
            ConfigError: If validation fails
        """
        if not keys:
            raise ConfigError("No keys provided for configuration")
        
        # Navigate to the parent
        current = self.config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        
        # Record history for potential rollback
        last_key = keys[-1]
        old_value = current.get(last_key) if last_key in current else None
        self._config_history.append({
            'keys': keys,
            'old_value': old_value,
            'new_value': value,
            'timestamp': datetime.utcnow()
        })
        
        # Set the value
        current[last_key] = value
        
        # Validate if requested
        if validate:
            self.validate_config()
        
        # Notify watchers
        self._notify_watchers(keys, value)
    
    def validate_config(self) -> bool:
        """
        Validate current configuration against schema.
        
        Returns:
            bool: True if valid
            
        Raises:
            ConfigError: If validation fails
        """
        try:
            jsonschema.validate(self.config, self.SCHEMA)
            return True
        except jsonschema.ValidationError as e:
            raise ConfigError(f"Configuration validation failed: {e}")
    
    def watch(self, key_path: List[str], callback: callable):
        """
        Watch for changes to a specific configuration key.
        
        Args:
            key_path (List[str]): Key path to watch
            callback (callable): Function to call when value changes
        """
        key_str = '.'.join(key_path)
        if key_str not in self._watchers:
            self._watchers[key_str] = []
        self._watchers[key_str].append(callback)
    
    def _notify_watchers(self, keys: tuple, new_value: Any):
        """
        Notify watchers of configuration changes.
        
        Args:
            keys (tuple): Changed key path
            new_value (Any): New value
        """
        key_str = '.'.join(keys)
        if key_str in self._watchers:
            for callback in self._watchers[key_str]:
                try:
                    callback(new_value)
                except Exception as e:
                    # Log but don't fail
                    print(f"Watcher callback failed: {e}")
    
    def get_secret(self, secret_name: str, required: bool = False) -> Optional[str]:
        """
        Get a secret from environment or vault.
        
        Args:
            secret_name (str): Name of the secret
            required (bool): Whether secret is required
            
        Returns:
            Optional[str]: Secret value
            
        Raises:
            ConfigError: If required secret not found
        """
        if self.env_manager:
            # Try environment variable first
            env_value = self.env_manager.get(
                f"VF_SECRET_{secret_name.upper()}",
                required=required
            )
            if env_value is not None:
                return env_value
        
        # Try configuration file
        secrets = self.get_config('secrets', default={})
        if secret_name in secrets:
            return secrets[secret_name]
        
        if required:
            raise ConfigError(f"Required secret '{secret_name}' not found")
        
        return None
    
    def to_dict(self) -> Dict:
        """
        Export configuration as dictionary.
        
        Returns:
            Dict: Configuration dictionary
        """
        return copy.deepcopy(self.config)
    
    def save(self, file_path: str, format: str = 'yaml'):
        """
        Save current configuration to file.
        
        Args:
            file_path (str): Path to save configuration
            format (str): Output format ('yaml' or 'json')
            
        Raises:
            ConfigError: If saving fails
        """
        try:
            with open(file_path, 'w') as f:
                if format.lower() == 'yaml':
                    yaml.dump(self.config, f, default_flow_style=False)
                elif format.lower() == 'json':
                    json.dump(self.config, f, indent=2)
                else:
                    raise ConfigError(f"Unsupported format: {format}")
        except Exception as e:
            raise ConfigError(f"Failed to save configuration: {e}")
    
    def rollback(self, steps: int = 1):
        """
        Rollback configuration changes.
        
        Args:
            steps (int): Number of changes to rollback
            
        Raises:
            ConfigError: If rollback fails
        """
        if not self._config_history:
            raise ConfigError("No configuration history to rollback")
        
        for _ in range(min(steps, len(self._config_history))):
            entry = self._config_history.pop()
            
            # Restore old value
            if entry['old_value'] is not None:
                self.set_config(entry['old_value'], *entry['keys'], validate=False)
            else:
                # Delete the key
                current = self.config
                for key in entry['keys'][:-1]:
                    current = current[key]
                del current[entry['keys'][-1]]
    
    def get_version_hash(self) -> str:
        """
        Get hash of current configuration for version tracking.
        
        Returns:
            str: SHA256 hash of configuration
        """
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:8]


def load_config(config_path: str, env_file: Optional[str] = None) -> ConfigManager:
    """
    Convenience function to load configuration.
    
    Args:
        config_path (str): Path to configuration file
        env_file (str, optional): Path to .env file
        
    Returns:
        ConfigManager: Configured configuration manager
    """
    return ConfigManager(config_path, env_file)


def validate_config(config: Dict, schema: Optional[Dict] = None) -> bool:
    """
    Validate configuration against schema.
    
    Args:
        config (Dict): Configuration to validate
        schema (Dict, optional): Custom schema, uses default if None
        
    Returns:
        bool: True if valid
        
    Raises:
        ConfigError: If validation fails
    """
    if schema is None:
        schema = ConfigManager.SCHEMA
    
    try:
        jsonschema.validate(config, schema)
        return True
    except jsonschema.ValidationError as e:
        raise ConfigError(f"Configuration validation failed: {e}")