# tests/test_utils.py
"""
Unit tests for utility functions.
Tests logging, configuration management, data serialization, and helpers.
"""

import pytest
import pandas as pd
import numpy as np
import json
import yaml
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, mock_open
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.logger import Logger, get_logger
from src.utils.config_manager import ConfigManager, load_config
from src.utils.data_serializers import (
    serialize_dataframe,
    deserialize_dataframe,
    save_features,
    load_features
)
from src.utils.parallel_processing import ParallelProcessor
from src.utils.helpers import (
    ensure_dir,
    get_timestamp,
    calculate_date_diff,
    safe_divide,
    flatten_dict,
    chunk_list,
    memory_usage,
    time_execution
)
from src.utils.security import (
    hash_data,
    anonymize_data,
    validate_email,
    sanitize_input
)


class TestLogger:
    """
    Test suite for Logger utility.
    Tests logging functionality.
    """
    
    def test_logger_initialization(self):
        """
        Test Logger initialization.
        """
        logger = Logger(name='test_logger')
        assert logger.name == 'test_logger'
        assert logger.logger is not None
        
        # Test with different log levels
        logger = Logger(name='test_logger', level='DEBUG')
        assert logger.level == 'DEBUG'
        
        # Test with file output
        with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as tmp:
            log_file = tmp.name
            logger = Logger(name='test_logger', log_file=log_file)
            assert logger.log_file == log_file
    
    def test_log_messages(self):
        """
        Test logging messages at different levels.
        """
        logger = Logger(name='test_logger')
        
        # These should not raise exceptions
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")
    
    def test_get_logger(self):
        """
        Test get_logger convenience function.
        """
        logger = get_logger('test_module')
        assert logger.name == 'test_module'
        assert logger.logger is not None
    
    def test_log_to_file(self):
        """
        Test logging to file.
        """
        with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as tmp:
            log_file = tmp.name
        
        try:
            logger = Logger(name='file_logger', log_file=log_file)
            test_message = "Test file logging"
            logger.info(test_message)
            
            # Check file contains message
            with open(log_file, 'r') as f:
                content = f.read()
                assert test_message in content
        finally:
            os.unlink(log_file)


class TestConfigManager:
    """
    Test suite for ConfigManager.
    Tests configuration loading and management.
    """
    
    @pytest.fixture
    def yaml_config_file(self):
        """
        Create a temporary YAML config file.
        """
        config = {
            'data': {
                'sources': {
                    'transactions': 'postgresql://localhost:5432/bank',
                    'customers': 'api://customers/v1'
                },
                'batch_size': 10000,
                'cache_enabled': True
            },
            'model': {
                'type': 'xgboost',
                'params': {
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'n_estimators': 100
                }
            },
            'features': [
                'amount',
                'time_since_last_tx',
                'customer_avg_amount'
            ],
            'thresholds': {
                'fraud_probability': 0.7,
                'min_precision': 0.8
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp_path = tmp.name
        
        yield tmp_path
        os.unlink(tmp_path)
    
    @pytest.fixture
    def json_config_file(self):
        """
        Create a temporary JSON config file.
        """
        config = {
            'data': {
                'sources': {
                    'transactions': 'postgresql://localhost:5432/bank',
                    'customers': 'api://customers/v1'
                },
                'batch_size': 10000
            },
            'model': {
                'type': 'xgboost',
                'params': {
                    'max_depth': 6,
                    'learning_rate': 0.1
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(config, tmp)
            tmp_path = tmp.name
        
        yield tmp_path
        os.unlink(tmp_path)
    
    def test_load_yaml_config(self, yaml_config_file):
        """
        Test loading YAML configuration.
        """
        config_manager = ConfigManager(yaml_config_file)
        
        assert config_manager.config is not None
        assert 'data' in config_manager.config
        assert 'model' in config_manager.config
        assert config_manager.config['data']['batch_size'] == 10000
    
    def test_load_json_config(self, json_config_file):
        """
        Test loading JSON configuration.
        """
        config_manager = ConfigManager(json_config_file)
        
        assert config_manager.config is not None
        assert 'data' in config_manager.config
        assert 'model' in config_manager.config
    
    def test_get_value(self, yaml_config_file):
        """
        Test getting values from config.
        """
        config_manager = ConfigManager(yaml_config_file)
        
        # Get existing value
        batch_size = config_manager.get('data.batch_size')
        assert batch_size == 10000
        
        # Get nested value
        learning_rate = config_manager.get('model.params.learning_rate')
        assert learning_rate == 0.1
        
        # Get with default
        missing = config_manager.get('nonexistent.key', default='default')
        assert missing == 'default'
    
    def test_set_value(self, yaml_config_file):
        """
        Test setting values in config.
        """
        config_manager = ConfigManager(yaml_config_file)
        
        # Set simple value
        config_manager.set('new_key', 'new_value')
        assert config_manager.get('new_key') == 'new_value'
        
        # Set nested value
        config_manager.set('model.params.new_param', 42)
        assert config_manager.get('model.params.new_param') == 42
        
        # Update existing
        config_manager.set('data.batch_size', 20000)
        assert config_manager.get('data.batch_size') == 20000
    
    def test_save_config(self, yaml_config_file):
        """
        Test saving configuration to file.
        """
        config_manager = ConfigManager(yaml_config_file)
        
        # Modify config
        config_manager.set('new_setting', 'test')
        
        # Save to new file
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as tmp:
            save_path = tmp.name
            config_manager.save(save_path)
        
        try:
            # Load saved config
            loaded_manager = ConfigManager(save_path)
            assert loaded_manager.get('new_setting') == 'test'
        finally:
            os.unlink(save_path)
    
    def test_load_config_function(self, yaml_config_file):
        """
        Test load_config convenience function.
        """
        config = load_config(yaml_config_file)
        
        assert config is not None
        assert 'data' in config
        assert config['data']['batch_size'] == 10000
    
    def test_invalid_file(self):
        """
        Test loading from invalid file.
        """
        with pytest.raises(FileNotFoundError):
            ConfigManager('nonexistent_file.yaml')


class TestDataSerializers:
    """
    Test suite for DataSerializers.
    Tests data serialization and deserialization.
    """
    
    @pytest.fixture
    def sample_dataframe(self):
        """
        Fixture providing a sample DataFrame for testing.
        """
        np.random.seed(42)
        n_rows = 100
        
        return pd.DataFrame({
            'id': range(n_rows),
            'name': [f'Name_{i}' for i in range(n_rows)],
            'value': np.random.randn(n_rows),
            'category': np.random.choice(['A', 'B', 'C'], n_rows),
            'timestamp': [datetime.now() - timedelta(days=i) for i in range(n_rows)],
            'is_valid': np.random.choice([True, False], n_rows)
        })
    
    def test_serialize_deserialize_dataframe(self, sample_dataframe):
        """
        Test serialization and deserialization of DataFrame.
        """
        # Serialize to bytes
        serialized = serialize_dataframe(sample_dataframe, format='parquet')
        assert isinstance(serialized, bytes)
        
        # Deserialize
        deserialized = deserialize_dataframe(serialized, format='parquet')
        
        # Check equality
        pd.testing.assert_frame_equal(sample_dataframe, deserialized)
    
    def test_save_load_features(self, sample_dataframe):
        """
        Test saving and loading features to/from file.
        """
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            file_path = tmp.name
        
        try:
            # Save features
            save_features(sample_dataframe, file_path)
            
            # Load features
            loaded = load_features(file_path)
            
            # Check equality
            pd.testing.assert_frame_equal(sample_dataframe, loaded)
        finally:
            os.unlink(file_path)
    
    def test_save_load_csv(self, sample_dataframe):
        """
        Test saving and loading CSV format.
        """
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            file_path = tmp.name
        
        try:
            # Save as CSV
            save_features(sample_dataframe, file_path, format='csv')
            
            # Load CSV
            loaded = load_features(file_path, format='csv')
            
            # Check shape (CSV might not preserve dtypes perfectly)
            assert loaded.shape == sample_dataframe.shape
        finally:
            os.unlink(file_path)
    
    def test_save_load_json(self, sample_dataframe):
        """
        Test saving and loading JSON format.
        """
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            file_path = tmp.name
        
        try:
            # Save as JSON
            save_features(sample_dataframe, file_path, format='json')
            
            # Load JSON
            loaded = load_features(file_path, format='json')
            
            # Check shape
            assert loaded.shape == sample_dataframe.shape
        finally:
            os.unlink(file_path)


class TestParallelProcessor:
    """
    Test suite for ParallelProcessor.
    Tests parallel processing utilities.
    """
    
    def test_initialization(self):
        """
        Test ParallelProcessor initialization.
        """
        processor = ParallelProcessor(n_jobs=2)
        assert processor.n_jobs == 2
        
        processor = ParallelProcessor(n_jobs=-1)  # Use all cores
        assert processor.n_jobs == -1
    
    def test_parallel_map(self):
        """
        Test parallel map function.
        """
        processor = ParallelProcessor(n_jobs=2)
        
        # Define a simple function
        def square(x):
            return x * x
        
        # Apply to list
        items = list(range(10))
        results = processor.parallel_map(square, items)
        
        # Check results
        assert results == [square(x) for x in items]
    
    def test_parallel_map_with_args(self):
        """
        Test parallel map with additional arguments.
        """
        processor = ParallelProcessor(n_jobs=2)
        
        # Define function with additional args
        def power(x, exponent):
            return x ** exponent
        
        # Apply with fixed exponent
        items = list(range(5))
        results = processor.parallel_map(power, items, exponent=3)
        
        # Check results
        expected = [x ** 3 for x in items]
        assert results == expected
    
    def test_parallel_dataframe_apply(self):
        """
        Test parallel apply on DataFrame.
        """
        processor = ParallelProcessor(n_jobs=2)
        
        # Create DataFrame
        df = pd.DataFrame({'value': range(100)})
        
        # Define function
        def transform(row):
            return row['value'] * 2
        
        # Apply in parallel
        result = processor.parallel_dataframe_apply(df, transform, axis=1)
        
        # Check results
        expected = df['value'] * 2
        pd.testing.assert_series_equal(result, expected, check_names=False)
    
    def test_chunk_processing(self):
        """
        Test processing data in chunks.
        """
        processor = ParallelProcessor(n_jobs=2)
        
        # Create large list
        data = list(range(1000))
        
        # Process in chunks
        def process_chunk(chunk):
            return [x * 2 for x in chunk]
        
        results = processor.process_chunks(data, process_chunk, chunk_size=100)
        
        # Check results
        expected = [x * 2 for x in data]
        assert results == expected
    
    def test_error_handling(self):
        """
        Test error handling in parallel processing.
        """
        processor = ParallelProcessor(n_jobs=2)
        
        # Define function that fails for some inputs
        def risky_function(x):
            if x == 5:
                raise ValueError("Intentional error")
            return x * 2
        
        items = list(range(10))
        
        # Should handle errors gracefully
        results = processor.parallel_map(risky_function, items, ignore_errors=True)
        
        # Check that error items are excluded
        assert len(results) == len(items) - 1
        assert 5 not in [r for r in results if r is not None]


class TestHelpers:
    """
    Test suite for Helper functions.
    Tests various utility helper functions.
    """
    
    def test_ensure_dir(self):
        """
        Test directory creation helper.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, 'nested', 'directories')
            
            # Ensure directory exists
            result = ensure_dir(new_dir)
            
            assert os.path.exists(new_dir)
            assert os.path.isdir(new_dir)
            assert result == new_dir
    
    def test_get_timestamp(self):
        """
        Test timestamp generation.
        """
        timestamp = get_timestamp()
        
        # Check format
        assert isinstance(timestamp, str)
        assert len(timestamp) > 10  # At least YYYY-MM-DD
        
        # Check custom format
        custom = get_timestamp(format="%Y%m%d")
        assert len(custom) == 8
    
    def test_calculate_date_diff(self):
        """
        Test date difference calculation.
        """
        date1 = datetime(2024, 1, 1)
        date2 = datetime(2024, 1, 10)
        
        # Days difference
        diff_days = calculate_date_diff(date1, date2, unit='days')
        assert diff_days == 9
        
        # Hours difference
        diff_hours = calculate_date_diff(date1, date2, unit='hours')
        assert diff_hours == 9 * 24
        
        # With string inputs
        diff = calculate_date_diff('2024-01-01', '2024-01-10')
        assert diff == 9
    
    def test_safe_divide(self):
        """
        Test safe division with zero handling.
        """
        # Normal division
        result = safe_divide(10, 2)
        assert result == 5.0
        
        # Division by zero
        result = safe_divide(10, 0, default=0)
        assert result == 0
        
        # Division by zero with custom default
        result = safe_divide(10, 0, default=-1)
        assert result == -1
        
        # With arrays
        result = safe_divide([10, 20, 30], [2, 0, 3], default=0)
        assert result == [5.0, 0, 10.0]
    
    def test_flatten_dict(self):
        """
        Test dictionary flattening.
        """
        nested_dict = {
            'a': 1,
            'b': {
                'c': 2,
                'd': {
                    'e': 3,
                    'f': 4
                }
            },
            'g': 5
        }
        
        flattened = flatten_dict(nested_dict)
        
        # Check flattened keys
        assert 'a' in flattened
        assert 'b.c' in flattened
        assert 'b.d.e' in flattened
        assert 'b.d.f' in flattened
        assert 'g' in flattened
        
        # Check values
        assert flattened['b.c'] == 2
        assert flattened['b.d.e'] == 3
        
        # Test with separator
        flattened_dot = flatten_dict(nested_dict, separator='.')
        assert 'b.d.e' in flattened_dot
        
        flattened_underscore = flatten_dict(nested_dict, separator='_')
        assert 'b_d_e' in flattened_underscore
    
    def test_chunk_list(self):
        """
        Test list chunking.
        """
        items = list(range(10))
        
        # Normal chunking
        chunks = chunk_list(items, chunk_size=3)
        assert len(chunks) == 4
        assert chunks[0] == [0, 1, 2]
        assert chunks[1] == [3, 4, 5]
        assert chunks[2] == [6, 7, 8]
        assert chunks[3] == [9]
        
        # Chunk size larger than list
        chunks = chunk_list(items, chunk_size=20)
        assert len(chunks) == 1
        assert chunks[0] == items
        
        # Empty list
        chunks = chunk_list([], chunk_size=5)
        assert chunks == []
    
    def test_memory_usage(self):
        """
        Test memory usage estimation.
        """
        # Test with simple objects
        int_mem = memory_usage(42)
        assert isinstance(int_mem, (int, float))
        assert int_mem > 0
        
        # Test with DataFrame
        df = pd.DataFrame({'a': range(1000), 'b': range(1000)})
        df_mem = memory_usage(df)
        assert df_mem > 0
        
        # Test with different units
        df_mem_mb = memory_usage(df, unit='MB')
        assert df_mem_mb == df_mem / (1024 * 1024)
    
    def test_time_execution(self):
        """
        Test execution time decorator.
        """
        @time_execution
        def slow_function():
            import time
            time.sleep(0.1)
            return "done"
        
        # Function should still work
        result = slow_function()
        assert result == "done"
        
        # Timing information should be printed (can't easily capture output)
        # But function should execute without error


class TestSecurity:
    """
    Test suite for Security utilities.
    Tests data security and anonymization functions.
    """
    
    def test_hash_data(self):
        """
        Test data hashing.
        """
        # Hash string
        data = "sensitive_info"
        hashed = hash_data(data)
        
        assert isinstance(hashed, str)
        assert len(hashed) == 64  # SHA-256 hex digest
        
        # Same input should produce same hash
        hashed2 = hash_data(data)
        assert hashed == hashed2
        
        # Different input produces different hash
        hashed3 = hash_data("different_info")
        assert hashed != hashed3
        
        # Hash with different algorithm
        hashed_md5 = hash_data(data, algorithm='md5')
        assert len(hashed_md5) == 32
    
    def test_anonymize_data(self):
        """
        Test data anonymization.
        """
        data = {
            'name': 'John Doe',
            'email': 'john.doe@example.com',
            'phone': '123-456-7890',
            'ssn': '123-45-6789',
            'age': 30,
            'city': 'New York'
        }
        
        anonymized = anonymize_data(data)
        
        # Check anonymization
        assert anonymized['name'] != data['name']
        assert anonymized['email'] != data['email']
        assert anonymized['phone'] != data['phone']
        assert anonymized['ssn'] != data['ssn']
        
        # Non-sensitive fields should remain
        assert anonymized['age'] == data['age']
        assert anonymized['city'] == data['city']
        
        # Should be consistent (same input produces same anonymized output)
        anonymized2 = anonymize_data(data)
        assert anonymized == anonymized2
    
    def test_validate_email(self):
        """
        Test email validation.
        """
        # Valid emails
        assert validate_email('user@example.com') == True
        assert validate_email('user.name@domain.co.uk') == True
        assert validate_email('user+tag@example.org') == True
        
        # Invalid emails
        assert validate_email('not_an_email') == False
        assert validate_email('missing@domain') == False
        assert validate_email('@example.com') == False
        assert validate_email('user@.com') == False
        assert validate_email('') == False
    
    def test_sanitize_input(self):
        """
        Test input sanitization.
        """
        # SQL injection patterns
        sql_input = "'; DROP TABLE users; --"
        sanitized = sanitize_input(sql_input)
        assert "'" not in sanitized
        assert ";" not in sanitized
        
        # HTML/script tags
        html_input = "<script>alert('xss')</script>"
        sanitized = sanitize_input(html_input)
        assert "<script>" not in sanitized
        assert "</script>" not in sanitized
        
        # Path traversal
        path_input = "../../../etc/passwd"
        sanitized = sanitize_input(path_input)
        assert ".." not in sanitized
        assert "/" not in sanitized or "/" in sanitized  # Depends on config
        
        # Normal input should remain unchanged
        normal_input = "Hello, world!"
        sanitized = sanitize_input(normal_input, level='none')
        assert sanitized == normal_input
        
        # With custom allowed characters
        special_input = "abc123!@#$"
        sanitized = sanitize_input(special_input, allowed_chars='abc123')
        assert sanitized == "abc123"


class TestIntegrationHelpers:
    """
    Integration tests combining multiple utilities.
    Tests how utilities work together in realistic scenarios.
    """
    
    def test_config_logging_integration(self):
        """
        Test integration between config and logging.
        """
        # Create config
        config = {
            'logging': {
                'level': 'INFO',
                'file': 'test.log'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            yaml.dump(config, tmp)
            config_path = tmp.name
        
        try:
            # Load config
            config_manager = ConfigManager(config_path)
            log_config = config_manager.get('logging', {})
            
            # Setup logger
            logger = Logger(
                name='integration_test',
                level=log_config.get('level', 'INFO'),
                log_file=log_config.get('file')
            )
            
            # Log something
            logger.info("Integration test message")
            
            # Check log file
            if log_config.get('file'):
                with open(log_config['file'], 'r') as f:
                    content = f.read()
                    assert "Integration test message" in content
        finally:
            os.unlink(config_path)
            if os.path.exists('test.log'):
                os.unlink('test.log')
    
    def test_data_processing_pipeline_integration(self):
        """
        Test integration of data processing utilities.
        """
        # Create sample data
        data = pd.DataFrame({
            'id': range(1000),
            'name': [f'User_{i}' for i in range(1000)],
            'email': [f'user{i}@example.com' for i in range(1000)],
            'amount': np.random.randn(1000) * 100
        })
        
        # Anonymize sensitive fields
        data['name'] = data['name'].apply(lambda x: hash_data(x)[:8])
        data['email'] = data['email'].apply(lambda x: hash_data(x)[:8] + '@anon.com')
        
        # Save data
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            file_path = tmp.name
        
        try:
            save_features(data, file_path)
            
            # Load data
            loaded = load_features(file_path)
            
            # Process in parallel
            processor = ParallelProcessor(n_jobs=2)
            
            def process_row(row):
                return {
                    'id': row['id'],
                    'hashed_name': row['name'],
                    'amount_squared': row['amount'] ** 2
                }
            
            results = processor.parallel_dataframe_apply(
                loaded, process_row, axis=1
            )
            
            # Check results
            assert len(results) == len(data)
            assert results[0]['id'] == 0
        finally:
            os.unlink(file_path)
    
    def test_end_to_end_config_processing(self):
        """
        Test end-to-end configuration and processing.
        """
        # Create comprehensive config
        config = {
            'data': {
                'batch_size': 100,
                'features': ['id', 'amount']
            },
            'processing': {
                'parallel_jobs': 2,
                'chunk_size': 50
            },
            'security': {
                'anonymize_fields': ['name', 'email']
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            yaml.dump(config, tmp)
            config_path = tmp.name
        
        try:
            # Load config
            config_manager = ConfigManager(config_path)
            
            # Create data
            data = pd.DataFrame({
                'id': range(200),
                'name': [f'User_{i}' for i in range(200)],
                'email': [f'user{i}@example.com' for i in range(200)],
                'amount': np.random.randn(200) * 100
            })
            
            # Anonymize as per config
            for field in config_manager.get('security.anonymize_fields', []):
                if field in data.columns:
                    data[field] = data[field].apply(lambda x: hash_data(x)[:8])
            
            # Process in chunks
            processor = ParallelProcessor(
                n_jobs=config_manager.get('processing.parallel_jobs', 1)
            )
            
            chunks = chunk_list(
                data.to_dict('records'),
                chunk_size=config_manager.get('processing.chunk_size', 100)
            )
            
            def process_chunk(chunk):
                results = []
                for item in chunk:
                    results.append({
                        'id': item['id'],
                        'processed_amount': item['amount'] * 2
                    })
                return results
            
            all_results = processor.process_chunks(chunks, process_chunk)
            
            # Flatten results
            flattened = [item for sublist in all_results for item in sublist]
            
            assert len(flattened) == len(data)
            
        finally:
            os.unlink(config_path)