"""
Database Connectors Module
=========================
Provides robust database connections for various data stores used in fraud detection.
Includes connection pooling, retry logic, and comprehensive error handling for:
- PostgreSQL (transaction data, customer profiles)
- MongoDB (device fingerprints, session data)
- Redis (real-time features, caching)
- Cassandra (time-series transaction history)

Each connector implements best practices for production database access.
"""

import psycopg2
import psycopg2.extras
from psycopg2 import pool
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import redis
from redis import Redis
from redis.connection import ConnectionPool
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.policies import DCAwareRoundRobinPolicy, RetryPolicy
from contextlib import contextmanager
import pandas as pd
from typing import Optional, Dict, List, Any, Generator
import logging
from datetime import datetime, timedelta
import json
import time
from functools import wraps
import threading
from queue import Queue

logger = logging.getLogger(__name__)


def retry_on_failure(max_retries=3, delay=1, backoff=2):
    """
    Decorator for retrying database operations on failure.
    
    Implements exponential backoff between retries.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay in seconds
        backoff: Multiplier for exponential backoff
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                    
                    if attempt < max_retries - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff
            
            logger.error(f"All {max_retries} attempts failed")
            raise last_exception
        
        return wrapper
    
    return decorator


class PostgreSQLConnector:
    """
    PostgreSQL database connector with connection pooling.
    
    Handles connections to the main transaction database with:
    - Connection pooling for performance
    - Automatic reconnection
    - Query execution with retries
    - Batch operations
    - Transaction management
    """
    
    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        min_connections: int = 5,
        max_connections: int = 20,
        connection_timeout: int = 30
    ):
        """
        Initialize PostgreSQL connector with connection pool.
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Username
            password: Password
            min_connections: Minimum connections in pool
            max_connections: Maximum connections in pool
            connection_timeout: Connection timeout in seconds
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        
        self.connection_pool = None
        self._initialize_pool()
        
        logger.info(f"PostgreSQL connector initialized for {host}:{port}/{database}")
    
    def _initialize_pool(self):
        """Initialize connection pool."""
        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=self.min_connections,
                maxconn=self.max_connections,
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                connect_timeout=self.connection_timeout,
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=10,
                keepalives_count=5
            )
            logger.info("PostgreSQL connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {str(e)}")
            raise
    
    @contextmanager
    def get_connection(self) -> Generator:
        """
        Get a connection from the pool.
        
        Yields:
            psycopg2.extensions.connection: Database connection
        
        Raises:
            Exception: If connection cannot be obtained
        """
        connection = None
        try:
            connection = self.connection_pool.getconn()
            yield connection
        except Exception as e:
            logger.error(f"Error getting connection from pool: {str(e)}")
            raise
        finally:
            if connection:
                self.connection_pool.putconn(connection)
    
    @contextmanager
    def get_cursor(self, cursor_factory=None):
        """
        Get a cursor from a connection.
        
        Args:
            cursor_factory: Optional cursor factory (e.g., RealDictCursor)
            
        Yields:
            psycopg2.extensions.cursor: Database cursor
        """
        with self.get_connection() as connection:
            cursor = connection.cursor(cursor_factory=cursor_factory)
            try:
                yield cursor
                connection.commit()
            except Exception as e:
                connection.rollback()
                logger.error(f"Error in cursor operation: {str(e)}")
                raise
            finally:
                cursor.close()
    
    @retry_on_failure(max_retries=3)
    def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """
        Execute a query and return results as list of dictionaries.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List[Dict]: Query results
        """
        with self.get_cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()
    
    @retry_on_failure(max_retries=3)
    def execute_query_to_dataframe(self, query: str, params: tuple = None) -> pd.DataFrame:
        """
        Execute a query and return results as DataFrame.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            pd.DataFrame: Query results
        """
        with self.get_connection() as connection:
            return pd.read_sql_query(query, connection, params=params)
    
    @retry_on_failure(max_retries=3)
    def execute_batch(self, query: str, params_list: List[tuple]) -> int:
        """
        Execute batch operation (multiple rows).
        
        Args:
            query: SQL query string
            params_list: List of parameter tuples
            
        Returns:
            int: Number of rows affected
        """
        with self.get_cursor() as cursor:
            psycopg2.extras.execute_batch(cursor, query, params_list)
            return cursor.rowcount
    
    def get_transactions_by_customer(
        self,
        customer_id: str,
        start_date: datetime,
        end_date: datetime,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Get transactions for a specific customer within date range.
        
        Args:
            customer_id: Customer identifier
            start_date: Start date
            end_date: End date
            limit: Maximum number of transactions
            
        Returns:
            pd.DataFrame: Transaction data
        """
        query = """
            SELECT 
                transaction_id,
                customer_id,
                account_id,
                amount,
                currency,
                transaction_time,
                merchant_id,
                merchant_category,
                country,
                city,
                device_id,
                ip_address,
                is_fraud,
                fraud_score,
                created_at
            FROM transactions
            WHERE customer_id = %s
                AND transaction_time BETWEEN %s AND %s
            ORDER BY transaction_time DESC
            LIMIT %s
        """
        
        params = (customer_id, start_date, end_date, limit)
        
        return self.execute_query_to_dataframe(query, params)
    
    def insert_transaction(self, transaction_data: Dict) -> str:
        """
        Insert a single transaction.
        
        Args:
            transaction_data: Transaction data dictionary
            
        Returns:
            str: Inserted transaction ID
        """
        query = """
            INSERT INTO transactions (
                transaction_id, customer_id, account_id, amount,
                currency, transaction_time, merchant_id,
                merchant_category, country, city, device_id,
                ip_address, is_fraud, fraud_score, created_at
            ) VALUES (
                %(transaction_id)s, %(customer_id)s, %(account_id)s,
                %(amount)s, %(currency)s, %(transaction_time)s,
                %(merchant_id)s, %(merchant_category)s, %(country)s,
                %(city)s, %(device_id)s, %(ip_address)s,
                %(is_fraud)s, %(fraud_score)s, NOW()
            )
            RETURNING transaction_id
        """
        
        with self.get_cursor() as cursor:
            cursor.execute(query, transaction_data)
            result = cursor.fetchone()
            return result[0] if result else None
    
    def insert_transactions_batch(self, transactions_df: pd.DataFrame) -> int:
        """
        Insert multiple transactions efficiently.
        
        Args:
            transactions_df: DataFrame with transaction data
            
        Returns:
            int: Number of inserted rows
        """
        # Prepare data for insertion
        columns = [
            'transaction_id', 'customer_id', 'account_id', 'amount',
            'currency', 'transaction_time', 'merchant_id',
            'merchant_category', 'country', 'city', 'device_id',
            'ip_address', 'is_fraud', 'fraud_score'
        ]
        
        # Ensure all required columns exist
        for col in columns:
            if col not in transactions_df.columns:
                transactions_df[col] = None
        
        # Create list of tuples for batch insert
        data_tuples = [
            tuple(row[col] for col in columns)
            for _, row in transactions_df.iterrows()
        ]
        
        # Create INSERT query
        placeholders = ', '.join(['%s'] * len(columns))
        columns_str = ', '.join(columns)
        query = f"""
            INSERT INTO transactions ({columns_str})
            VALUES ({placeholders})
            ON CONFLICT (transaction_id) DO NOTHING
        """
        
        # Execute batch insert
        affected_rows = self.execute_batch(query, data_tuples)
        
        logger.info(f"Inserted {affected_rows} transactions")
        return affected_rows
    
    def get_customer_profile(self, customer_id: str) -> Dict:
        """
        Get customer profile information.
        
        Args:
            customer_id: Customer identifier
            
        Returns:
            Dict: Customer profile data
        """
        query = """
            SELECT 
                customer_id,
                first_name,
                last_name,
                email,
                phone,
                date_of_birth,
                address,
                city,
                country,
                postal_code,
                kyc_status,
                kyc_completed_at,
                risk_rating,
                account_created_at,
                last_login_at,
                total_transactions,
                total_spent,
                avg_transaction_amount,
                max_transaction_amount,
                preferred_currency
            FROM customers
            WHERE customer_id = %s
        """
        
        results = self.execute_query(query, (customer_id,))
        return results[0] if results else {}
    
    def update_risk_score(self, customer_id: str, risk_score: float) -> bool:
        """
        Update customer risk score.
        
        Args:
            customer_id: Customer identifier
            risk_score: New risk score
            
        Returns:
            bool: True if successful
        """
        query = """
            UPDATE customers
            SET risk_rating = %s,
                risk_updated_at = NOW()
            WHERE customer_id = %s
        """
        
        with self.get_cursor() as cursor:
            cursor.execute(query, (risk_score, customer_id))
            return cursor.rowcount > 0
    
    def close(self):
        """Close all connections in the pool."""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("PostgreSQL connection pool closed")


class MongoDBConnector:
    """
    MongoDB connector for document storage.
    
    Used for storing:
    - Device fingerprints
    - Session data
    - Unstructured fraud reports
    - Audit logs
    - User behavior profiles
    """
    
    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        username: str = None,
        password: str = None,
        replica_set: str = None,
        max_pool_size: int = 100,
        min_pool_size: int = 10,
        max_idle_time_ms: int = 10000,
        retry_writes: bool = True
    ):
        """
        Initialize MongoDB connector.
        
        Args:
            host: MongoDB host
            port: MongoDB port
            database: Database name
            username: Optional username
            password: Optional password
            replica_set: Optional replica set name
            max_pool_size: Maximum connection pool size
            min_pool_size: Minimum connection pool size
            max_idle_time_ms: Maximum idle time before closing connection
            retry_writes: Whether to retry write operations
        """
        self.host = host
        self.port = port
        self.database_name = database
        self.username = username
        self.password = password
        
        # Build connection URI
        auth_part = f"{username}:{password}@" if username and password else ""
        replica_part = f"/?replicaSet={replica_set}" if replica_set else ""
        
        self.connection_uri = f"mongodb://{auth_part}{host}:{port}{replica_part}"
        
        # Connection options
        self.options = {
            'maxPoolSize': max_pool_size,
            'minPoolSize': min_pool_size,
            'maxIdleTimeMS': max_idle_time_ms,
            'retryWrites': retry_writes,
            'serverSelectionTimeoutMS': 5000,
            'connectTimeoutMS': 10000,
            'socketTimeoutMS': 45000
        }
        
        self.client = None
        self.db = None
        self._connect()
        
        logger.info(f"MongoDB connector initialized for {host}:{port}/{database}")
    
    def _connect(self):
        """Establish connection to MongoDB."""
        try:
            self.client = MongoClient(self.connection_uri, **self.options)
            
            # Test connection
            self.client.admin.command('ping')
            
            self.db = self.client[self.database_name]
            logger.info("MongoDB connection established")
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise
        except ServerSelectionTimeoutError as e:
            logger.error(f"MongoDB server selection timeout: {str(e)}")
            raise
    
    @retry_on_failure(max_retries=3)
    def insert_document(self, collection: str, document: Dict) -> str:
        """
        Insert a single document.
        
        Args:
            collection: Collection name
            document: Document to insert
            
        Returns:
            str: Inserted document ID
        """
        # Add timestamps
        document['created_at'] = datetime.utcnow()
        document['updated_at'] = datetime.utcnow()
        
        result = self.db[collection].insert_one(document)
        return str(result.inserted_id)
    
    @retry_on_failure(max_retries=3)
    def insert_many_documents(self, collection: str, documents: List[Dict]) -> List[str]:
        """
        Insert multiple documents.
        
        Args:
            collection: Collection name
            documents: List of documents to insert
            
        Returns:
            List[str]: List of inserted document IDs
        """
        # Add timestamps
        for doc in documents:
            doc['created_at'] = datetime.utcnow()
            doc['updated_at'] = datetime.utcnow()
        
        result = self.db[collection].insert_many(documents)
        return [str(id) for id in result.inserted_ids]
    
    @retry_on_failure(max_retries=3)
    def find_documents(
        self,
        collection: str,
        filter_query: Dict,
        projection: Dict = None,
        limit: int = 0,
        sort: List[tuple] = None
    ) -> List[Dict]:
        """
        Find documents matching filter.
        
        Args:
            collection: Collection name
            filter_query: MongoDB filter query
            projection: Fields to return
            limit: Maximum number of documents
            sort: Sort criteria
            
        Returns:
            List[Dict]: Matching documents
        """
        cursor = self.db[collection].find(filter_query, projection)
        
        if sort:
            cursor = cursor.sort(sort)
        
        if limit > 0:
            cursor = cursor.limit(limit)
        
        return list(cursor)
    
    @retry_on_failure(max_retries=3)
    def find_one_document(self, collection: str, filter_query: Dict) -> Optional[Dict]:
        """
        Find a single document.
        
        Args:
            collection: Collection name
            filter_query: MongoDB filter query
            
        Returns:
            Optional[Dict]: Matching document or None
        """
        return self.db[collection].find_one(filter_query)
    
    @retry_on_failure(max_retries=3)
    def update_document(
        self,
        collection: str,
        filter_query: Dict,
        update: Dict,
        upsert: bool = False
    ) -> int:
        """
        Update documents matching filter.
        
        Args:
            collection: Collection name
            filter_query: MongoDB filter query
            update: Update operations
            upsert: Whether to insert if not exists
            
        Returns:
            int: Number of modified documents
        """
        # Add update timestamp
        if '$set' in update:
            update['$set']['updated_at'] = datetime.utcnow()
        else:
            update['$set'] = {'updated_at': datetime.utcnow()}
        
        result = self.db[collection].update_many(filter_query, update, upsert=upsert)
        return result.modified_count
    
    @retry_on_failure(max_retries=3)
    def delete_documents(self, collection: str, filter_query: Dict) -> int:
        """
        Delete documents matching filter.
        
        Args:
            collection: Collection name
            filter_query: MongoDB filter query
            
        Returns:
            int: Number of deleted documents
        """
        result = self.db[collection].delete_many(filter_query)
        return result.deleted_count
    
    def store_device_fingerprint(self, device_fingerprint: Dict) -> str:
        """
        Store device fingerprint data.
        
        Args:
            device_fingerprint: Device fingerprint data
            
        Returns:
            str: Document ID
        """
        collection = 'device_fingerprints'
        
        # Add indexed fields for quick lookup
        device_fingerprint['fingerprint_hash'] = device_fingerprint.get('fingerprint')
        
        # Store in device_fingerprints collection
        return self.insert_document(collection, device_fingerprint)
    
    def get_device_fingerprint(self, fingerprint_hash: str) -> Optional[Dict]:
        """
        Get device fingerprint by hash.
        
        Args:
            fingerprint_hash: Device fingerprint hash
            
        Returns:
            Optional[Dict]: Device fingerprint data
        """
        collection = 'device_fingerprints'
        return self.find_one_document(collection, {'fingerprint_hash': fingerprint_hash})
    
    def store_fraud_alert(self, alert_data: Dict) -> str:
        """
        Store fraud alert for investigation.
        
        Args:
            alert_data: Alert data
            
        Returns:
            str: Document ID
        """
        collection = 'fraud_alerts'
        
        # Add metadata
        alert_data['status'] = 'open'
        alert_data['assigned_to'] = None
        alert_data['investigation_notes'] = []
        alert_data['created_at'] = datetime.utcnow()
        
        return self.insert_document(collection, alert_data)
    
    def get_open_alerts(self, limit: int = 100) -> List[Dict]:
        """
        Get open fraud alerts.
        
        Args:
            limit: Maximum number of alerts
            
        Returns:
            List[Dict]: Open alerts
        """
        collection = 'fraud_alerts'
        filter_query = {'status': 'open'}
        sort = [('created_at', -1)]
        
        return self.find_documents(collection, filter_query, limit=limit, sort=sort)
    
    def store_audit_log(self, log_entry: Dict) -> str:
        """
        Store audit log entry.
        
        Args:
            log_entry: Log entry data
            
        Returns:
            str: Document ID
        """
        collection = 'audit_logs'
        
        # Add timestamp
        log_entry['timestamp'] = datetime.utcnow()
        
        return self.insert_document(collection, log_entry)
    
    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


class RedisConnector:
    """
    Redis connector for caching and real-time data.
    
    Used for:
    - Caching frequent queries
    - Real-time feature store
    - Session management
    - Rate limiting counters
    - Pub/sub for real-time alerts
    - Distributed locks
    """
    
    def __init__(
        self,
        host: str,
        port: int,
        password: str = None,
        db: int = 0,
        decode_responses: bool = True,
        max_connections: int = 50,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
        retry_on_timeout: bool = True,
        health_check_interval: int = 30
    ):
        """
        Initialize Redis connector with connection pool.
        
        Args:
            host: Redis host
            port: Redis port
            password: Optional password
            db: Database number
            decode_responses: Whether to decode responses to strings
            max_connections: Maximum connection pool size
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Connection timeout in seconds
            retry_on_timeout: Whether to retry on timeout
            health_check_interval: Health check interval in seconds
        """
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        
        # Create connection pool
        self.pool = ConnectionPool(
            host=host,
            port=port,
            password=password,
            db=db,
            decode_responses=decode_responses,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            retry_on_timeout=retry_on_timeout,
            health_check_interval=health_check_interval
        )
        
        # Create Redis client
        self.client = Redis(connection_pool=self.pool)
        
        # Test connection
        self._test_connection()
        
        logger.info(f"Redis connector initialized for {host}:{port}/{db}")
    
    def _test_connection(self):
        """Test Redis connection."""
        try:
            self.client.ping()
            logger.info("Redis connection successful")
        except Exception as e:
            logger.error(f"Redis connection failed: {str(e)}")
            raise
    
    @retry_on_failure(max_retries=3)
    def set(self, key: str, value: Any, expire: int = None) -> bool:
        """
        Set a key-value pair with optional expiration.
        
        Args:
            key: Key name
            value: Value (will be JSON serialized if not string)
            expire: Expiration time in seconds
            
        Returns:
            bool: True if successful
        """
        # Serialize value if not string
        if not isinstance(value, (str, bytes)):
            value = json.dumps(value, default=str)
        
        if expire:
            return self.client.setex(key, expire, value)
        else:
            return self.client.set(key, value)
    
    @retry_on_failure(max_retries=3)
    def get(self, key: str, deserialize: bool = True) -> Any:
        """
        Get value for a key.
        
        Args:
            key: Key name
            deserialize: Whether to deserialize JSON
            
        Returns:
            Any: Value
        """
        value = self.client.get(key)
        
        if value and deserialize:
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                pass
        
        return value
    
    @retry_on_failure(max_retries=3)
    def delete(self, *keys: str) -> int:
        """
        Delete one or more keys.
        
        Args:
            *keys: Keys to delete
            
        Returns:
            int: Number of keys deleted
        """
        return self.client.delete(*keys)
    
    @retry_on_failure(max_retries=3)
    def exists(self, key: str) -> bool:
        """
        Check if key exists.
        
        Args:
            key: Key name
            
        Returns:
            bool: True if key exists
        """
        return self.client.exists(key) > 0
    
    @retry_on_failure(max_retries=3)
    def expire(self, key: str, seconds: int) -> bool:
        """
        Set expiration on a key.
        
        Args:
            key: Key name
            seconds: Expiration time in seconds
            
        Returns:
            bool: True if successful
        """
        return self.client.expire(key, seconds)
    
    @retry_on_failure(max_retries=3)
    def incr(self, key: str, amount: int = 1) -> int:
        """
        Increment a counter.
        
        Args:
            key: Key name
            amount: Increment amount
            
        Returns:
            int: New value
        """
        return self.client.incr(key, amount)
    
    @retry_on_failure(max_retries=3)
    def decr(self, key: str, amount: int = 1) -> int:
        """
        Decrement a counter.
        
        Args:
            key: Key name
            amount: Decrement amount
            
        Returns:
            int: New value
        """
        return self.client.decr(key, amount)
    
    # Hash operations
    @retry_on_failure(max_retries=3)
    def hset(self, key: str, field: str, value: Any) -> int:
        """
        Set hash field.
        
        Args:
            key: Hash key
            field: Field name
            value: Field value
            
        Returns:
            int: 1 if new field, 0 if updated
        """
        # Serialize value if not string
        if not isinstance(value, (str, bytes)):
            value = json.dumps(value, default=str)
        
        return self.client.hset(key, field, value)
    
    @retry_on_failure(max_retries=3)
    def hget(self, key: str, field: str, deserialize: bool = True) -> Any:
        """
        Get hash field.
        
        Args:
            key: Hash key
            field: Field name
            deserialize: Whether to deserialize JSON
            
        Returns:
            Any: Field value
        """
        value = self.client.hget(key, field)
        
        if value and deserialize:
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                pass
        
        return value
    
    @retry_on_failure(max_retries=3)
    def hgetall(self, key: str, deserialize: bool = True) -> Dict:
        """
        Get all hash fields.
        
        Args:
            key: Hash key
            deserialize: Whether to deserialize JSON values
            
        Returns:
            Dict: All fields and values
        """
        result = self.client.hgetall(key)
        
        if deserialize:
            for field, value in result.items():
                try:
                    result[field] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    pass
        
        return result
    
    # List operations
    @retry_on_failure(max_retries=3)
    def lpush(self, key: str, *values) -> int:
        """
        Push values to the beginning of a list.
        
        Args:
            key: List key
            *values: Values to push
            
        Returns:
            int: Length of list after push
        """
        # Serialize non-string values
        serialized = []
        for value in values:
            if not isinstance(value, (str, bytes)):
                value = json.dumps(value, default=str)
            serialized.append(value)
        
        return self.client.lpush(key, *serialized)
    
    @retry_on_failure(max_retries=3)
    def rpop(self, key: str, deserialize: bool = True) -> Any:
        """
        Pop and return the last element of a list.
        
        Args:
            key: List key
            deserialize: Whether to deserialize JSON
            
        Returns:
            Any: Popped value
        """
        value = self.client.rpop(key)
        
        if value and deserialize:
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                pass
        
        return value
    
    @retry_on_failure(max_retries=3)
    def lrange(self, key: str, start: int, end: int, deserialize: bool = True) -> List:
        """
        Get range of elements from a list.
        
        Args:
            key: List key
            start: Start index
            end: End index
            deserialize: Whether to deserialize JSON
            
        Returns:
            List: List elements
        """
        values = self.client.lrange(key, start, end)
        
        if deserialize:
            deserialized = []
            for value in values:
                try:
                    deserialized.append(json.loads(value))
                except (json.JSONDecodeError, TypeError):
                    deserialized.append(value)
            return deserialized
        
        return values
    
    # Set operations
    @retry_on_failure(max_retries=3)
    def sadd(self, key: str, *members) -> int:
        """
        Add members to a set.
        
        Args:
            key: Set key
            *members: Members to add
            
        Returns:
            int: Number of new members added
        """
        return self.client.sadd(key, *members)
    
    @retry_on_failure(max_retries=3)
    def smembers(self, key: str) -> set:
        """
        Get all members of a set.
        
        Args:
            key: Set key
            
        Returns:
            set: Set members
        """
        return self.client.smembers(key)
    
    # Sorted set operations
    @retry_on_failure(max_retries=3)
    def zadd(self, key: str, mapping: Dict) -> int:
        """
        Add members to a sorted set.
        
        Args:
            key: Sorted set key
            mapping: Member-score mapping
            
        Returns:
            int: Number of new members added
        """
        return self.client.zadd(key, mapping)
    
    @retry_on_failure(max_retries=3)
    def zrangebyscore(
        self,
        key: str,
        min_score: float,
        max_score: float,
        withscores: bool = False
    ) -> List:
        """
        Get members with scores in range.
        
        Args:
            key: Sorted set key
            min_score: Minimum score
            max_score: Maximum score
            withscores: Whether to return scores
            
        Returns:
            List: Members (and optionally scores)
        """
        return self.client.zrangebyscore(
            key, min_score, max_score,
            withscores=withscores
        )
    
    # Pub/Sub
    def publish(self, channel: str, message: Any) -> int:
        """
        Publish a message to a channel.
        
        Args:
            channel: Channel name
            message: Message to publish
            
        Returns:
            int: Number of subscribers that received the message
        """
        # Serialize message if not string
        if not isinstance(message, (str, bytes)):
            message = json.dumps(message, default=str)
        
        return self.client.publish(channel, message)
    
    def subscribe(self, channel: str) -> redis.client.PubSub:
        """
        Subscribe to a channel.
        
        Args:
            channel: Channel name
            
        Returns:
            redis.client.PubSub: PubSub object
        """
        pubsub = self.client.pubsub()
        pubsub.subscribe(channel)
        return pubsub
    
    # Rate limiting
    def check_rate_limit(
        self,
        key: str,
        max_requests: int,
        time_window: int
    ) -> tuple:
        """
        Check rate limit using sliding window.
        
        Args:
            key: Rate limit key
            max_requests: Maximum requests allowed
            time_window: Time window in seconds
            
        Returns:
            tuple: (allowed: bool, current_count: int)
        """
        current_time = time.time()
        window_start = current_time - time_window
        
        # Clean old entries
        self.client.zremrangebyscore(key, 0, window_start)
        
        # Get current count
        current_count = self.client.zcard(key)
        
        if current_count >= max_requests:
            return False, current_count
        
        # Add current request
        self.client.zadd(key, {str(current_time): current_time})
        self.client.expire(key, time_window)
        
        return True, current_count + 1
    
    # Distributed lock
    def acquire_lock(self, lock_name: str, acquire_timeout: int = 10, lock_timeout: int = 30) -> bool:
        """
        Acquire a distributed lock.
        
        Args:
            lock_name: Lock name
            acquire_timeout: Timeout for acquiring lock
            lock_timeout: Lock expiration time
            
        Returns:
            bool: True if lock acquired
        """
        lock_key = f"lock:{lock_name}"
        identifier = str(time.time())
        
        end = time.time() + acquire_timeout
        while time.time() < end:
            # Try to set lock with NX (only if not exists)
            if self.client.set(lock_key, identifier, nx=True, ex=lock_timeout):
                return identifier
            
            time.sleep(0.1)
        
        return False
    
    def release_lock(self, lock_name: str, identifier: str) -> bool:
        """
        Release a distributed lock.
        
        Args:
            lock_name: Lock name
            identifier: Lock identifier from acquire_lock
            
        Returns:
            bool: True if lock released
        """
        lock_key = f"lock:{lock_name}"
        
        # Lua script to ensure we only release if we own the lock
        lua_script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
        """
        
        release_script = self.client.register_script(lua_script)
        result = release_script(keys=[lock_key], args=[identifier])
        
        return result == 1
    
    # Feature store operations
    def get_feature(self, feature_key: str) -> Any:
        """
        Get feature value from feature store.
        
        Args:
            feature_key: Feature key
            
        Returns:
            Any: Feature value
        """
        return self.get(feature_key)
    
    def set_feature(self, feature_key: str, value: Any, ttl: int = 3600) -> bool:
        """
        Set feature value in feature store.
        
        Args:
            feature_key: Feature key
            value: Feature value
            ttl: Time-to-live in seconds
            
        Returns:
            bool: True if successful
        """
        return self.set(feature_key, value, expire=ttl)
    
    def get_customer_features(self, customer_id: str) -> Dict:
        """
        Get all features for a customer.
        
        Args:
            customer_id: Customer identifier
            
        Returns:
            Dict: Customer features
        """
        feature_key = f"customer:features:{customer_id}"
        return self.hgetall(feature_key)
    
    def set_customer_feature(self, customer_id: str, feature_name: str, value: Any) -> bool:
        """
        Set a single customer feature.
        
        Args:
            customer_id: Customer identifier
            feature_name: Feature name
            value: Feature value
            
        Returns:
            bool: True if successful
        """
        feature_key = f"customer:features:{customer_id}"
        self.hset(feature_key, feature_name, value)
        
        # Set expiration (7 days)
        self.expire(feature_key, 604800)
        
        return True
    
    def close(self):
        """Close Redis connection."""
        if self.client:
            self.client.close()
            logger.info("Redis connection closed")


class CassandraConnector:
    """
    Cassandra connector for time-series data.
    
    Used for:
    - Transaction history
    - Customer behavior time series
    - Feature values over time
    - Audit logs
    - Event streams
    
    Cassandra provides high write throughput and time-based queries.
    """
    
    def __init__(
        self,
        contact_points: List[str],
        port: int = 9042,
        keyspace: str = None,
        username: str = None,
        password: str = None,
        datacenter: str = 'datacenter1',
        consistency_level: str = 'LOCAL_QUORUM'
    ):
        """
        Initialize Cassandra connector.
        
        Args:
            contact_points: List of contact point hosts
            port: CQL port
            keyspace: Keyspace name
            username: Optional username
            password: Optional password
            datacenter: Datacenter name
            consistency_level: Consistency level
        """
        self.contact_points = contact_points
        self.port = port
        self.keyspace = keyspace
        self.datacenter = datacenter
        
        # Authentication
        auth_provider = None
        if username and password:
            auth_provider = PlainTextAuthProvider(username=username, password=password)
        
        # Connection options
        self.cluster = Cluster(
            contact_points=contact_points,
            port=port,
            auth_provider=auth_provider,
            load_balancing_policy=DCAwareRoundRobinPolicy(local_dc=datacenter),
            default_retry_policy=RetryPolicy(),
            protocol_version=4,
            connect_timeout=10,
            control_connection_timeout=10
        )
        
        # Connect to keyspace if provided
        self.session = None
        if keyspace:
            self.set_keyspace(keyspace)
        
        # Consistency level mapping
        self.consistency_levels = {
            'ANY': 0,
            'ONE': 1,
            'TWO': 2,
            'THREE': 3,
            'QUORUM': 4,
            'ALL': 5,
            'LOCAL_QUORUM': 6,
            'EACH_QUORUM': 7,
            'SERIAL': 8,
            'LOCAL_SERIAL': 9,
            'LOCAL_ONE': 10
        }
        
        self.consistency_level = self.consistency_levels.get(consistency_level, 6)
        
        logger.info(f"Cassandra connector initialized for {contact_points}:{port}/{keyspace}")
    
    def set_keyspace(self, keyspace: str):
        """
        Set keyspace for the session.
        
        Args:
            keyspace: Keyspace name
        """
        if not self.session:
            self.session = self.cluster.connect()
        
        self.session.set_keyspace(keyspace)
        self.keyspace = keyspace
        logger.info(f"Set keyspace to {keyspace}")
    
    @retry_on_failure(max_retries=3)
    def execute_query(
        self,
        query: str,
        parameters: Dict = None,
        consistency_level: str = None
    ) -> List[Dict]:
        """
        Execute a CQL query.
        
        Args:
            query: CQL query string
            parameters: Query parameters
            consistency_level: Consistency level for this query
            
        Returns:
            List[Dict]: Query results
        """
        if not self.session:
            raise Exception("No Cassandra session. Call set_keyspace() first.")
        
        # Set consistency
        cl = self.consistency_level
        if consistency_level:
            cl = self.consistency_levels.get(consistency_level, cl)
        
        # Prepare and execute
        prepared = self.session.prepare(query)
        prepared.consistency_level = cl
        
        result = self.session.execute(prepared, parameters)
        
        # Convert to list of dicts
        columns = result.column_names
        return [dict(zip(columns, row)) for row in result]
    
    @retry_on_failure(max_retries=3)
    def execute_batch(self, queries: List[tuple]) -> bool:
        """
        Execute batch of queries.
        
        Args:
            queries: List of (query, parameters) tuples
            
        Returns:
            bool: True if successful
        """
        if not self.session:
            raise Exception("No Cassandra session. Call set_keyspace() first.")
        
        batch_query = "BEGIN BATCH "
        batch_query += "; ".join([q[0] for q in queries])
        batch_query += "; APPLY BATCH;"
        
        # Combine all parameters
        all_params = {}
        for i, (_, params) in enumerate(queries):
            if params:
                for key, value in params.items():
                    all_params[f"{key}_{i}"] = value
        
        prepared = self.session.prepare(batch_query)
        self.session.execute(prepared, all_params)
        
        return True
    
    def create_transactions_table(self):
        """Create transactions time-series table."""
        query = """
            CREATE TABLE IF NOT EXISTS transactions_by_time (
                date text,
                hour int,
                transaction_id text,
                customer_id text,
                amount decimal,
                currency text,
                merchant_id text,
                merchant_category text,
                country text,
                is_fraud boolean,
                fraud_score float,
                created_at timestamp,
                PRIMARY KEY ((date, hour), created_at, transaction_id)
            ) WITH CLUSTERING ORDER BY (created_at DESC, transaction_id ASC)
            AND compaction = {'class': 'TimeWindowCompactionStrategy'}
            AND default_time_to_live = 7776000  -- 90 days
        """
        self.execute_query(query)
        logger.info("Transactions table created/verified")
    
    def insert_transaction(self, transaction: Dict) -> bool:
        """
        Insert a transaction into time-series table.
        
        Args:
            transaction: Transaction data
            
        Returns:
            bool: True if successful
        """
        # Extract date and hour for partitioning
        transaction_time = transaction.get('transaction_time')
        if isinstance(transaction_time, str):
            transaction_time = datetime.fromisoformat(transaction_time.replace('Z', '+00:00'))
        
        date_key = transaction_time.strftime('%Y-%m-%d')
        hour_key = transaction_time.hour
        
        query = """
            INSERT INTO transactions_by_time (
                date, hour, transaction_id, customer_id,
                amount, currency, merchant_id, merchant_category,
                country, is_fraud, fraud_score, created_at
            ) VALUES (
                :date, :hour, :transaction_id, :customer_id,
                :amount, :currency, :merchant_id, :merchant_category,
                :country, :is_fraud, :fraud_score, :created_at
            )
        """
        
        params = {
            'date': date_key,
            'hour': hour_key,
            'transaction_id': transaction.get('transaction_id'),
            'customer_id': transaction.get('customer_id'),
            'amount': transaction.get('amount', 0.0),
            'currency': transaction.get('currency', 'USD'),
            'merchant_id': transaction.get('merchant_id'),
            'merchant_category': transaction.get('merchant_category'),
            'country': transaction.get('country'),
            'is_fraud': transaction.get('is_fraud', False),
            'fraud_score': transaction.get('fraud_score', 0.0),
            'created_at': transaction_time
        }
        
        self.execute_query(query, params)
        return True
    
    def get_transactions_by_date(
        self,
        date: datetime,
        hour: Optional[int] = None,
        limit: int = 1000
    ) -> List[Dict]:
        """
        Get transactions for a specific date/hour.
        
        Args:
            date: Date to query
            hour: Optional hour filter
            limit: Maximum number of transactions
            
        Returns:
            List[Dict]: Transactions
        """
        date_key = date.strftime('%Y-%m-%d')
        
        if hour is not None:
            query = """
                SELECT * FROM transactions_by_time
                WHERE date = :date AND hour = :hour
                ORDER BY created_at DESC
                LIMIT :limit
            """
            params = {'date': date_key, 'hour': hour, 'limit': limit}
        else:
            query = """
                SELECT * FROM transactions_by_time
                WHERE date = :date
                ORDER BY created_at DESC
                LIMIT :limit
            """
            params = {'date': date_key, 'limit': limit}
        
        return self.execute_query(query, params)
    
    def get_customer_transactions(
        self,
        customer_id: str,
        start_date: datetime,
        end_date: datetime,
        limit: int = 1000
    ) -> List[Dict]:
        """
        Get transactions for a customer within date range.
        
        Args:
            customer_id: Customer identifier
            start_date: Start date
            end_date: End date
            limit: Maximum number of transactions
            
        Returns:
            List[Dict]: Transactions
        """
        # This query requires an index on customer_id
        # In production, you'd have a separate table by customer
        query = """
            SELECT * FROM transactions_by_time
            WHERE customer_id = :customer_id
                AND created_at >= :start_date
                AND created_at <= :end_date
            ORDER BY created_at DESC
            LIMIT :limit
            ALLOW FILTERING
        """
        
        params = {
            'customer_id': customer_id,
            'start_date': start_date,
            'end_date': end_date,
            'limit': limit
        }
        
        return self.execute_query(query, params)
    
    def close(self):
        """Close Cassandra connection."""
        if self.cluster:
            self.cluster.shutdown()
            logger.info("Cassandra connection closed")


class DatabaseConnectorFactory:
    """
    Factory class for creating and managing database connectors.
    
    Provides centralized creation and configuration of all database
    connectors with dependency injection and lifecycle management.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize factory with configuration.
        
        Args:
            config: Configuration dictionary with database settings
        """
        self.config = config
        self.connectors = {}
        
        logger.info("Database connector factory initialized")
    
    def get_postgresql_connector(self) -> PostgreSQLConnector:
        """
        Get or create PostgreSQL connector.
        
        Returns:
            PostgreSQLConnector: Configured PostgreSQL connector
        """
        if 'postgresql' not in self.connectors:
            pg_config = self.config.get('postgresql', {})
            self.connectors['postgresql'] = PostgreSQLConnector(
                host=pg_config.get('host', 'localhost'),
                port=pg_config.get('port', 5432),
                database=pg_config.get('database'),
                user=pg_config.get('user'),
                password=pg_config.get('password'),
                min_connections=pg_config.get('min_connections', 5),
                max_connections=pg_config.get('max_connections', 20)
            )
        
        return self.connectors['postgresql']
    
    def get_mongodb_connector(self) -> MongoDBConnector:
        """
        Get or create MongoDB connector.
        
        Returns:
            MongoDBConnector: Configured MongoDB connector
        """
        if 'mongodb' not in self.connectors:
            mongo_config = self.config.get('mongodb', {})
            self.connectors['mongodb'] = MongoDBConnector(
                host=mongo_config.get('host', 'localhost'),
                port=mongo_config.get('port', 27017),
                database=mongo_config.get('database'),
                username=mongo_config.get('username'),
                password=mongo_config.get('password')
            )
        
        return self.connectors['mongodb']
    
    def get_redis_connector(self) -> RedisConnector:
        """
        Get or create Redis connector.
        
        Returns:
            RedisConnector: Configured Redis connector
        """
        if 'redis' not in self.connectors:
            redis_config = self.config.get('redis', {})
            self.connectors['redis'] = RedisConnector(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                password=redis_config.get('password'),
                db=redis_config.get('db', 0)
            )
        
        return self.connectors['redis']
    
    def get_cassandra_connector(self) -> CassandraConnector:
        """
        Get or create Cassandra connector.
        
        Returns:
            CassandraConnector: Configured Cassandra connector
        """
        if 'cassandra' not in self.connectors:
            cassandra_config = self.config.get('cassandra', {})
            self.connectors['cassandra'] = CassandraConnector(
                contact_points=cassandra_config.get('contact_points', ['localhost']),
                port=cassandra_config.get('port', 9042),
                keyspace=cassandra_config.get('keyspace'),
                username=cassandra_config.get('username'),
                password=cassandra_config.get('password')
            )
        
        return self.connectors['cassandra']
    
    def close_all(self):
        """Close all database connections."""
        for name, connector in self.connectors.items():
            try:
                if hasattr(connector, 'close'):
                    connector.close()
                logger.info(f"Closed connector: {name}")
            except Exception as e:
                logger.error(f"Error closing connector {name}: {str(e)}")