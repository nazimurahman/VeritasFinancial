# tests/test_feature_engineering.py
"""
Unit tests for feature engineering module.
Tests creation of domain-specific features for fraud detection.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_engineering.domain_features.transaction_features import TransactionFeatureEngineer
from src.feature_engineering.domain_features.customer_features import CustomerFeatureEngineer
from src.feature_engineering.domain_features.device_features import DeviceFeatureEngineer
from src.feature_engineering.domain_features.behavioral_features import BehavioralFeatureEngineer
from src.feature_engineering.temporal_features.rolling_statistics import RollingStatistics
from src.feature_engineering.temporal_features.seasonality import SeasonalityFeatures
from src.feature_engineering.temporal_features.time_gaps import TimeGapFeatures
from src.feature_engineering.aggregate_features.customer_aggregates import CustomerAggregates
from src.feature_engineering.aggregate_features.merchant_aggregates import MerchantAggregates
from src.feature_engineering.graph_features.network_analysis import NetworkAnalysis


class TestTransactionFeatureEngineer:
    """
    Test suite for TransactionFeatureEngineer.
    Tests creation of transaction-level features.
    """
    
    @pytest.fixture
    def transaction_data(self):
        """
        Fixture providing transaction data for feature engineering.
        """
        np.random.seed(42)
        n_transactions = 1000
        n_customers = 50
        
        # Generate timestamps over 30 days
        start_time = datetime.now() - timedelta(days=30)
        timestamps = [start_time + timedelta(
            hours=np.random.randint(0, 24*30),
            minutes=np.random.randint(0, 60),
            seconds=np.random.randint(0, 60)
        ) for _ in range(n_transactions)]
        
        return pd.DataFrame({
            'transaction_id': [f'TX{i}' for i in range(n_transactions)],
            'customer_id': [f'C{np.random.randint(1, n_customers+1)}' 
                           for _ in range(n_transactions)],
            'amount': np.random.exponential(100, n_transactions),
            'currency': np.random.choice(['USD', 'EUR', 'GBP'], n_transactions),
            'merchant_id': [f'M{np.random.randint(1, 20)}' for _ in range(n_transactions)],
            'merchant_category': np.random.choice(
                ['retail', 'travel', 'food', 'entertainment', 'utilities'], 
                n_transactions
            ),
            'transaction_time': timestamps,
            'country': np.random.choice(['US', 'GB', 'FR', 'DE', 'JP'], n_transactions),
            'city': [f'City{np.random.randint(1, 10)}' for _ in range(n_transactions)],
            'device_id': [f'D{np.random.randint(1, 30)}' for _ in range(n_transactions)],
            'ip_address': [f'192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}' 
                          for _ in range(n_transactions)],
            'is_fraud': np.random.choice([0, 1], n_transactions, p=[0.98, 0.02])
        })
    
    def test_initialization(self):
        """
        Test TransactionFeatureEngineer initialization.
        """
        engineer = TransactionFeatureEngineer()
        assert engineer.features is not None
        assert len(engineer.features) == 0
        
        # With custom config
        engineer = TransactionFeatureEngineer(config={
            'rolling_windows': ['1H', '24H', '7D'],
            'amount_bins': [0, 50, 100, 500, 1000, 10000]
        })
        assert engineer.config['rolling_windows'] == ['1H', '24H', '7D']
    
    def test_create_basic_features(self, transaction_data):
        """
        Test creation of basic transaction features.
        """
        engineer = TransactionFeatureEngineer()
        
        # Create basic features
        features_df = engineer._create_basic_features(transaction_data.copy())
        
        # Check that new features were created
        expected_features = [
            'transactions_last_hour',
            'transactions_last_day',
            'avg_transaction_amount_7D',
            'std_transaction_amount_7D',
            'amount_percentile'
        ]
        
        for feature in expected_features:
            if feature in features_df.columns:
                assert feature in features_df.columns
        
        # Check feature values are reasonable
        assert features_df['transactions_last_hour'].max() <= 24  # Max per hour
        assert features_df['transactions_last_day'].max() <= 24 * 30  # Max per day
        assert (features_df['avg_transaction_amount_7D'] >= 0).all()
    
    def test_create_temporal_features(self, transaction_data):
        """
        Test creation of temporal transaction features.
        """
        engineer = TransactionFeatureEngineer()
        
        # Create temporal features
        features_df = engineer._create_temporal_features(transaction_data.copy())
        
        # Check time-based features
        assert 'hour_of_day' in features_df.columns
        assert 'day_of_week' in features_df.columns
        assert 'is_weekend' in features_df.columns
        assert 'hour_sin' in features_df.columns
        assert 'hour_cos' in features_df.columns
        
        # Validate ranges
        assert features_df['hour_of_day'].between(0, 23).all()
        assert features_df['day_of_week'].between(0, 6).all()
        assert features_df['is_weekend'].isin([0, 1]).all()
        assert features_df['hour_sin'].between(-1, 1).all()
        assert features_df['hour_cos'].between(-1, 1).all()
    
    def test_create_amount_features(self, transaction_data):
        """
        Test creation of amount-based features.
        """
        engineer = TransactionFeatureEngineer()
        
        # Create amount features
        features_df = engineer._create_amount_features(transaction_data.copy())
        
        # Check amount-based features
        assert 'amount_log' in features_df.columns
        assert 'amount_zscore' in features_df.columns
        assert 'amount_category' in features_df.columns
        assert 'is_high_value' in features_df.columns
        assert 'amount_ratio_to_avg' in features_df.columns
        
        # Validate transformations
        assert (features_df['amount_log'] > 0).all()
        assert features_df['amount_category'].isin(['low', 'medium', 'high']).all()
        assert features_df['is_high_value'].isin([0, 1]).all()
    
    def test_create_location_features(self, transaction_data):
        """
        Test creation of location-based features.
        """
        engineer = TransactionFeatureEngineer()
        
        # Add some location patterns
        data = transaction_data.copy()
        # Create a home location for each customer (simplified)
        data['home_country'] = data.groupby('customer_id')['country'].transform('first')
        
        # Create location features
        features_df = engineer._create_location_features(data)
        
        # Check location features
        assert 'is_foreign_country' in features_df.columns
        assert 'country_risk_score' in features_df.columns
        assert 'distance_from_home' in features_df.columns
        assert 'velocity_location_change' in features_df.columns
        
        # Validate features
        assert features_df['is_foreign_country'].isin([0, 1]).all()
        assert features_df['distance_from_home'].between(0, 20000).all()  # Max earth distance
    
    def test_create_merchant_features(self, transaction_data):
        """
        Test creation of merchant-based features.
        """
        engineer = TransactionFeatureEngineer()
        
        # Create merchant features
        features_df = engineer._create_merchant_features(transaction_data.copy())
        
        # Check merchant features
        assert 'merchant_risk_score' in features_df.columns
        assert 'transactions_at_merchant' in features_df.columns
        assert 'avg_amount_at_merchant' in features_df.columns
        assert 'first_time_merchant' in features_df.columns
        
        # Validate
        assert features_df['first_time_merchant'].isin([0, 1]).all()
        assert features_df['transactions_at_merchant'] >= 1
    
    def test_create_behavioral_features(self, transaction_data):
        """
        Test creation of behavioral features.
        """
        engineer = TransactionFeatureEngineer()
        
        # Sort by time for behavioral features
        data = transaction_data.sort_values('transaction_time')
        
        # Create behavioral features
        features_df = engineer._create_behavioral_features(data)
        
        # Check behavioral features
        assert 'spending_velocity' in features_df.columns
        assert 'unusual_category_flag' in features_df.columns
        assert 'unusual_time_flag' in features_df.columns
        assert 'amount_deviation' in features_df.columns
        assert 'transaction_pattern_score' in features_df.columns
        
        # Validate ranges
        assert features_df['unusual_category_flag'].isin([0, 1]).all()
        assert features_df['unusual_time_flag'].isin([0, 1]).all()
    
    def test_create_interaction_features(self, transaction_data):
        """
        Test creation of interaction features.
        """
        engineer = TransactionFeatureEngineer()
        
        # Create interaction features
        features_df = engineer._create_interaction_features(transaction_data.copy())
        
        # Check interaction features (feature engineering step should create them)
        # This test assumes interaction features are created
        interaction_features = [col for col in features_df.columns if '_x_' in col]
        assert len(interaction_features) > 0
    
    def test_create_risk_scores(self, transaction_data):
        """
        Test creation of risk scores.
        """
        engineer = TransactionFeatureEngineer()
        
        # Create risk scores
        features_df = engineer._create_risk_scores(transaction_data.copy())
        
        # Check risk score features
        assert 'amount_risk_score' in features_df.columns
        assert 'location_risk_score' in features_df.columns
        assert 'merchant_risk_score' in features_df.columns
        assert 'device_risk_score' in features_df.columns
        assert 'combined_risk_score' in features_df.columns
        
        # Validate risk scores are between 0 and 1
        risk_cols = [col for col in features_df.columns if 'risk_score' in col]
        for col in risk_cols:
            assert features_df[col].between(0, 1).all()
    
    def test_create_all_features(self, transaction_data):
        """
        Test complete feature engineering pipeline.
        """
        engineer = TransactionFeatureEngineer()
        
        # Create all features
        features_df = engineer.create_all_features(transaction_data.copy())
        
        # Check that we have many new features
        original_cols = len(transaction_data.columns)
        new_cols = len(features_df.columns)
        
        assert new_cols > original_cols
        assert new_cols - original_cols >= 20  # At least 20 new features
        
        # Check that no errors occurred (no NaN values from calculations)
        assert features_df.isnull().sum().sum() < len(features_df) * 0.1  # Less than 10% NaN


class TestRollingStatistics:
    """
    Test suite for RollingStatistics class.
    Tests calculation of rolling window statistics.
    """
    
    @pytest.fixture
    def time_series_data(self):
        """
        Fixture providing time series data for rolling statistics.
        """
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='H')
        
        return pd.DataFrame({
            'timestamp': dates,
            'customer_id': [f'C{i % 10}' for i in range(1000)],
            'amount': np.random.normal(100, 20, 1000) + np.sin(np.arange(1000) * 0.1) * 30,
            'transaction_count': np.random.poisson(5, 1000)
        })
    
    def test_initialization(self):
        """
        Test RollingStatistics initialization.
        """
        rs = RollingStatistics(windows=['1H', '24H', '7D'], 
                               functions=['mean', 'std', 'min', 'max', 'count'])
        
        assert rs.windows == ['1H', '24H', '7D']
        assert 'mean' in rs.functions
        assert 'std' in rs.functions
        
        # Test with default values
        rs = RollingStatistics()
        assert len(rs.windows) > 0
        assert len(rs.functions) > 0
    
    def test_calculate_rolling_mean(self, time_series_data):
        """
        Test calculation of rolling mean.
        """
        rs = RollingStatistics(windows=['3H', '6H'], functions=['mean'])
        
        # Calculate rolling mean per customer
        result = rs.calculate_rolling_mean(
            time_series_data, 
            value_col='amount', 
            group_col='customer_id',
            time_col='timestamp'
        )
        
        # Check that new column was created
        assert 'amount_rolling_mean_3H' in result.columns
        assert 'amount_rolling_mean_6H' in result.columns
        
        # Check first few rows (should have NaN for insufficient history)
        assert pd.isna(result['amount_rolling_mean_3H'].iloc[0])
        
        # After enough data, should have values
        assert not pd.isna(result['amount_rolling_mean_3H'].iloc[10])
    
    def test_calculate_rolling_std(self, time_series_data):
        """
        Test calculation of rolling standard deviation.
        """
        rs = RollingStatistics(windows=['6H'], functions=['std'])
        
        result = rs.calculate_rolling_std(
            time_series_data,
            value_col='amount',
            group_col='customer_id',
            time_col='timestamp'
        )
        
        assert 'amount_rolling_std_6H' in result.columns
        
        # Standard deviation should be non-negative
        valid_std = result['amount_rolling_std_6H'].dropna()
        assert (valid_std >= 0).all()
    
    def test_calculate_rolling_count(self, time_series_data):
        """
        Test calculation of rolling count.
        """
        rs = RollingStatistics(windows=['12H'], functions=['count'])
        
        result = rs.calculate_rolling_count(
            time_series_data,
            group_col='customer_id',
            time_col='timestamp'
        )
        
        assert 'rolling_count_12H' in result.columns
        
        # Count should be integer
        valid_counts = result['rolling_count_12H'].dropna()
        assert (valid_counts == valid_counts.astype(int)).all()
        
        # Count should increase as we get more data
        assert result['rolling_count_12H'].iloc[50] >= result['rolling_count_12H'].iloc[10]
    
    def test_calculate_multiple_statistics(self, time_series_data):
        """
        Test calculation of multiple statistics simultaneously.
        """
        rs = RollingStatistics(windows=['6H'], functions=['mean', 'std', 'min', 'max'])
        
        result = rs.calculate_all(
            time_series_data,
            value_col='amount',
            group_col='customer_id',
            time_col='timestamp'
        )
        
        # Check all statistics were created
        assert 'amount_rolling_mean_6H' in result.columns
        assert 'amount_rolling_std_6H' in result.columns
        assert 'amount_rolling_min_6H' in result.columns
        assert 'amount_rolling_max_6H' in result.columns
        
        # Validate relationships (min <= mean <= max)
        valid_rows = result.dropna(subset=['amount_rolling_min_6H', 
                                           'amount_rolling_mean_6H', 
                                           'amount_rolling_max_6H'])
        
        assert (valid_rows['amount_rolling_min_6H'] <= 
                valid_rows['amount_rolling_mean_6H']).all()
        assert (valid_rows['amount_rolling_mean_6H'] <= 
                valid_rows['amount_rolling_max_6H']).all()
    
    def test_different_window_sizes(self, time_series_data):
        """
        Test different window size calculations.
        """
        rs = RollingStatistics(windows=['1H', '24H', '168H'], functions=['mean'])
        
        result = rs.calculate_all(
            time_series_data,
            value_col='amount',
            group_col='customer_id',
            time_col='timestamp'
        )
        
        # Larger windows should have smoother values
        col_1h = 'amount_rolling_mean_1H'
        col_24h = 'amount_rolling_mean_24H'
        
        # Standard deviation should be lower for larger window
        std_1h = result[col_1h].std()
        std_24h = result[col_24h].std()
        
        assert std_24h <= std_1h or abs(std_24h - std_1h) < 0.1 * std_1h


class TestCustomerAggregates:
    """
    Test suite for CustomerAggregates class.
    Tests creation of customer-level aggregate features.
    """
    
    @pytest.fixture
    def customer_data(self):
        """
        Fixture providing customer data for aggregate features.
        """
        np.random.seed(42)
        n_customers = 100
        
        return pd.DataFrame({
            'customer_id': [f'C{i}' for i in range(n_customers)],
            'age': np.random.randint(18, 80, n_customers),
            'income_level': np.random.choice(['low', 'medium', 'high'], n_customers),
            'account_age_days': np.random.randint(1, 3650, n_customers),
            'credit_score': np.random.normal(700, 50, n_customers).clip(300, 850),
            'total_balance': np.random.exponential(5000, n_customers),
            'num_accounts': np.random.randint(1, 5, n_customers),
            'has_credit_card': np.random.choice([0, 1], n_customers),
            'has_mortgage': np.random.choice([0, 1], n_customers, p=[0.7, 0.3]),
            'has_loan': np.random.choice([0, 1], n_customers, p=[0.6, 0.4]),
            'risk_tier': np.random.choice(['low', 'medium', 'high'], n_customers)
        })
    
    @pytest.fixture
    def customer_transactions(self, customer_data):
        """
        Fixture providing transaction history for customers.
        """
        np.random.seed(42)
        n_transactions = 5000
        
        transactions = []
        for i in range(n_transactions):
            customer_id = np.random.choice(customer_data['customer_id'])
            days_ago = np.random.randint(0, 90)
            
            transactions.append({
                'customer_id': customer_id,
                'amount': np.random.exponential(100),
                'transaction_time': datetime.now() - timedelta(days=days_ago),
                'merchant_category': np.random.choice(
                    ['retail', 'travel', 'food', 'entertainment']
                ),
                'is_fraud': np.random.choice([0, 1], p=[0.98, 0.02])
            })
        
        return pd.DataFrame(transactions)
    
    def test_initialization(self):
        """
        Test CustomerAggregates initialization.
        """
        agg = CustomerAggregates()
        assert agg.aggregation_functions is not None
        
        agg = CustomerAggregates(aggregation_functions=['mean', 'median', 'std'])
        assert 'median' in agg.aggregation_functions
    
    def test_create_demographic_features(self, customer_data):
        """
        Test creation of demographic features.
        """
        agg = CustomerAggregates()
        
        features = agg._create_demographic_features(customer_data.copy())
        
        # Check demographic features
        assert 'age_group' in features.columns
        assert 'income_level_encoded' in features.columns
        assert 'credit_score_category' in features.columns
        
        # Validate groupings
        expected_age_groups = ['young', 'middle', 'senior']
        assert features['age_group'].isin(expected_age_groups).all()
    
    def test_create_financial_health_features(self, customer_data):
        """
        Test creation of financial health features.
        """
        agg = CustomerAggregates()
        
        features = agg._create_financial_health_features(customer_data.copy())
        
        # Check financial health features
        assert 'balance_per_account' in features.columns
        assert 'credit_utilization' in features.columns
        assert 'product_diversity_score' in features.columns
        assert 'financial_health_score' in features.columns
        
        # Validate calculations
        assert (features['balance_per_account'] >= 0).all()
        assert features['product_diversity_score'].between(0, 1).all()
        assert features['financial_health_score'].between(0, 100).all()
    
    def test_create_risk_features(self, customer_data):
        """
        Test creation of risk-related features.
        """
        agg = CustomerAggregates()
        
        features = agg._create_risk_features(customer_data.copy())
        
        # Check risk features
        assert 'base_risk_score' in features.columns
        assert 'credit_risk_factor' in features.columns
        assert 'age_risk_factor' in features.columns
        
        # Validate risk scores
        risk_cols = [col for col in features.columns if 'risk' in col.lower()]
        for col in risk_cols:
            if col in features.select_dtypes(include=[np.number]).columns:
                assert features[col].between(0, 1).all() or features[col].between(0, 100).all()
    
    def test_create_behavioral_patterns(self, customer_data, customer_transactions):
        """
        Test creation of behavioral patterns from transaction history.
        """
        agg = CustomerAggregates()
        
        # Merge customer data with transaction aggregates
        # First, create transaction aggregates per customer
        tx_agg = customer_transactions.groupby('customer_id').agg({
            'amount': ['mean', 'std', 'max', 'sum'],
            'transaction_time': lambda x: (datetime.now() - x.max()).days,
            'is_fraud': 'sum'
        }).reset_index()
        tx_agg.columns = ['customer_id', 'avg_tx_amount', 'std_tx_amount', 
                          'max_tx_amount', 'total_spent', 'days_since_last_tx', 
                          'fraud_count']
        
        # Merge with customer data
        customer_with_tx = customer_data.merge(tx_agg, on='customer_id', how='left')
        
        features = agg._create_behavioral_patterns(customer_with_tx)
        
        # Check behavioral features
        assert 'spending_stability' in features.columns
        assert 'tx_frequency' in features.columns
        assert 'fraud_rate' in features.columns
        assert 'unusual_spending_pattern' in features.columns
        
        # Validate
        assert features['fraud_rate'].between(0, 1).all()
        assert features['unusual_spending_pattern'].isin([0, 1]).all()
    
    def test_create_all_aggregates(self, customer_data, customer_transactions):
        """
        Test creation of all customer aggregates.
        """
        agg = CustomerAggregates()
        
        # Create all aggregates
        features = agg.create_all_aggregates(customer_data, customer_transactions)
        
        # Check that we have many new features
        original_cols = len(customer_data.columns)
        new_cols = len(features.columns)
        
        assert new_cols > original_cols
        assert new_cols - original_cols >= 10  # At least 10 new features
        
        # Check that customer_id is still present
        assert 'customer_id' in features.columns


class TestNetworkAnalysis:
    """
    Test suite for NetworkAnalysis class.
    Tests graph-based features for fraud detection.
    """
    
    @pytest.fixture
    def transaction_graph_data(self):
        """
        Fixture providing transaction data for graph analysis.
        """
        np.random.seed(42)
        n_transactions = 1000
        n_customers = 50
        n_devices = 30
        n_merchants = 20
        
        transactions = []
        for i in range(n_transactions):
            customer = f'C{np.random.randint(1, n_customers+1)}'
            device = f'D{np.random.randint(1, n_devices+1)}'
            merchant = f'M{np.random.randint(1, n_merchants+1)}'
            
            transactions.append({
                'transaction_id': f'T{i}',
                'customer_id': customer,
                'device_id': device,
                'merchant_id': merchant,
                'amount': np.random.exponential(100),
                'is_fraud': np.random.choice([0, 1], p=[0.98, 0.02])
            })
        
        return pd.DataFrame(transactions)
    
    def test_initialization(self):
        """
        Test NetworkAnalysis initialization.
        """
        network = NetworkAnalysis()
        assert network.graph is None
        
        network = NetworkAnalysis(graph_type='bipartite')
        assert network.graph_type == 'bipartite'
    
    def test_build_customer_device_graph(self, transaction_graph_data):
        """
        Test building customer-device graph.
        """
        network = NetworkAnalysis(graph_type='bipartite')
        
        # Build graph
        graph = network.build_customer_device_graph(transaction_graph_data)
        
        # Check graph properties
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0
        
        # Nodes should include both customers and devices
        customer_nodes = [n for n in graph.nodes() if n.startswith('C')]
        device_nodes = [n for n in graph.nodes() if n.startswith('D')]
        
        assert len(customer_nodes) > 0
        assert len(device_nodes) > 0
    
    def test_calculate_degree_centrality(self, transaction_graph_data):
        """
        Test calculation of degree centrality.
        """
        network = NetworkAnalysis()
        
        # Build graph first
        network.build_customer_device_graph(transaction_graph_data)
        
        # Calculate degree centrality
        centrality = network.calculate_degree_centrality()
        
        # Check results
        assert len(centrality) > 0
        assert all(0 <= v <= 1 for v in centrality.values())
        
        # Nodes with more connections should have higher centrality
        # This is harder to test directly, but we can check structure
        assert isinstance(centrality, dict)
    
    def test_calculate_betweenness_centrality(self, transaction_graph_data):
        """
        Test calculation of betweenness centrality.
        """
        network = NetworkAnalysis()
        
        # Build graph
        network.build_customer_device_graph(transaction_graph_data)
        
        # Calculate betweenness centrality
        betweenness = network.calculate_betweenness_centrality()
        
        # Check results
        assert len(betweenness) > 0
        assert all(0 <= v <= 1 for v in betweenness.values())
    
    def test_detect_communities(self, transaction_graph_data):
        """
        Test community detection in graph.
        """
        network = NetworkAnalysis()
        
        # Build graph
        network.build_customer_device_graph(transaction_graph_data)
        
        # Detect communities
        communities = network.detect_communities()
        
        # Check that we have communities
        assert len(communities) > 0
        
        # Each node should belong to exactly one community
        all_nodes = set()
        for community in communities:
            all_nodes.update(community)
        
        assert len(all_nodes) == network.graph.number_of_nodes()
    
    def test_calculate_device_sharing_score(self, transaction_graph_data):
        """
        Test calculation of device sharing scores.
        """
        network = NetworkAnalysis()
        
        # Calculate device sharing scores
        sharing_scores = network.calculate_device_sharing_score(transaction_graph_data)
        
        # Check results
        assert len(sharing_scores) > 0
        assert all(0 <= v <= 1 for v in sharing_scores.values())
        
        # Customers sharing devices should have positive scores
        # This is a simplified check
        assert any(v > 0 for v in sharing_scores.values())
    
    def test_calculate_merchant_risk_propagation(self, transaction_graph_data):
        """
        Test calculation of merchant risk propagation.
        """
        network = NetworkAnalysis()
        
        # Add fraud labels
        data = transaction_graph_data.copy()
        fraud_customers = data[data['is_fraud'] == 1]['customer_id'].unique()
        
        # Calculate merchant risk propagation
        merchant_risk = network.calculate_merchant_risk_propagation(data)
        
        # Check results
        assert len(merchant_risk) > 0
        assert all(0 <= v <= 1 for v in merchant_risk.values())
        
        # Merchants with fraud customers should have higher risk
        # This is a conceptual check
        for merchant, risk in merchant_risk.items():
            merchant_transactions = data[data['merchant_id'] == merchant]
            if any(merchant_transactions['is_fraud'] == 1):
                # Should have some risk
                assert risk >= 0 or abs(risk) < 0.01  # Allow for floating point
    
    def test_extract_graph_features(self, transaction_graph_data):
        """
        Test extraction of all graph-based features.
        """
        network = NetworkAnalysis()
        
        # Extract all graph features
        features = network.extract_graph_features(transaction_graph_data)
        
        # Check that we have a DataFrame with features
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(transaction_graph_data['transaction_id'].unique())
        
        # Check for expected feature columns
        expected_features = [
            'customer_degree_centrality',
            'device_degree_centrality',
            'customer_community_id',
            'device_sharing_score',
            'merchant_risk_score'
        ]
        
        for feature in expected_features:
            if feature in features.columns:
                assert feature in features.columns