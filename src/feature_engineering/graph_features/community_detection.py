"""
Community Detection Features for Banking Fraud Detection
=======================================================
This module implements community detection algorithms to identify
groups of related entities that may represent fraud rings.

Key Concepts:
- Community detection: Finding groups of densely connected nodes
- Modularity: Measure of community structure quality
- Label propagation: Efficient community detection for large graphs
- Community characteristics: Size, density, fraud rate
"""

import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Set
import community as community_louvain  # python-louvain package
from sklearn.cluster import SpectralClustering


class CommunityDetectionFeatureEngineer:
    """
    Detect communities (groups) in the transaction network and create
    features based on community membership.
    
    Banking Context:
    - Fraud rings appear as communities in the network
    - Members of same community share risk
    - Community characteristics predict individual risk
    - Sudden community membership change is suspicious
    """
    
    def __init__(self):
        self.feature_columns = []
        self.graph = None
        self.community_map = {}  # node -> community_id
        self.community_stats = {}  # community_id -> statistics
        
    def detect_communities_louvain(self,
                                   graph: nx.Graph,
                                   resolution: float = 1.0) -> Dict:
        """
        Detect communities using the Louvain method.
        
        Louvain is a fast, greedy optimization method that maximizes
        modularity. It works well for large networks.
        
        Parameters:
        -----------
        graph : NetworkX graph
        resolution : Resolution parameter (higher = more communities)
        
        Returns:
        --------
        Dictionary mapping node to community ID
        """
        
        print("Detecting communities using Louvain method...")
        
        # Convert to undirected if necessary
        if graph.is_directed():
            graph = graph.to_undirected()
        
        # Run Louvain community detection
        partition = community_louvain.best_partition(
            graph, 
            resolution=resolution,
            randomize=True,
            random_state=42
        )
        
        self.community_map = partition
        
        # Get number of communities
        num_communities = len(set(partition.values()))
        print(f"  Found {num_communities} communities")
        
        # Calculate community statistics
        self._compute_community_statistics(graph)
        
        return partition
    
    def detect_communities_label_propagation(self,
                                            graph: nx.Graph,
                                            max_iterations: int = 100) -> Dict:
        """
        Detect communities using label propagation.
        
        Label propagation is very fast and works well for large graphs,
        but can be unstable (different runs give different results).
        """
        
        print("Detecting communities using label propagation...")
        
        # Run label propagation
        communities = nx.community.label_propagation_communities(graph)
        
        # Convert to node -> community_id mapping
        partition = {}
        for i, comm in enumerate(communities):
            for node in comm:
                partition[node] = i
        
        self.community_map = partition
        
        # Get number of communities
        num_communities = len(set(partition.values()))
        print(f"  Found {num_communities} communities")
        
        # Calculate community statistics
        self._compute_community_statistics(graph)
        
        return partition
    
    def detect_communities_greedy_modularity(self,
                                            graph: nx.Graph) -> Dict:
        """
        Detect communities using greedy modularity maximization.
        
        This is a slower but deterministic method.
        """
        
        print("Detecting communities using greedy modularity...")
        
        # Run greedy modularity
        communities = list(nx.community.greedy_modularity_communities(graph))
        
        # Convert to node -> community_id mapping
        partition = {}
        for i, comm in enumerate(communities):
            for node in comm:
                partition[node] = i
        
        self.community_map = partition
        
        # Get number of communities
        num_communities = len(communities)
        print(f"  Found {num_communities} communities")
        
        # Calculate community statistics
        self._compute_community_statistics(graph)
        
        return partition
    
    def _compute_community_statistics(self, graph: nx.Graph):
        """
        Compute statistics for each community.
        """
        
        # Group nodes by community
        community_nodes = defaultdict(list)
        for node, comm_id in self.community_map.items():
            community_nodes[comm_id].append(node)
        
        # Calculate statistics for each community
        for comm_id, nodes in community_nodes.items():
            # Community size
            size = len(nodes)
            
            # Subgraph of this community
            subgraph = graph.subgraph(nodes)
            
            # Internal edges
            internal_edges = subgraph.number_of_edges()
            
            # Density
            density = nx.density(subgraph) if size > 1 else 0
            
            # Average degree within community
            avg_degree = sum(dict(subgraph.degree()).values()) / size if size > 0 else 0
            
            # Count node types in community
            node_types = Counter()
            for node in nodes:
                if 'node_type' in graph.nodes[node]:
                    node_types[graph.nodes[node]['node_type']] += 1
            
            self.community_stats[comm_id] = {
                'size': size,
                'internal_edges': internal_edges,
                'density': density,
                'avg_degree': avg_degree,
                'node_types': dict(node_types)
            }
    
    def create_community_membership_features(self,
                                            df: pd.DataFrame,
                                            customer_id_col: str = 'customer_id',
                                            device_id_col: str = 'device_id',
                                            merchant_id_col: str = 'merchant_id') -> pd.DataFrame:
        """
        Create features based on community membership.
        
        Each entity gets:
        - Community ID
        - Community size
        - Community density
        - Node's role in community
        """
        
        result_df = df.copy()
        
        if not self.community_map:
            print("Warning: No communities detected. Run community detection first.")
            return result_df
        
        print("Creating community membership features...")
        
        # Map customers to communities
        def get_customer_community(customer_id):
            node = f"cust_{customer_id}"
            return self.community_map.get(node, -1)
        
        result_df['customer_community_id'] = result_df[customer_id_col].apply(get_customer_community)
        
        # Map community statistics
        result_df['customer_community_size'] = result_df['customer_community_id'].apply(
            lambda x: self.community_stats.get(x, {}).get('size', 0) if x >= 0 else 0
        )
        
        result_df['customer_community_density'] = result_df['customer_community_id'].apply(
            lambda x: self.community_stats.get(x, {}).get('density', 0) if x >= 0 else 0
        )
        
        result_df['customer_community_avg_degree'] = result_df['customer_community_id'].apply(
            lambda x: self.community_stats.get(x, {}).get('avg_degree', 0) if x >= 0 else 0
        )
        
        # Device communities
        if device_id_col in result_df.columns:
            def get_device_community(device_id):
                if pd.isna(device_id):
                    return -1
                node = f"dev_{device_id}"
                return self.community_map.get(node, -1)
            
            result_df['device_community_id'] = result_df[device_id_col].apply(get_device_community)
            
            result_df['device_community_size'] = result_df['device_community_id'].apply(
                lambda x: self.community_stats.get(x, {}).get('size', 0) if x >= 0 else 0
            )
            
            self.feature_columns.extend(['device_community_id', 'device_community_size'])
        
        # Merchant communities
        if merchant_id_col in result_df.columns:
            def get_merchant_community(merchant_id):
                if pd.isna(merchant_id):
                    return -1
                node = f"merch_{merchant_id}"
                return self.community_map.get(node, -1)
            
            result_df['merchant_community_id'] = result_df[merchant_id_col].apply(get_merchant_community)
            
            self.feature_columns.append('merchant_community_id')
        
        self.feature_columns.extend([
            'customer_community_id',
            'customer_community_size',
            'customer_community_density',
            'customer_community_avg_degree'
        ])
        
        return result_df
    
    def create_community_risk_features(self,
                                      df: pd.DataFrame,
                                      customer_id_col: str = 'customer_id',
                                      fraud_col: str = 'is_fraud') -> pd.DataFrame:
        """
        Calculate risk scores for communities based on historical fraud.
        
        If a community has high fraud rate, all members are higher risk.
        """
        
        result_df = df.copy()
        
        if 'customer_community_id' not in result_df.columns:
            result_df = self.create_community_membership_features(result_df, customer_id_col)
        
        # Calculate fraud rate per community
        community_fraud = result_df.groupby('customer_community_id').agg({
            fraud_col: ['mean', 'sum', 'count']
        })
        community_fraud.columns = ['_'.join(col).strip() for col in community_fraud.columns.values]
        community_fraud = community_fraud.reset_index()
        community_fraud.columns = ['customer_community_id', 'community_fraud_rate', 
                                   'community_fraud_count', 'community_total_members']
        
        # Merge back
        result_df = result_df.merge(
            community_fraud[['customer_community_id', 'community_fraud_rate', 
                            'community_fraud_count', 'community_total_members']],
            on='customer_community_id',
            how='left'
        )
        
        # Fill NaN for communities with no fraud data
        overall_fraud_rate = result_df[fraud_col].mean()
        result_df['community_fraud_rate'] = result_df['community_fraud_rate'].fillna(overall_fraud_rate)
        result_df['community_fraud_count'] = result_df['community_fraud_count'].fillna(0)
        result_df['community_total_members'] = result_df['community_total_members'].fillna(1)
        
        # Community risk score (smoothed)
        confidence_weight = 10
        result_df['community_risk_score'] = (
            (result_df['community_fraud_count'] + confidence_weight * overall_fraud_rate) /
            (result_df['community_total_members'] + confidence_weight)
        )
        
        # Is customer in a high-risk community?
        result_df['is_high_risk_community'] = (
            result_df['community_risk_score'] > 0.1
        ).astype(int)
        
        # Compare customer fraud rate to community average
        result_df['customer_vs_community_fraud'] = (
            result_df[fraud_col] - result_df['community_fraud_rate']
        )
        
        self.feature_columns.extend([
            'community_fraud_rate',
            'community_fraud_count',
            'community_total_members',
            'community_risk_score',
            'is_high_risk_community',
            'customer_vs_community_fraud'
        ])
        
        return result_df
    
    def create_community_role_features(self,
                                     df: pd.DataFrame,
                                     customer_id_col: str = 'customer_id',
                                     graph: Optional[nx.Graph] = None) -> pd.DataFrame:
        """
        Determine the role of each customer within their community.
        
        Roles include:
        - Hub: Highly connected within community
        - Bridge: Connects to other communities
        - Peripheral: Edge of community
        - Isolate: Alone in community
        """
        
        result_df = df.copy()
        
        if graph is None:
            graph = self.graph
        
        if graph is None:
            print("Warning: No graph available")
            return result_df
        
        if 'customer_community_id' not in result_df.columns:
            result_df = self.create_community_membership_features(result_df, customer_id_col)
        
        def get_node_role(customer_id):
            node = f"cust_{customer_id}"
            if node not in graph:
                return {'role': 'unknown', 'within_community_degree': 0, 'external_connections': 0}
            
            community_id = self.community_map.get(node, -1)
            if community_id == -1:
                return {'role': 'unknown', 'within_community_degree': 0, 'external_connections': 0}
            
            # Get all neighbors
            neighbors = list(graph.neighbors(node))
            
            # Count within-community vs external connections
            within_community = 0
            external = 0
            
            for neighbor in neighbors:
                neighbor_comm = self.community_map.get(neighbor, -1)
                if neighbor_comm == community_id:
                    within_community += 1
                else:
                    external += 1
            
            total = within_community + external
            
            # Determine role
            if total == 0:
                role = 'isolate'
            elif external > within_community:
                role = 'bridge'
            elif within_community > 5 and external < 2:
                role = 'hub'
            elif within_community > 2:
                role = 'core'
            else:
                role = 'peripheral'
            
            return {
                'role': role,
                'within_community_degree': within_community,
                'external_connections': external,
                'connection_ratio': within_community / (total + 1e-8)
            }
        
        # Apply to each customer
        roles = result_df[customer_id_col].apply(get_node_role)
        
        # Extract into columns
        result_df['customer_community_role'] = roles.apply(lambda x: x['role'])
        result_df['customer_within_community_degree'] = roles.apply(lambda x: x['within_community_degree'])
        result_df['customer_external_connections'] = roles.apply(lambda x: x['external_connections'])
        result_df['customer_connection_ratio'] = roles.apply(lambda x: x['connection_ratio'])
        
        # Encode role as numeric
        role_mapping = {
            'isolate': 0,
            'peripheral': 1,
            'core': 2,
            'hub': 3,
            'bridge': 4,
            'unknown': -1
        }
        result_df['customer_community_role_encoded'] = result_df['customer_community_role'].map(role_mapping)
        
        # Flag for bridges (connecting different communities) - high risk
        result_df['is_bridge_node'] = (result_df['customer_community_role'] == 'bridge').astype(int)
        
        # Flag for hubs (central in community) - potential ring leaders
        result_df['is_hub_node'] = (result_df['customer_community_role'] == 'hub').astype(int)
        
        self.feature_columns.extend([
            'customer_community_role_encoded',
            'customer_within_community_degree',
            'customer_external_connections',
            'customer_connection_ratio',
            'is_bridge_node',
            'is_hub_node'
        ])
        
        return result_df
    
    def create_community_change_features(self,
                                       df: pd.DataFrame,
                                       customer_id_col: str = 'customer_id',
                                       time_col: str = 'transaction_time') -> pd.DataFrame:
        """
        Detect when customers change communities over time.
        
        Changing communities can indicate:
        - Account takeover
        - Joining fraud ring
        - Behavioral shift
        """
        
        result_df = df.copy()
        result_df = result_df.sort_values([customer_id_col, time_col])
        
        if 'customer_community_id' not in result_df.columns:
            result_df = self.create_community_membership_features(result_df, customer_id_col)
        
        # Previous community for this customer
        result_df['prev_community_id'] = (
            result_df
            .groupby(customer_id_col)['customer_community_id']
            .shift(1)
        )
        
        # Did community change?
        result_df['community_changed'] = (
            (result_df['customer_community_id'] != result_df['prev_community_id']) &
            (result_df['prev_community_id'].notna())
        ).astype(int)
        
        # Days since last community change
        result_df['days_since_community_change'] = (
            result_df
            .groupby(customer_id_col)['time_col']
            .transform(lambda x: (x - x.shift(1)).dt.total_seconds() / (24 * 3600))
            .where(result_df['community_changed'] == 1)
        )
        result_df['days_since_community_change'] = (
            result_df
            .groupby(customer_id_col)['days_since_community_change']
            .fillna(method='ffill')
            .fillna(999)
        )
        
        # Number of community changes for this customer
        result_df['customer_community_changes'] = (
            result_df
            .groupby(customer_id_col)['community_changed']
            .transform('cumsum')
        )
        
        self.feature_columns.extend([
            'community_changed',
            'days_since_community_change',
            'customer_community_changes'
        ])
        
        return result_df
    
    def get_feature_names(self) -> List[str]:
        """Return list of created feature names."""
        return self.feature_columns