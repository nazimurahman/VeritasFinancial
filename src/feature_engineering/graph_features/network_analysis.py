"""
Network Analysis Features for Banking Fraud Detection
====================================================
This module implements graph-based features that capture relationships
between entities (customers, devices, merchants, IPs). Network analysis
is crucial for detecting fraud rings and organized crime.

Key Concepts:
- Graph construction: Nodes (entities) and edges (transactions)
- Centrality metrics: Importance of nodes in the network
- Connectivity patterns: How entities are connected
- Community detection: Identifying groups of related entities
- Suspicious patterns: Rings, stars, chains
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class NetworkAnalysisFeatureEngineer:
    """
    Create graph-based network features for fraud detection.
    
    Banking Context:
    - Fraudsters often operate in rings (connected groups)
    - Shared devices/phones indicate fraud rings
    - Money mule networks have characteristic structures
    - Sudden connection to suspicious nodes is risky
    """
    
    def __init__(self):
        self.feature_columns = []
        self.graph = None
        self.node_features = {}
        
    def build_transaction_network(self,
                                  df: pd.DataFrame,
                                  customer_id_col: str = 'customer_id',
                                  device_id_col: str = 'device_id',
                                  merchant_id_col: str = 'merchant_id',
                                  ip_col: str = 'ip_address',
                                  transaction_id_col: str = 'transaction_id',
                                  amount_col: str = 'amount',
                                  time_col: str = 'transaction_time') -> nx.Graph:
        """
        Build a heterogeneous graph from transaction data.
        
        Node types:
        - Customers
        - Devices
        - Merchants
        - IP addresses
        
        Edge types:
        - Customer-Device (uses)
        - Customer-Merchant (transacts with)
        - Device-IP (connects from)
        - Customer-IP (transacts from)
        - Merchant-Merchant (co-occurrence in transactions)
        """
        
        print("Building transaction network graph...")
        
        # Initialize graph
        G = nx.Graph()
        
        # Add nodes with type attributes
        unique_customers = df[customer_id_col].unique()
        unique_devices = df[device_id_col].unique() if device_id_col in df.columns else []
        unique_merchants = df[merchant_id_col].unique() if merchant_id_col in df.columns else []
        unique_ips = df[ip_col].unique() if ip_col in df.columns else []
        
        # Add customer nodes
        for cust in unique_customers:
            G.add_node(f"cust_{cust}", node_type='customer', id=cust)
        
        # Add device nodes
        for dev in unique_devices:
            G.add_node(f"dev_{dev}", node_type='device', id=dev)
        
        # Add merchant nodes
        for merch in unique_merchants:
            G.add_node(f"merch_{merch}", node_type='merchant', id=merch)
        
        # Add IP nodes
        for ip in unique_ips:
            G.add_node(f"ip_{ip}", node_type='ip', id=ip)
        
        print(f"  Added {G.number_of_nodes()} nodes")
        print(f"    Customers: {len(unique_customers)}")
        print(f"    Devices: {len(unique_devices)}")
        print(f"    Merchants: {len(unique_merchants)}")
        print(f"    IPs: {len(unique_ips)}")
        
        # Add edges based on transactions
        edge_count = 0
        
        # Group by transaction to add edges
        for _, transaction in df.iterrows():
            cust = f"cust_{transaction[customer_id_col]}"
            
            # Customer-Device edge
            if device_id_col in df.columns and pd.notna(transaction[device_id_col]):
                dev = f"dev_{transaction[device_id_col]}"
                if G.has_edge(cust, dev):
                    G[cust][dev]['weight'] += 1
                    G[cust][dev]['total_amount'] += transaction[amount_col]
                else:
                    G.add_edge(cust, dev, weight=1, total_amount=transaction[amount_col], 
                              edge_type='uses')
                    edge_count += 1
            
            # Customer-Merchant edge
            if merchant_id_col in df.columns and pd.notna(transaction[merchant_id_col]):
                merch = f"merch_{transaction[merchant_id_col]}"
                if G.has_edge(cust, merch):
                    G[cust][merch]['weight'] += 1
                    G[cust][merch]['total_amount'] += transaction[amount_col]
                else:
                    G.add_edge(cust, merch, weight=1, total_amount=transaction[amount_col],
                              edge_type='transacts_with')
                    edge_count += 1
            
            # Customer-IP edge
            if ip_col in df.columns and pd.notna(transaction[ip_col]):
                ip = f"ip_{transaction[ip_col]}"
                if G.has_edge(cust, ip):
                    G[cust][ip]['weight'] += 1
                else:
                    G.add_edge(cust, ip, weight=1, edge_type='connects_from')
                    edge_count += 1
            
            # Device-IP edge
            if (device_id_col in df.columns and ip_col in df.columns and 
                pd.notna(transaction[device_id_col]) and pd.notna(transaction[ip_col])):
                dev = f"dev_{transaction[device_id_col]}"
                ip = f"ip_{transaction[ip_col]}"
                if G.has_edge(dev, ip):
                    G[dev][ip]['weight'] += 1
                else:
                    G.add_edge(dev, ip, weight=1, edge_type='uses_ip')
                    edge_count += 1
        
        print(f"  Added {edge_count} edges")
        
        self.graph = G
        return G
    
    def create_centrality_features(self,
                                  df: pd.DataFrame,
                                  customer_id_col: str = 'customer_id',
                                  device_id_col: str = 'device_id',
                                  merchant_id_col: str = 'merchant_id') -> pd.DataFrame:
        """
        Create centrality metrics for nodes in the network.
        
        Centrality measures indicate how "important" or "connected"
        an entity is in the network. High centrality can indicate:
        - Hub in fraud ring
        - Money mule coordinator
        - Shared device/resource
        """
        
        result_df = df.copy()
        
        if self.graph is None:
            print("Warning: Graph not built. Call build_transaction_network first.")
            return result_df
        
        print("Calculating network centrality features...")
        
        # Calculate centrality measures for all nodes
        # Degree centrality (number of connections)
        degree_centrality = nx.degree_centrality(self.graph)
        
        # Betweenness centrality (nodes that connect different groups)
        # This is computationally expensive for large graphs
        # Use approximation for large graphs
        if self.graph.number_of_nodes() > 10000:
            print("  Using approximate betweenness centrality (large graph)")
            betweenness_centrality = nx.betweenness_centrality(self.graph, k=1000)
        else:
            betweenness_centrality = nx.betweenness_centrality(self.graph)
        
        # Closeness centrality (how close to all other nodes)
        closeness_centrality = nx.closeness_centrality(self.graph)
        
        # Eigenvector centrality (connected to important nodes)
        try:
            eigenvector_centrality = nx.eigenvector_centrality_numpy(self.graph)
        except:
            # Fallback to power iteration
            eigenvector_centrality = nx.eigenvector_centrality(self.graph, max_iter=1000)
        
        # PageRank (Google's algorithm - good for importance)
        pagerank = nx.pagerank(self.graph)
        
        # Map centralities to entities in dataframe
        # For customers
        result_df['customer_degree_centrality'] = result_df[customer_id_col].apply(
            lambda x: degree_centrality.get(f"cust_{x}", 0)
        )
        
        result_df['customer_betweenness_centrality'] = result_df[customer_id_col].apply(
            lambda x: betweenness_centrality.get(f"cust_{x}", 0)
        )
        
        result_df['customer_closeness_centrality'] = result_df[customer_id_col].apply(
            lambda x: closeness_centrality.get(f"cust_{x}", 0)
        )
        
        result_df['customer_eigenvector_centrality'] = result_df[customer_id_col].apply(
            lambda x: eigenvector_centrality.get(f"cust_{x}", 0)
        )
        
        result_df['customer_pagerank'] = result_df[customer_id_col].apply(
            lambda x: pagerank.get(f"cust_{x}", 0)
        )
        
        # For devices
        if device_id_col in result_df.columns:
            result_df['device_degree_centrality'] = result_df[device_id_col].apply(
                lambda x: degree_centrality.get(f"dev_{x}", 0) if pd.notna(x) else 0
            )
            
            result_df['device_pagerank'] = result_df[device_id_col].apply(
                lambda x: pagerank.get(f"dev_{x}", 0) if pd.notna(x) else 0
            )
        
        # For merchants
        if merchant_id_col in result_df.columns:
            result_df['merchant_degree_centrality'] = result_df[merchant_id_col].apply(
                lambda x: degree_centrality.get(f"merch_{x}", 0) if pd.notna(x) else 0
            )
            
            result_df['merchant_pagerank'] = result_df[merchant_id_col].apply(
                lambda x: pagerank.get(f"merch_{x}", 0) if pd.notna(x) else 0
            )
        
        self.feature_columns.extend([
            'customer_degree_centrality',
            'customer_betweenness_centrality',
            'customer_closeness_centrality',
            'customer_eigenvector_centrality',
            'customer_pagerank'
        ])
        
        if device_id_col in result_df.columns:
            self.feature_columns.extend(['device_degree_centrality', 'device_pagerank'])
        
        if merchant_id_col in result_df.columns:
            self.feature_columns.extend(['merchant_degree_centrality', 'merchant_pagerank'])
        
        return result_df
    
    def create_connectivity_features(self,
                                    df: pd.DataFrame,
                                    customer_id_col: str = 'customer_id',
                                    device_id_col: str = 'device_id',
                                    ip_col: str = 'ip_address') -> pd.DataFrame:
        """
        Create features based on connectivity patterns.
        
        These capture how entities are connected to each other:
        - Number of connections
        - Shared connections
        - Distance to known fraudsters
        """
        
        result_df = df.copy()
        
        if self.graph is None:
            return result_df
        
        # For each customer, find all directly connected entities
        def get_connected_entities(customer_id):
            node = f"cust_{customer_id}"
            if node not in self.graph:
                return {'devices': 0, 'ips': 0, 'merchants': 0}
            
            neighbors = list(self.graph.neighbors(node))
            
            # Count by type
            devices = sum(1 for n in neighbors if n.startswith('dev_'))
            ips = sum(1 for n in neighbors if n.startswith('ip_'))
            merchants = sum(1 for n in neighbors if n.startswith('merch_'))
            
            return {
                'connected_devices': devices,
                'connected_ips': ips,
                'connected_merchants': merchants,
                'total_connections': len(neighbors)
            }
        
        # Apply to each customer
        connectivity = result_df[customer_id_col].apply(get_connected_entities)
        
        # Extract into columns
        result_df['customer_connected_devices'] = connectivity.apply(lambda x: x['connected_devices'])
        result_df['customer_connected_ips'] = connectivity.apply(lambda x: x['connected_ips'])
        result_df['customer_connected_merchants'] = connectivity.apply(lambda x: x['connected_merchants'])
        result_df['customer_total_connections'] = connectivity.apply(lambda x: x['total_connections'])
        
        # Connection density (connections relative to network size)
        result_df['customer_connection_density'] = (
            result_df['customer_total_connections'] / (self.graph.number_of_nodes() + 1e-8)
        )
        
        # For devices, get connected customers
        if device_id_col in result_df.columns:
            def get_device_connections(device_id):
                if pd.isna(device_id):
                    return 0
                node = f"dev_{device_id}"
                if node not in self.graph:
                    return 0
                return len(list(self.graph.neighbors(node)))
            
            result_df['device_connected_customers'] = result_df[device_id_col].apply(get_device_connections)
            self.feature_columns.append('device_connected_customers')
        
        self.feature_columns.extend([
            'customer_connected_devices',
            'customer_connected_ips',
            'customer_connected_merchants',
            'customer_total_connections',
            'customer_connection_density'
        ])
        
        return result_df
    
    def create_distance_to_fraud_features(self,
                                        df: pd.DataFrame,
                                        customer_id_col: str = 'customer_id',
                                        fraud_col: str = 'is_fraud') -> pd.DataFrame:
        """
        Calculate shortest path distance to known fraudulent entities.
        
        If a customer is close (in network terms) to known fraudsters,
        they may be part of the same fraud ring.
        """
        
        result_df = df.copy()
        
        if self.graph is None:
            return result_df
        
        # Identify fraudulent customers (from training data)
        fraud_customers = set(
            result_df[result_df[fraud_col] == 1][customer_id_col].unique()
        )
        
        fraud_nodes = [f"cust_{c}" for c in fraud_customers if f"cust_{c}" in self.graph]
        
        if not fraud_nodes:
            print("No fraudulent nodes found in graph")
            return result_df
        
        print(f"Computing distances to {len(fraud_nodes)} fraudulent nodes...")
        
        # For efficiency, compute distances only for nodes in our dataframe
        unique_customers = result_df[customer_id_col].unique()
        customer_nodes = [f"cust_{c}" for c in unique_customers if f"cust_{c}" in self.graph]
        
        # Compute shortest path distances to nearest fraud node
        distances = {}
        
        for cust_node in customer_nodes:
            min_distance = float('inf')
            
            # Check if customer itself is fraudulent
            if cust_node in fraud_nodes:
                distances[cust_node] = 0
                continue
            
            # Find shortest path to any fraud node
            for fraud_node in fraud_nodes[:10]:  # Limit to 10 fraud nodes for efficiency
                try:
                    path_length = nx.shortest_path_length(self.graph, cust_node, fraud_node)
                    min_distance = min(min_distance, path_length)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
            
            if min_distance == float('inf'):
                distances[cust_node] = -1  # No path to any fraud node
            else:
                distances[cust_node] = min_distance
        
        # Map back to dataframe
        result_df['distance_to_fraud'] = result_df[customer_id_col].apply(
            lambda x: distances.get(f"cust_{x}", -1)
        )
        
        # Flag for close to fraud (within 2 hops)
        result_df['is_close_to_fraud'] = (
            (result_df['distance_to_fraud'] > 0) & 
            (result_df['distance_to_fraud'] <= 2)
        ).astype(int)
        
        self.feature_columns.extend(['distance_to_fraud', 'is_close_to_fraud'])
        
        return result_df
    
    def create_ego_network_features(self,
                                   df: pd.DataFrame,
                                   customer_id_col: str = 'customer_id') -> pd.DataFrame:
        """
        Create features based on ego networks (the subgraph of a node
        and its immediate neighbors).
        
        Ego network features capture local structure that might indicate
        fraud ring participation.
        """
        
        result_df = df.copy()
        
        if self.graph is None:
            return result_df
        
        def compute_ego_features(customer_id):
            node = f"cust_{customer_id}"
            if node not in self.graph:
                return {
                    'ego_size': 0,
                    'ego_edges': 0,
                    'ego_density': 0,
                    'ego_clustering': 0
                }
            
            # Get ego network (node + its neighbors)
            ego_net = nx.ego_graph(self.graph, node, radius=1)
            
            # Ego network size (number of nodes)
            ego_size = ego_net.number_of_nodes()
            
            # Number of edges in ego network
            ego_edges = ego_net.number_of_edges()
            
            # Density of ego network (excluding ego)
            if ego_size > 1:
                # Remove ego node for density calculation
                ego_net.remove_node(node)
                ego_density = nx.density(ego_net)
            else:
                ego_density = 0
            
            # Clustering coefficient (how connected neighbors are)
            # High clustering = tightly-knit group (potential ring)
            try:
                ego_clustering = nx.clustering(self.graph, node)
            except:
                ego_clustering = 0
            
            return {
                'ego_size': ego_size,
                'ego_edges': ego_edges,
                'ego_density': ego_density,
                'ego_clustering': ego_clustering
            }
        
        # Apply to each customer
        ego_features = result_df[customer_id_col].apply(compute_ego_features)
        
        # Extract into columns
        result_df['customer_ego_size'] = ego_features.apply(lambda x: x['ego_size'])
        result_df['customer_ego_edges'] = ego_features.apply(lambda x: x['ego_edges'])
        result_df['customer_ego_density'] = ego_features.apply(lambda x: x['ego_density'])
        result_df['customer_ego_clustering'] = ego_features.apply(lambda x: x['ego_clustering'])
        
        # Flag for dense ego networks (potential fraud rings)
        result_df['is_dense_ego_network'] = (
            result_df['customer_ego_density'] > 0.5
        ).astype(int)
        
        self.feature_columns.extend([
            'customer_ego_size',
            'customer_ego_edges',
            'customer_ego_density',
            'customer_ego_clustering',
            'is_dense_ego_network'
        ])
        
        return result_df
    
    def get_feature_names(self) -> List[str]:
        """Return list of created feature names."""
        return self.feature_columns