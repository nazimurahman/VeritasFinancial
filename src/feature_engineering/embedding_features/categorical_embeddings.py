# src/feature_engineering/embedding_features/categorical_embeddings.py
"""
Categorical Embeddings with Rotary Positional Encodings
Built from scratch using only core PyTorch classes

This module implements comprehensive categorical embeddings for various
categorical features in banking transactions including merchant categories,
device types, countries, and more.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class CategoricalEmbedding(nn.Module):
    """
    Base class for categorical embeddings with support for:
    - Standard embedding lookup
    - Hash-based embeddings for large vocabularies
    - Multi-hot encoding for multi-label categories
    - Weight sharing across related categories
    """
    
    def __init__(self,
                 num_categories: int,
                 embedding_dim: int,
                 padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None,
                 norm_type: float = 2.0,
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False,
                 use_hash: bool = False,
                 hash_bucket_size: Optional[int] = None):
        """
        Args:
            num_categories: Number of unique categories
            embedding_dim: Dimension of embeddings
            padding_idx: Index to ignore for gradient (like padding)
            max_norm: Max norm for embeddings
            norm_type: Type of norm for max_norm
            scale_grad_by_freq: Scale gradients by frequency
            sparse: Use sparse gradients
            use_hash: Use hashing trick for large vocabularies
            hash_bucket_size: Size of hash table (if use_hash=True)
        """
        super(CategoricalEmbedding, self).__init__()
        
        self.num_categories = num_categories
        self.embedding_dim = embedding_dim
        self.use_hash = use_hash
        self.hash_bucket_size = hash_bucket_size or num_categories
        
        if use_hash:
            # For hashing trick, we use a smaller embedding table
            # and map categories via hash function
            self.embedding = nn.Embedding(
                num_embeddings=self.hash_bucket_size,
                embedding_dim=embedding_dim,
                padding_idx=padding_idx,
                max_norm=max_norm,
                norm_type=norm_type,
                scale_grad_by_freq=scale_grad_by_freq,
                sparse=sparse
            )
            self.register_buffer('hash_seed', torch.randint(0, 10000, (1,)))
        else:
            # Standard embedding lookup
            self.embedding = nn.Embedding(
                num_embeddings=num_categories,
                embedding_dim=embedding_dim,
                padding_idx=padding_idx,
                max_norm=max_norm,
                norm_type=norm_type,
                scale_grad_by_freq=scale_grad_by_freq,
                sparse=sparse
            )
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize embeddings with Xavier uniform distribution"""
        nn.init.xavier_uniform_(self.embedding.weight)
    
    def _hash_category(self, category_ids: torch.Tensor) -> torch.Tensor:
        """
        Hash category IDs to bucket indices.
        
        Uses a simple hash function: (id * seed) % bucket_size
        More sophisticated: use two independent hash functions
        """
        if not self.use_hash:
            return category_ids
        
        # Simple hash: (id * seed) % bucket_size
        hashed = (category_ids * self.hash_seed) % self.hash_bucket_size
        return hashed.long()
    
    def forward(self, category_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for categorical embedding.
        
        Args:
            category_ids: Category indices [batch_size, seq_len] or [batch_size]
        
        Returns:
            Embeddings [batch_size, seq_len, embedding_dim] or [batch_size, embedding_dim]
        """
        original_shape = category_ids.shape
        
        # Apply hashing if enabled
        if self.use_hash:
            category_ids = self._hash_category(category_ids)
        
        # Get embeddings
        embedded = self.embedding(category_ids)
        
        return embedded


class MultiHotCategoricalEmbedding(nn.Module):
    """
    Embedding for multi-hot categorical features (e.g., multiple merchant tags).
    
    This handles cases where each transaction can have multiple categories
    (like a restaurant that is both "Fast Food" and "Delivery").
    """
    
    def __init__(self,
                 num_categories: int,
                 embedding_dim: int,
                 aggregation: str = 'mean',  # 'mean', 'sum', 'max', 'attention'
                 use_attention: bool = True):
        """
        Args:
            num_categories: Number of unique categories
            embedding_dim: Dimension of output embeddings
            aggregation: How to aggregate multiple embeddings
            use_attention: Whether to use attention for aggregation
        """
        super(MultiHotCategoricalEmbedding, self).__init__()
        
        self.num_categories = num_categories
        self.embedding_dim = embedding_dim
        self.aggregation = aggregation
        self.use_attention = use_attention
        
        # Base embedding for each category
        self.embedding = nn.Embedding(num_categories, embedding_dim)
        
        if use_attention and aggregation == 'attention':
            # Attention mechanism for weighted aggregation
            self.attention_projection = nn.Linear(embedding_dim, 1)
    
    def forward(self, category_ids: torch.Tensor, category_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multi-hot categories.
        
        Args:
            category_ids: Category indices [batch, seq, max_categories_per_item]
            category_mask: Mask indicating valid categories [batch, seq, max_categories]
        
        Returns:
            Aggregated embeddings [batch, seq, embedding_dim]
        """
        batch_size, seq_len, max_categories = category_ids.shape
        
        # Get embeddings for all categories
        # Shape: [batch, seq, max_categories, embedding_dim]
        all_embeddings = self.embedding(category_ids)
        
        # Apply mask
        mask_expanded = category_mask.unsqueeze(-1).float()  # [batch, seq, max_categories, 1]
        masked_embeddings = all_embeddings * mask_expanded
        
        if self.aggregation == 'mean':
            # Mean aggregation
            sum_embeddings = masked_embeddings.sum(dim=2)
            valid_counts = mask_expanded.sum(dim=2).clamp(min=1.0)
            aggregated = sum_embeddings / valid_counts
            
        elif self.aggregation == 'sum':
            # Sum aggregation
            aggregated = masked_embeddings.sum(dim=2)
            
        elif self.aggregation == 'max':
            # Max aggregation (element-wise max across categories)
            # Replace zeros with -inf for masked positions
            masked_embeddings = masked_embeddings.masked_fill(
                mask_expanded == 0, -float('inf')
            )
            aggregated, _ = torch.max(masked_embeddings, dim=2)
            # Replace -inf with 0 for completely masked items
            aggregated = torch.where(
                torch.isinf(aggregated),
                torch.zeros_like(aggregated),
                aggregated
            )
            
        elif self.aggregation == 'attention':
            # Attention-based aggregation
            # Compute attention scores
            attention_scores = self.attention_projection(all_embeddings)  # [batch, seq, max_categories, 1]
            
            # Mask attention scores
            attention_scores = attention_scores.masked_fill(
                mask_expanded == 0, -1e9
            )
            
            # Apply softmax
            attention_weights = F.softmax(attention_scores, dim=2)
            
            # Weighted sum
            aggregated = (all_embeddings * attention_weights).sum(dim=2)
        
        return aggregated


class HierarchicalCategoricalEmbedding(nn.Module):
    """
    Hierarchical categorical embeddings for features with parent-child relationships.
    
    Examples:
        - Country -> State -> City
        - Merchant Category -> Subcategory
        - Device Type -> OS -> Browser
    """
    
    def __init__(self,
                 hierarchy_levels: List[int],  # [num_parents, num_children, num_grandchildren]
                 embedding_dims: List[int],    # [parent_dim, child_dim, grandchild_dim]
                 hierarchy_relations: List[Tuple[int, int]]):  # [(parent_idx, child_idx), ...]
        """
        Args:
            hierarchy_levels: Number of categories at each level
            embedding_dims: Embedding dimension for each level
            hierarchy_relations: List of (parent_level, child_level) relations
        """
        super(HierarchicalCategoricalEmbedding, self).__init__()
        
        assert len(hierarchy_levels) == len(embedding_dims), \
            "Number of levels must match number of embedding dimensions"
        
        self.num_levels = len(hierarchy_levels)
        self.hierarchy_relations = hierarchy_relations
        
        # Create embeddings for each level
        self.embeddings = nn.ModuleList()
        for i in range(self.num_levels):
            self.embeddings.append(
                nn.Embedding(hierarchy_levels[i], embedding_dims[i])
            )
        
        # Create projection matrices for hierarchical relationships
        self.hierarchy_projections = nn.ModuleDict()
        for parent_level, child_level in hierarchy_relations:
            # Projection from parent to child space
            proj_name = f"proj_{parent_level}_{child_level}"
            self.hierarchy_projections[proj_name] = nn.Linear(
                embedding_dims[parent_level],
                embedding_dims[child_level]
            )
            
            # Inverse projection (child to parent)
            proj_name_inv = f"proj_{child_level}_{parent_level}"
            self.hierarchy_projections[proj_name_inv] = nn.Linear(
                embedding_dims[child_level],
                embedding_dims[parent_level]
            )
    
    def forward(self,
                level_ids: List[torch.Tensor],
                propagate_hierarchy: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass for hierarchical embeddings.
        
        Args:
            level_ids: List of IDs for each hierarchy level
            propagate_hierarchy: Whether to propagate information between levels
        
        Returns:
            Dictionary of embeddings for each level
        """
        assert len(level_ids) == self.num_levels, \
            f"Expected {self.num_levels} levels, got {len(level_ids)}"
        
        # Get base embeddings for each level
        base_embeddings = {}
        for i, (ids, embedding) in enumerate(zip(level_ids, self.embeddings)):
            base_embeddings[f"level_{i}"] = embedding(ids)
        
        if not propagate_hierarchy:
            return base_embeddings
        
        # Propagate information through hierarchy
        enhanced_embeddings = base_embeddings.copy()
        
        for parent_level, child_level in self.hierarchy_relations:
            # Get embeddings
            parent_emb = base_embeddings[f"level_{parent_level}"]
            child_emb = base_embeddings[f"level_{child_level}"]
            
            # Project parent to child space and add to child
            proj_name = f"proj_{parent_level}_{child_level}"
            parent_projected = self.hierarchy_projections[proj_name](parent_emb)
            enhanced_embeddings[f"level_{child_level}"] = child_emb + parent_projected
            
            # Project child to parent space and add to parent
            proj_name_inv = f"proj_{child_level}_{parent_level}"
            child_projected = self.hierarchy_projections[proj_name_inv](child_emb)
            enhanced_embeddings[f"level_{parent_level}"] = parent_emb + child_projected
        
        return enhanced_embeddings


class FrequencyAwareCategoricalEmbedding(CategoricalEmbedding):
    """
    Categorical embedding that accounts for category frequencies.
    
    This is important for handling rare categories in fraud detection,
    where rare categories might be more indicative of fraud.
    """
    
    def __init__(self,
                 num_categories: int,
                 embedding_dim: int,
                 category_frequencies: torch.Tensor,
                 rare_threshold: float = 0.01,
                 use_frequency_weighting: bool = True,
                 **kwargs):
        """
        Args:
            num_categories: Number of categories
            embedding_dim: Embedding dimension
            category_frequencies: Frequency of each category [num_categories]
            rare_threshold: Threshold for considering a category rare
            use_frequency_weighting: Whether to weight embeddings by frequency
            **kwargs: Additional arguments for parent class
        """
        super(FrequencyAwareCategoricalEmbedding, self).__init__(
            num_categories=num_categories,
            embedding_dim=embedding_dim,
            **kwargs
        )
        
        self.rare_threshold = rare_threshold
        self.use_frequency_weighting = use_frequency_weighting
        
        # Register frequencies as buffer
        self.register_buffer('category_frequencies', category_frequencies)
        
        # Create rare category mask
        rare_mask = category_frequencies < rare_threshold
        self.register_buffer('rare_mask', rare_mask)
        
        # Separate embedding for rare categories (if needed)
        if rare_mask.any():
            self.rare_embedding = nn.Embedding(
                num_embeddings=rare_mask.sum().item(),
                embedding_dim=embedding_dim
            )
            nn.init.xavier_uniform_(self.rare_embedding.weight)
            
            # Mapping from original indices to rare indices
            rare_indices = torch.where(rare_mask)[0]
            self.register_buffer('rare_index_map', rare_indices)
        
        # Frequency-based scaling factors
        if use_frequency_weighting:
            # Inverse frequency scaling: sqrt(1/freq) or similar
            # Add small epsilon to avoid division by zero
            freq_scales = 1.0 / torch.sqrt(category_frequencies + 1e-8)
            # Normalize
            freq_scales = freq_scales / freq_scales.mean()
            self.register_buffer('freq_scales', freq_scales)
    
    def forward(self, category_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with frequency awareness.
        
        Args:
            category_ids: Category indices
        
        Returns:
            Frequency-aware embeddings
        """
        original_shape = category_ids.shape
        flat_ids = category_ids.view(-1)
        
        # Get base embeddings
        if self.use_hash:
            flat_ids = self._hash_category(flat_ids)
            base_embeddings = self.embedding(flat_ids)
        else:
            base_embeddings = self.embedding(flat_ids)
        
        # Handle rare categories separately
        if hasattr(self, 'rare_embedding') and self.rare_mask.any():
            # Find rare categories
            is_rare = self.rare_mask[flat_ids]
            rare_ids = flat_ids[is_rare]
            
            if len(rare_ids) > 0:
                # Map to rare indices
                rare_indices = torch.searchsorted(self.rare_index_map, rare_ids)
                rare_embeddings = self.rare_embedding(rare_indices)
                
                # Replace base embeddings with rare embeddings for rare categories
                base_embeddings[is_rare] = rare_embeddings
        
        # Apply frequency weighting
        if self.use_frequency_weighting:
            # Get frequency scales for each category
            scales = self.freq_scales[flat_ids]
            
            # Expand scales to embedding dimension
            scales = scales.unsqueeze(-1).expand(-1, self.embedding_dim)
            
            # Apply scaling
            base_embeddings = base_embeddings * scales
        
        # Reshape back
        embeddings = base_embeddings.view(*original_shape, self.embedding_dim)
        
        return embeddings


class CrossFeatureCategoricalEmbedding(nn.Module):
    """
    Embedding for cross features (combinations of categorical variables).
    
    This is crucial for fraud detection where combinations like
    (merchant_category, country) might be more predictive than individually.
    """
    
    def __init__(self,
                 feature_configs: List[Dict],
                 embedding_dim: int,
                 interaction_method: str = 'concat',  # 'concat', 'sum', 'product', 'attention'
                 use_self_attention: bool = True):
        """
        Args:
            feature_configs: List of configs for each feature
                Each config: {'name': str, 'num_categories': int, 'embedding_dim': int}
            embedding_dim: Output embedding dimension
            interaction_method: How to combine feature embeddings
            use_self_attention: Whether to use self-attention for interactions
        """
        super(CrossFeatureCategoricalEmbedding, self).__init__()
        
        self.feature_configs = feature_configs
        self.num_features = len(feature_configs)
        self.interaction_method = interaction_method
        self.use_self_attention = use_self_attention
        
        # Create embeddings for each feature
        self.feature_embeddings = nn.ModuleList()
        self.feature_names = []
        
        for config in feature_configs:
            self.feature_embeddings.append(
                nn.Embedding(
                    num_embeddings=config['num_categories'],
                    embedding_dim=config['embedding_dim']
                )
            )
            self.feature_names.append(config['name'])
        
        # Compute total embedding dimension after combining
        if interaction_method == 'concat':
            total_dim = sum([cfg['embedding_dim'] for cfg in feature_configs])
        else:
            # For sum/product, all embeddings must have same dimension
            assert all(cfg['embedding_dim'] == feature_configs[0]['embedding_dim'] 
                      for cfg in feature_configs), \
                "All embeddings must have same dimension for sum/product interaction"
            total_dim = feature_configs[0]['embedding_dim']
        
        # Projection to output dimension
        self.output_projection = nn.Linear(total_dim, embedding_dim)
        
        # Self-attention for feature interactions
        if use_self_attention:
            self.self_attention = nn.MultiheadAttention(
                embed_dim=total_dim,
                num_heads=4,
                batch_first=True
            )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for all embeddings"""
        for embedding in self.feature_embeddings:
            nn.init.xavier_uniform_(embedding.weight)
        
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
    
    def forward(self, feature_ids: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for cross-feature embedding.
        
        Args:
            feature_ids: List of ID tensors for each feature
                        Each shape: [batch_size, seq_len] or [batch_size]
        
        Returns:
            Combined cross-feature embeddings
        """
        assert len(feature_ids) == self.num_features, \
            f"Expected {self.num_features} features, got {len(feature_ids)}"
        
        batch_size = feature_ids[0].shape[0]
        has_seq_dim = len(feature_ids[0].shape) == 2
        
        # Get embeddings for each feature
        feature_embeddings = []
        for i, (ids, embedding) in enumerate(zip(feature_ids, self.feature_embeddings)):
            emb = embedding(ids)  # [batch, seq, feat_dim] or [batch, feat_dim]
            feature_embeddings.append(emb)
        
        # Combine based on interaction method
        if self.interaction_method == 'concat':
            # Concatenate along feature dimension
            combined = torch.cat(feature_embeddings, dim=-1)
            
        elif self.interaction_method == 'sum':
            # Sum all embeddings
            combined = torch.stack(feature_embeddings, dim=0).sum(dim=0)
            
        elif self.interaction_method == 'product':
            # Element-wise product (all embeddings must have same shape)
            combined = torch.stack(feature_embeddings, dim=0).prod(dim=0)
            
        elif self.interaction_method == 'attention':
            # Stack embeddings and apply self-attention
            stacked = torch.stack(feature_embeddings, dim=1)  # [batch, num_features, feat_dim]
            
            if has_seq_dim:
                # Handle sequence dimension
                batch_size, seq_len, num_features, feat_dim = stacked.shape
                stacked = stacked.view(batch_size * seq_len, num_features, feat_dim)
            
            # Apply self-attention
            attended, _ = self.self_attention(stacked, stacked, stacked)
            
            if has_seq_dim:
                attended = attended.view(batch_size, seq_len, num_features, feat_dim)
            
            # Aggregate across features (mean)
            combined = attended.mean(dim=1)
        
        # Project to output dimension
        if has_seq_dim:
            batch_size, seq_len, feat_dim = combined.shape
            combined_flat = combined.view(-1, feat_dim)
            projected_flat = self.output_projection(combined_flat)
            output = projected_flat.view(batch_size, seq_len, -1)
        else:
            output = self.output_projection(combined)
        
        return output


class TimeAwareCategoricalEmbedding(nn.Module):
    """
    Categorical embedding that incorporates temporal dynamics.
    
    Categories can change their meaning over time (e.g., merchant risk changes),
    so embeddings should be time-aware.
    """
    
    def __init__(self,
                 num_categories: int,
                 embedding_dim: int,
                 time_embedding_dim: int = 32,
                 use_temporal_gate: bool = True,
                 **kwargs):
        """
        Args:
            num_categories: Number of categories
            embedding_dim: Base embedding dimension
            time_embedding_dim: Dimension for time encoding
            use_temporal_gate: Whether to use gating mechanism
            **kwargs: Additional arguments for base embedding
        """
        super(TimeAwareCategoricalEmbedding, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.time_embedding_dim = time_embedding_dim
        self.use_temporal_gate = use_temporal_gate
        
        # Base categorical embedding
        self.base_embedding = CategoricalEmbedding(
            num_categories=num_categories,
            embedding_dim=embedding_dim,
            **kwargs
        )
        
        # Time encoding (using sinusoidal encodings)
        self.time_encoder = TimeDeltaEncoder(
            embedding_dim=time_embedding_dim,
            max_delta=86400 * 30  # 30 days
        )
        
        if use_temporal_gate:
            # Gate to control temporal influence
            self.temporal_gate = nn.Sequential(
                nn.Linear(embedding_dim + time_embedding_dim, embedding_dim),
                nn.Sigmoid()
            )
        
        # Temporal transformation
        self.temporal_transform = nn.Linear(time_embedding_dim, embedding_dim)
    
    def forward(self,
                category_ids: torch.Tensor,
                timestamps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with temporal awareness.
        
        Args:
            category_ids: Category indices [batch, seq]
            timestamps: Transaction timestamps [batch, seq]
        
        Returns:
            Time-aware categorical embeddings
        """
        # Get base categorical embeddings
        base_emb = self.base_embedding(category_ids)  # [batch, seq, emb_dim]
        
        # Encode time
        time_encoded = self.time_encoder(timestamps)  # [batch, seq, time_dim]
        
        # Transform time to embedding space
        time_transformed = self.temporal_transform(time_encoded)  # [batch, seq, emb_dim]
        
        if self.use_temporal_gate:
            # Compute gate value
            gate_input = torch.cat([base_emb, time_encoded], dim=-1)
            gate = self.temporal_gate(gate_input)  # [batch, seq, emb_dim]
            
            # Apply gate: base_emb + gate * time_transformed
            output = base_emb + gate * time_transformed
        else:
            # Simple addition
            output = base_emb + time_transformed
        
        return output


class GraphAwareCategoricalEmbedding(nn.Module):
    """
    Categorical embedding that incorporates graph structure.
    
    In fraud detection, categories often have graph relationships
    (e.g., merchants sharing devices, customers sharing IPs).
    """
    
    def __init__(self,
                 num_categories: int,
                 embedding_dim: int,
                 adjacency_matrix: Optional[torch.Tensor] = None,
                 use_graph_conv: bool = True,
                 num_graph_layers: int = 2,
                 **kwargs):
        """
        Args:
            num_categories: Number of categories
            embedding_dim: Embedding dimension
            adjacency_matrix: Adjacency matrix of the category graph
            use_graph_conv: Whether to use graph convolution
            num_graph_layers: Number of graph convolution layers
            **kwargs: Additional arguments for base embedding
        """
        super(GraphAwareCategoricalEmbedding, self).__init__()
        
        self.num_categories = num_categories
        self.embedding_dim = embedding_dim
        self.use_graph_conv = use_graph_conv
        
        # Base categorical embedding
        self.base_embedding = CategoricalEmbedding(
            num_categories=num_categories,
            embedding_dim=embedding_dim,
            **kwargs
        )
        
        if use_graph_conv:
            # Store adjacency matrix
            if adjacency_matrix is not None:
                self.register_buffer('adjacency', adjacency_matrix)
                
                # Normalize adjacency matrix
                degree = adjacency_matrix.sum(dim=-1, keepdim=True)
                degree = torch.where(degree > 0, degree, torch.ones_like(degree))
                self.register_buffer('norm_adj', adjacency_matrix / degree)
            
            # Graph convolution layers
            self.graph_convs = nn.ModuleList()
            for i in range(num_graph_layers):
                self.graph_convs.append(
                    nn.Linear(embedding_dim, embedding_dim)
                )
            
            # Combination layer
            self.combine_layer = nn.Linear(embedding_dim * (num_graph_layers + 1), embedding_dim)
    
    def forward(self,
                category_ids: torch.Tensor,
                return_graph_embeddings: bool = False) -> Union[torch.Tensor, Dict]:
        """
        Forward pass with graph awareness.
        
        Args:
            category_ids: Category indices
            return_graph_embeddings: Return intermediate graph embeddings
        
        Returns:
            Graph-aware embeddings
        """
        # Get base embeddings
        base_emb = self.base_embedding(category_ids)
        
        if not self.use_graph_conv or not hasattr(self, 'norm_adj'):
            return base_emb
        
        # Apply graph convolution
        all_embeddings = [base_emb]
        current_emb = base_emb
        
        batch_size, seq_len, emb_dim = base_emb.shape
        
        for conv_layer in self.graph_convs:
            # For each item in batch, propagate through graph
            # This is a simplified version - in practice, you'd use sparse operations
            
            # Reshape to [batch * seq, emb_dim]
            current_flat = current_emb.view(-1, emb_dim)
            
            # Apply graph convolution: A * X * W
            # A is [num_categories, num_categories], X is [batch*seq, num_categories, emb_dim]?
            # This is simplified - actual implementation depends on graph structure
            
            # For demonstration, we'll do a simple propagation
            # In reality, you'd need to map category_ids to graph nodes
            
            # Placeholder for graph convolution
            # In practice, implement proper graph convolution using the adjacency matrix
            convolved = conv_layer(current_flat)
            convolved = convolved.view(batch_size, seq_len, emb_dim)
            
            all_embeddings.append(convolved)
            current_emb = convolved
        
        # Combine all graph convolution outputs
        combined = torch.cat(all_embeddings, dim=-1)
        combined_flat = combined.view(-1, combined.shape[-1])
        output_flat = self.combine_layer(combined_flat)
        output = output_flat.view(batch_size, seq_len, emb_dim)
        
        if return_graph_embeddings:
            return {
                'base': base_emb,
                'graph_embeddings': all_embeddings[1:],
                'final': output
            }
        
        return output


# Utility function to create a complete categorical embedding pipeline
def create_categorical_embedding_pipeline(feature_configs: Dict,
                                         embedding_dim: int = 64,
                                         use_time_aware: bool = True,
                                         use_graph_aware: bool = False) -> nn.ModuleDict:
    """
    Factory function to create a complete categorical embedding pipeline.
    
    Args:
        feature_configs: Configuration for each categorical feature
        embedding_dim: Base embedding dimension
        use_time_aware: Whether to use time-aware embeddings
        use_graph_aware: Whether to use graph-aware embeddings
    
    Returns:
        ModuleDict of configured embeddings
    """
    
    embeddings = nn.ModuleDict()
    
    for feature_name, config in feature_configs.items():
        feature_type = config.get('type', 'standard')
        num_categories = config['num_categories']
        
        if feature_type == 'standard':
            embeddings[feature_name] = CategoricalEmbedding(
                num_categories=num_categories,
                embedding_dim=config.get('embedding_dim', embedding_dim),
                use_hash=config.get('use_hash', num_categories > 10000),
                hash_bucket_size=config.get('hash_bucket_size', min(num_categories, 10000))
            )
            
        elif feature_type == 'multi_hot':
            embeddings[feature_name] = MultiHotCategoricalEmbedding(
                num_categories=num_categories,
                embedding_dim=config.get('embedding_dim', embedding_dim),
                aggregation=config.get('aggregation', 'attention')
            )
            
        elif feature_type == 'hierarchical':
            embeddings[feature_name] = HierarchicalCategoricalEmbedding(
                hierarchy_levels=config['hierarchy_levels'],
                embedding_dims=config.get('embedding_dims', [embedding_dim] * len(config['hierarchy_levels'])),
                hierarchy_relations=config['hierarchy_relations']
            )
            
        elif feature_type == 'frequency_aware':
            # Need to pass frequencies
            frequencies = torch.tensor(config['frequencies'])
            embeddings[feature_name] = FrequencyAwareCategoricalEmbedding(
                num_categories=num_categories,
                embedding_dim=config.get('embedding_dim', embedding_dim),
                category_frequencies=frequencies,
                rare_threshold=config.get('rare_threshold', 0.01)
            )
        
        # Wrap with time awareness if requested
        if use_time_aware and config.get('time_aware', False):
            embeddings[feature_name] = TimeAwareCategoricalEmbedding(
                num_categories=num_categories,
                embedding_dim=config.get('embedding_dim', embedding_dim),
                use_temporal_gate=True
            )
        
        # Wrap with graph awareness if requested
        if use_graph_aware and config.get('graph_aware', False):
            # Need to pass adjacency matrix
            if 'adjacency' in config:
                adjacency = torch.tensor(config['adjacency'])
                embeddings[feature_name] = GraphAwareCategoricalEmbedding(
                    num_categories=num_categories,
                    embedding_dim=config.get('embedding_dim', embedding_dim),
                    adjacency_matrix=adjacency
                )
    
    return embeddings


# Example configuration for banking categorical features
BANKING_CATEGORICAL_CONFIG = {
    'merchant_category': {
        'type': 'hierarchical',
        'hierarchy_levels': [50, 200],  # 50 main categories, 200 subcategories
        'embedding_dims': [32, 16],
        'hierarchy_relations': [(0, 1)],  # main category -> subcategory
        'time_aware': True,
        'graph_aware': True
    },
    'device_type': {
        'type': 'standard',
        'num_categories': 100,
        'embedding_dim': 16,
        'use_hash': False,
        'time_aware': True
    },
    'country': {
        'type': 'frequency_aware',
        'num_categories': 250,
        'embedding_dim': 16,
        'frequencies': [],  # Would be populated from data
        'rare_threshold': 0.01,
        'time_aware': False
    },
    'transaction_tags': {
        'type': 'multi_hot',
        'num_categories': 500,
        'embedding_dim': 32,
        'aggregation': 'attention',
        'time_aware': True
    },
    'risk_level': {
        'type': 'standard',
        'num_categories': 5,
        'embedding_dim': 8,
        'use_hash': False
    }
}


# Re-export TimeDeltaEncoder for convenience
from .transaction_embeddings import TimeDeltaEncoder

__all__ = [
    'CategoricalEmbedding',
    'MultiHotCategoricalEmbedding',
    'HierarchicalCategoricalEmbedding',
    'FrequencyAwareCategoricalEmbedding',
    'CrossFeatureCategoricalEmbedding',
    'TimeAwareCategoricalEmbedding',
    'GraphAwareCategoricalEmbedding',
    'create_categorical_embedding_pipeline',
    'BANKING_CATEGORICAL_CONFIG',
    'TimeDeltaEncoder'
]