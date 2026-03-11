# src/feature_engineering/embedding_features/transaction_embeddings.py
"""
Transaction Embeddings with Rotary Positional Encodings (RoPE)
Built from scratch using only core PyTorch classes

This module implements comprehensive transaction embeddings that convert
raw transaction data into dense vector representations with rotary positional
encodings for capturing sequential patterns in banking transactions.
"""

import torch
import torch.nn as nn
import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) implementation from scratch.
    
    RoPE rotates query and key vectors based on their position in the sequence,
    allowing the model to capture relative position information without adding
    position vectors to the embeddings.
    
    Mathematical Formulation:
        For a position m and dimension i, the rotation angle is:
        θ_i = 10000^(-2i/d) where d is the embedding dimension
        
        The rotation matrix R(θ) rotates (x₁, x₂) pairs:
        [x₁ * cos(θ) - x₂ * sin(θ), x₁ * sin(θ) + x₂ * cos(θ)]
    
    Why RoPE for Banking Transactions:
        - Preserves relative position information crucial for transaction sequences
        - Better generalization to longer sequences than learned positional embeddings
        - Natural decay of attention with distance (useful for recent transaction focus)
    """
    
    def __init__(self, embedding_dim: int, max_seq_length: int = 512, base: int = 10000):
        """
        Args:
            embedding_dim: Dimension of embeddings (must be even)
            max_seq_length: Maximum sequence length to pre-compute
            base: Base for the geometric series (standard is 10000)
        """
        super(RotaryPositionalEmbedding, self).__init__()
        
        # Validate embedding dimension (must be even for pairwise rotation)
        assert embedding_dim % 2 == 0, f"Embedding dimension must be even, got {embedding_dim}"
        
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.base = base
        
        # Pre-compute all rotation angles for all positions and dimensions
        # This is done once and cached for efficiency
        self._compute_rotation_matrix()
        
    def _compute_rotation_matrix(self):
        """
        Pre-compute the rotation matrix for all positions up to max_seq_length.
        
        The rotation matrix is stored as cos and sin values for each position
        and each dimension pair.
        
        Structure:
            For position m and dimension i (0 to d/2 - 1):
                angle = m * θ_i where θ_i = base^(-2i/d)
                cos_cache[m, i] = cos(angle)
                sin_cache[m, i] = sin(angle)
        """
        # Create position indices [0, 1, 2, ..., max_seq_length-1]
        positions = torch.arange(self.max_seq_length, dtype=torch.float32)
        
        # Create dimension indices for each pair [0, 2, 4, ..., d-2]
        # We only need half the dimensions because we rotate pairs
        dim_pairs = torch.arange(0, self.embedding_dim, 2, dtype=torch.float32)
        
        # Calculate θ_i for each dimension pair: base^(-2i/d)
        # This creates a geometric series decreasing with dimension
        theta = 1.0 / (self.base ** (dim_pairs / self.embedding_dim))
        
        # Calculate angles for all positions: positions * θ_i (broadcasting)
        # Shape: [max_seq_length, embedding_dim/2]
        angles = positions.unsqueeze(-1) * theta.unsqueeze(0)
        
        # Compute cos and sin for all angles
        # Shape: [max_seq_length, embedding_dim/2]
        self.cos_cache = torch.cos(angles)  # cos(m * θ_i)
        self.sin_cache = torch.sin(angles)  # sin(m * θ_i)
        
        # Register as buffers (not trainable parameters)
        self.register_buffer('cos_cached', self.cos_cache)
        self.register_buffer('sin_cached', self.sin_cache)
    
    def _rotate_pair(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Apply rotation to a pair of dimensions (x₁, x₂).
        
        Args:
            x: Input tensor of shape [..., 2] (last dimension is the pair)
            cos: Cosine of rotation angle for this position
            sin: Sine of rotation angle for this position
            
        Returns:
            Rotated tensor of same shape
        """
        # Split the pair into two components
        x1, x2 = x[..., 0], x[..., 1]
        
        # Apply rotation: 
        # x₁' = x₁ * cos - x₂ * sin
        # x₂' = x₁ * sin + x₂ * cos
        x1_rotated = x1 * cos - x2 * sin
        x2_rotated = x1 * sin + x2 * cos
        
        # Stack back together
        return torch.stack([x1_rotated, x2_rotated], dim=-1)
    
    def rotate_queries_and_keys(self, 
                                queries: torch.Tensor, 
                                keys: torch.Tensor,
                                positions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary positional encoding to query and key tensors.
        
        This is the core RoPE operation: rotate queries and keys based on their
        positions before computing attention scores.
        
        Args:
            queries: Query tensor of shape [batch_size, seq_len, embedding_dim]
            keys: Key tensor of shape [batch_size, seq_len, embedding_dim]
            positions: Optional position indices. If None, uses [0, 1, ..., seq_len-1]
            
        Returns:
            Rotated queries and keys
        """
        batch_size, seq_len, embed_dim = queries.shape
        
        # Validate dimensions
        assert embed_dim == self.embedding_dim, \
            f"Expected embedding dim {self.embedding_dim}, got {embed_dim}"
        assert seq_len <= self.max_seq_length, \
            f"Sequence length {seq_len} exceeds maximum {self.max_seq_length}"
        
        # If positions not provided, use default [0, 1, ..., seq_len-1]
        if positions is None:
            positions = torch.arange(seq_len, device=queries.device)
        
        # Get cos and sin for these positions
        # Shape: [seq_len, embedding_dim/2]
        cos = self.cos_cached[positions]
        sin = self.sin_cached[positions]
        
        # Reshape queries and keys to separate dimension pairs
        # From [batch_size, seq_len, embedding_dim] 
        # To   [batch_size, seq_len, embedding_dim/2, 2]
        queries_reshaped = queries.view(batch_size, seq_len, -1, 2)
        keys_reshaped = keys.view(batch_size, seq_len, -1, 2)
        
        # Expand cos and sin for broadcasting
        # From [seq_len, embedding_dim/2] 
        # To   [batch_size, seq_len, embedding_dim/2, 1]
        cos = cos.unsqueeze(0).unsqueeze(-1)  # [1, seq_len, embed_dim/2, 1]
        sin = sin.unsqueeze(0).unsqueeze(-1)  # [1, seq_len, embed_dim/2, 1]
        
        # Apply rotation to each pair
        queries_rotated = self._rotate_pair(queries_reshaped, cos, sin)
        keys_rotated = self._rotate_pair(keys_reshaped, cos, sin)
        
        # Reshape back to original format
        queries_rotated = queries_rotated.view(batch_size, seq_len, embed_dim)
        keys_rotated = keys_rotated.view(batch_size, seq_len, embed_dim)
        
        return queries_rotated, keys_rotated


class TransactionEmbedding(nn.Module):
    """
    Base class for transaction embeddings that converts raw transaction data
    into dense vector representations.
    
    This class handles the embedding of numerical transaction features and
    provides the foundation for more specialized embeddings.
    """
    
    def __init__(self, 
                 numerical_features: List[str],
                 embedding_dim: int = 128,
                 use_layer_norm: bool = True,
                 dropout_rate: float = 0.1):
        """
        Args:
            numerical_features: List of numerical feature names
            embedding_dim: Dimension of output embeddings
            use_layer_norm: Whether to apply layer normalization
            dropout_rate: Dropout probability for regularization
        """
        super(TransactionEmbedding, self).__init__()
        
        self.numerical_features = numerical_features
        self.embedding_dim = embedding_dim
        self.num_numerical = len(numerical_features)
        
        # Linear projection for numerical features
        # Maps from raw numerical features to embedding space
        self.numerical_projection = nn.Linear(self.num_numerical, embedding_dim)
        
        # Optional layer normalization for stable training
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Activation function
        self.activation = nn.GELU()  # GELU often works better than ReLU
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, 
                numerical_tensor: torch.Tensor,
                return_raw: bool = False) -> torch.Tensor:
        """
        Forward pass for transaction embedding.
        
        Args:
            numerical_tensor: Tensor of numerical features 
                             Shape: [batch_size, num_numerical]
            return_raw: If True, return before dropout and layer norm
        
        Returns:
            Embedded transactions: [batch_size, embedding_dim]
        """
        # Project numerical features to embedding dimension
        # Shape: [batch_size, embedding_dim]
        embedded = self.numerical_projection(numerical_tensor)
        
        # Apply activation
        embedded = self.activation(embedded)
        
        if return_raw:
            return embedded
        
        # Apply layer normalization if enabled
        if self.use_layer_norm:
            embedded = self.layer_norm(embedded)
        
        # Apply dropout
        embedded = self.dropout(embedded)
        
        return embedded


class TemporalTransactionEmbedding(TransactionEmbedding):
    """
    Transaction embedding with temporal encoding using Rotary Positional Embeddings.
    
    This extends the base transaction embedding by adding:
    1. Time-based features processing
    2. Rotary positional encoding for sequence position
    3. Time difference encoding for irregular transaction intervals
    """
    
    def __init__(self,
                 numerical_features: List[str],
                 time_features: List[str] = ['hour_of_day', 'day_of_week', 'time_since_last_tx'],
                 embedding_dim: int = 128,
                 max_sequence_length: int = 512,
                 use_rotary: bool = True,
                 use_time_delta: bool = True,
                 dropout_rate: float = 0.1):
        """
        Args:
            numerical_features: List of numerical feature names
            time_features: List of time-based feature names
            embedding_dim: Dimension of output embeddings
            max_sequence_length: Maximum sequence length for RoPE
            use_rotary: Whether to use rotary positional embeddings
            use_time_delta: Whether to encode time differences
            dropout_rate: Dropout probability
        """
        super(TemporalTransactionEmbedding, self).__init__(
            numerical_features=numerical_features,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate
        )
        
        self.time_features = time_features
        self.num_time_features = len(time_features)
        self.use_rotary = use_rotary
        self.use_time_delta = use_time_delta
        
        # Time feature projection (if we have time features)
        if self.num_time_features > 0:
            self.time_projection = nn.Linear(self.num_time_features, embedding_dim)
        
        # Time delta encoding for irregular intervals
        if use_time_delta:
            # Time delta is a scalar (seconds/minutes between transactions)
            # We'll project it to a vector using sinusoidal encoding
            self.time_delta_encoder = TimeDeltaEncoder(
                embedding_dim=embedding_dim,
                max_delta=86400  # 24 hours in seconds
            )
        
        # Rotary positional embedding for sequence positions
        if use_rotary:
            self.rotary_embedding = RotaryPositionalEmbedding(
                embedding_dim=embedding_dim,
                max_seq_length=max_sequence_length
            )
        
        # Combine projections
        # We'll have multiple components: numerical, time, time_delta
        # We need to combine them into the final embedding
        combine_input_dim = embedding_dim  # numerical projection
        
        if self.num_time_features > 0:
            combine_input_dim += embedding_dim  # add time projection dimension
        
        if use_time_delta:
            combine_input_dim += embedding_dim  # add time delta dimension
        
        self.combine_projection = nn.Linear(combine_input_dim, embedding_dim)
        
        # Final layer norm and dropout
        self.final_layer_norm = nn.LayerNorm(embedding_dim)
        self.final_dropout = nn.Dropout(dropout_rate)
    
    def forward(self,
                numerical_tensor: torch.Tensor,
                time_tensor: Optional[torch.Tensor] = None,
                time_deltas: Optional[torch.Tensor] = None,
                positions: Optional[torch.Tensor] = None,
                return_components: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with temporal features and rotary encoding.
        
        Args:
            numerical_tensor: Numerical features [batch_size, seq_len, num_numerical]
            time_tensor: Time features [batch_size, seq_len, num_time_features]
            time_deltas: Time differences between transactions [batch_size, seq_len]
            positions: Position indices for rotary encoding [batch_size, seq_len]
            return_components: If True, return all components separately
        
        Returns:
            Embedded transactions with temporal information
        """
        batch_size, seq_len, _ = numerical_tensor.shape
        
        # Reshape for processing (combine batch and sequence for linear layers)
        # From [batch_size, seq_len, features] to [batch_size * seq_len, features]
        numerical_flat = numerical_tensor.view(-1, self.num_numerical)
        
        # Get base numerical embedding
        # Shape: [batch_size * seq_len, embedding_dim]
        numerical_embedded = self.numerical_projection(numerical_flat)
        numerical_embedded = self.activation(numerical_embedded)
        
        # Reshape back to [batch_size, seq_len, embedding_dim]
        numerical_embedded = numerical_embedded.view(batch_size, seq_len, -1)
        
        # Initialize list of components
        components = [numerical_embedded]
        component_dict = {'numerical': numerical_embedded}
        
        # Process time features if provided
        if time_tensor is not None and self.num_time_features > 0:
            # Flatten time features
            time_flat = time_tensor.view(-1, self.num_time_features)
            
            # Project time features
            time_embedded = self.time_projection(time_flat)
            time_embedded = self.activation(time_embedded)
            
            # Reshape back
            time_embedded = time_embedded.view(batch_size, seq_len, -1)
            
            components.append(time_embedded)
            component_dict['time'] = time_embedded
        
        # Process time deltas if provided and enabled
        if time_deltas is not None and self.use_time_delta:
            # Encode time deltas
            # Shape: [batch_size, seq_len, embedding_dim]
            delta_embedded = self.time_delta_encoder(time_deltas)
            components.append(delta_embedded)
            component_dict['time_delta'] = delta_embedded
        
        # Combine all components along the feature dimension
        # Shape: [batch_size, seq_len, combined_dim]
        combined = torch.cat(components, dim=-1)
        
        # Flatten for projection
        combined_flat = combined.view(-1, combined.shape[-1])
        
        # Project to final embedding dimension
        final_flat = self.combine_projection(combined_flat)
        final = final_flat.view(batch_size, seq_len, -1)
        
        # Apply activation
        final = self.activation(final)
        
        # Apply rotary positional encoding if enabled
        if self.use_rotary and positions is not None:
            # For rotary encoding, we need to rotate queries and keys
            # But since we're just creating embeddings, we'll store the positions
            # and the rotary information will be used in attention layers
            # Here we're just adding positional information to the embeddings
            # In practice, rotary is applied during attention, not to embeddings directly
            # So we'll store the positions for later use
            component_dict['positions'] = positions
        
        # Apply final layer norm and dropout
        final = self.final_layer_norm(final)
        final = self.final_dropout(final)
        
        if return_components:
            component_dict['final'] = final
            return component_dict
        
        return final


class TimeDeltaEncoder(nn.Module):
    """
    Encodes time differences between transactions using sinusoidal embeddings.
    
    This is inspired by positional encodings but adapted for continuous time deltas.
    It allows the model to understand irregular time intervals between transactions.
    
    Mathematical Formulation:
        For a time delta Δt and dimension i, the encoding is:
        PE(Δt, 2i) = sin(Δt / 10000^(2i/d))
        PE(Δt, 2i+1) = cos(Δt / 10000^(2i/d))
    """
    
    def __init__(self, embedding_dim: int, max_delta: float = 86400):
        """
        Args:
            embedding_dim: Dimension of output encoding
            max_delta: Maximum expected time delta (for scaling)
        """
        super(TimeDeltaEncoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.max_delta = max_delta
        
        # Create frequency bands (geometric series)
        # Shape: [embedding_dim // 2]
        self.frequencies = 1.0 / (10000.0 ** (torch.arange(0, embedding_dim, 2) / embedding_dim))
        
        # Register as buffer (not trainable)
        self.register_buffer('freq_cached', self.frequencies)
        
        # Linear projection for learned components (optional)
        # Some methods use both sinusoidal and learned components
        self.learned_projection = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, time_deltas: torch.Tensor) -> torch.Tensor:
        """
        Encode time deltas.
        
        Args:
            time_deltas: Time differences between transactions
                        Shape: [batch_size, seq_len] or [batch_size * seq_len]
        
        Returns:
            Encoded time deltas: [batch_size, seq_len, embedding_dim]
        """
        original_shape = time_deltas.shape
        
        # Ensure we're working with 2D tensor [batch * seq, 1]
        if len(original_shape) == 2:
            batch_size, seq_len = original_shape
            time_deltas_flat = time_deltas.view(-1, 1)
        else:
            # Assume it's already flat
            time_deltas_flat = time_deltas.unsqueeze(-1)
            batch_size, seq_len = -1, -1  # Will be set after
        
        # Normalize time deltas to [0, 1] range for stability
        # This helps prevent extreme values in sin/cos
        time_deltas_norm = time_deltas_flat / self.max_delta
        time_deltas_norm = torch.clamp(time_deltas_norm, 0.0, 1.0)
        
        # Compute angles: Δt * frequencies
        # Shape: [batch * seq, embedding_dim // 2]
        angles = time_deltas_norm * self.freq_cached.unsqueeze(0)
        
        # Create sinusoidal encoding
        # Shape: [batch * seq, embedding_dim]
        encoding = torch.zeros(time_deltas_flat.shape[0], self.embedding_dim, 
                              device=time_deltas.device)
        
        # Fill even indices with sin, odd with cos
        encoding[:, 0::2] = torch.sin(angles)
        encoding[:, 1::2] = torch.cos(angles)
        
        # Apply learned projection
        encoding = self.learned_projection(encoding)
        
        # Reshape back to [batch, seq, embedding_dim] if needed
        if len(original_shape) == 2:
            encoding = encoding.view(batch_size, seq_len, self.embedding_dim)
        
        return encoding


class AmountAwareTransactionEmbedding(TemporalTransactionEmbedding):
    """
    Advanced transaction embedding that is aware of transaction amounts.
    
    This embedding specifically handles the importance of transaction amounts
    in fraud detection by creating amount-specific encodings and interactions.
    """
    
    def __init__(self,
                 numerical_features: List[str],
                 time_features: List[str],
                 amount_feature: str = 'amount',
                 embedding_dim: int = 128,
                 amount_embedding_dim: int = 32,
                 use_amount_gating: bool = True,
                 **kwargs):
        """
        Args:
            numerical_features: List of numerical features
            time_features: List of time features
            amount_feature: Name of the amount feature
            embedding_dim: Total embedding dimension
            amount_embedding_dim: Dimension for amount-specific embedding
            use_amount_gating: Whether to use gating mechanism for amount
            **kwargs: Additional arguments for parent class
        """
        super(AmountAwareTransactionEmbedding, self).__init__(
            numerical_features=numerical_features,
            time_features=time_features,
            embedding_dim=embedding_dim,
            **kwargs
        )
        
        self.amount_feature = amount_feature
        self.amount_embedding_dim = amount_embedding_dim
        self.use_amount_gating = use_amount_gating
        
        # Amount-specific embedding
        # We'll create multiple representations of amount
        self.amount_projection = nn.Linear(1, amount_embedding_dim)
        
        # Amount bucket embedding (categorical view of amount ranges)
        self.num_amount_buckets = 20  # 20 amount ranges
        self.amount_bucket_embedding = nn.Embedding(self.num_amount_buckets, amount_embedding_dim)
        
        # Amount gating mechanism
        # This learns to weight the importance of amount based on context
        if use_amount_gating:
            self.amount_gate = nn.Sequential(
                nn.Linear(amount_embedding_dim, amount_embedding_dim // 2),
                nn.ReLU(),
                nn.Linear(amount_embedding_dim // 2, 1),
                nn.Sigmoid()
            )
        
        # Combine amount embeddings with main embedding
        # We'll need to adjust the combine projection dimension
        old_combine_dim = self.combine_projection.in_features
        new_combine_dim = old_combine_dim + amount_embedding_dim + amount_embedding_dim
        
        # Replace combine projection with new one
        self.combine_projection = nn.Linear(new_combine_dim, embedding_dim)
        
        # Re-initialize weights for the new layer
        nn.init.xavier_uniform_(self.combine_projection.weight)
        nn.init.zeros_(self.combine_projection.bias)
    
    def _amount_to_bucket(self, amount: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous amount values to discrete buckets.
        
        Args:
            amount: Transaction amounts [batch_size, seq_len]
        
        Returns:
            Bucket indices [batch_size, seq_len]
        """
        # Log transform amounts for better bucket distribution
        # Add small epsilon to avoid log(0)
        log_amount = torch.log(amount + 1e-8)
        
        # Define bucket boundaries (can be learned or fixed)
        # Here we use fixed boundaries based on typical transaction amounts
        boundaries = torch.tensor([
            -float('inf'), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
            11, 12, 13, 14, 15, 16, 17, 18, float('inf')
        ], device=amount.device)
        
        # Digitize: find which bucket each amount falls into
        # This is a simplified version - in practice you'd use torch.bucketize
        buckets = torch.zeros_like(amount, dtype=torch.long)
        for i in range(1, len(boundaries) - 1):
            mask = (log_amount > boundaries[i]) & (log_amount <= boundaries[i + 1])
            buckets[mask] = i
        
        return buckets
    
    def forward(self,
                numerical_tensor: torch.Tensor,
                time_tensor: torch.Tensor,
                amount_tensor: torch.Tensor,
                time_deltas: Optional[torch.Tensor] = None,
                positions: Optional[torch.Tensor] = None,
                return_components: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with amount awareness.
        
        Args:
            numerical_tensor: Numerical features [batch, seq, num_numerical]
            time_tensor: Time features [batch, seq, num_time]
            amount_tensor: Transaction amounts [batch, seq]
            time_deltas: Time differences [batch, seq]
            positions: Position indices
            return_components: Return all components
        
        Returns:
            Amount-aware transaction embeddings
        """
        batch_size, seq_len = amount_tensor.shape
        
        # Get parent embeddings (numerical + time + time_delta)
        parent_output = super().forward(
            numerical_tensor=numerical_tensor,
            time_tensor=time_tensor,
            time_deltas=time_deltas,
            positions=positions,
            return_components=True
        )
        
        # Parent components include: numerical, time, time_delta, final
        parent_components = parent_output if isinstance(parent_output, dict) else {}
        
        # Process amount
        amount_flat = amount_tensor.view(-1, 1)  # [batch * seq, 1]
        
        # Continuous amount embedding
        amount_continuous = self.amount_projection(amount_flat)  # [batch * seq, amount_dim]
        amount_continuous = self.activation(amount_continuous)
        amount_continuous = amount_continuous.view(batch_size, seq_len, -1)
        
        # Amount bucket embedding
        amount_buckets = self._amount_to_bucket(amount_tensor)  # [batch, seq]
        amount_bucket_flat = amount_buckets.view(-1)
        amount_bucket_embedded = self.amount_bucket_embedding(amount_bucket_flat)
        amount_bucket_embedded = amount_bucket_embedded.view(batch_size, seq_len, -1)
        
        # Apply gating if enabled
        if self.use_amount_gating:
            # Gate value based on continuous amount embedding
            gate_flat = amount_continuous.view(-1, self.amount_embedding_dim)
            gate_values = self.amount_gate(gate_flat)  # [batch * seq, 1]
            gate_values = gate_values.view(batch_size, seq_len, 1)
            
            # Apply gate to bucket embedding
            amount_bucket_embedded = amount_bucket_embedded * gate_values
        
        # Combine all components
        all_components = []
        
        # Add parent components (excluding final)
        for comp_name in ['numerical', 'time', 'time_delta']:
            if comp_name in parent_components:
                all_components.append(parent_components[comp_name])
        
        # Add amount components
        all_components.append(amount_continuous)
        all_components.append(amount_bucket_embedded)
        
        # Concatenate
        combined = torch.cat(all_components, dim=-1)
        
        # Project to final dimension
        combined_flat = combined.view(-1, combined.shape[-1])
        final_flat = self.combine_projection(combined_flat)
        final = final_flat.view(batch_size, seq_len, -1)
        
        # Apply activation and normalization
        final = self.activation(final)
        final = self.final_layer_norm(final)
        final = self.final_dropout(final)
        
        if return_components:
            return {
                'numerical': parent_components.get('numerical'),
                'time': parent_components.get('time'),
                'time_delta': parent_components.get('time_delta'),
                'amount_continuous': amount_continuous,
                'amount_bucket': amount_bucket_embedded,
                'positions': positions,
                'final': final
            }
        
        return final


class MerchantCategoryEmbedding(nn.Module):
    """
    Specialized embedding for merchant categories with hierarchical structure.
    
    Merchant categories often have hierarchical relationships (e.g., 
    "Restaurants" -> "Fast Food" -> "Pizza"). This embedding captures
    these relationships using a hierarchical embedding approach.
    """
    
    def __init__(self,
                 num_categories: int,
                 num_subcategories: int,
                 category_embedding_dim: int = 32,
                 subcategory_embedding_dim: int = 16,
                 use_hierarchy: bool = True):
        """
        Args:
            num_categories: Number of main categories
            num_subcategories: Number of subcategories
            category_embedding_dim: Dimension for category embeddings
            subcategory_embedding_dim: Dimension for subcategory embeddings
            use_hierarchy: Whether to use hierarchical structure
        """
        super(MerchantCategoryEmbedding, self).__init__()
        
        self.use_hierarchy = use_hierarchy
        self.category_embedding_dim = category_embedding_dim
        self.subcategory_embedding_dim = subcategory_embedding_dim
        
        # Main category embedding
        self.category_embedding = nn.Embedding(num_categories, category_embedding_dim)
        
        if use_hierarchy:
            # Subcategory embedding
            self.subcategory_embedding = nn.Embedding(num_subcategories, subcategory_embedding_dim)
            
            # Hierarchy projection (maps subcategory to category space)
            self.hierarchy_projection = nn.Linear(subcategory_embedding_dim, category_embedding_dim)
            
            # Combination layer
            self.combine_layer = nn.Linear(category_embedding_dim * 2, category_embedding_dim)
    
    def forward(self,
                category_ids: torch.Tensor,
                subcategory_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for merchant category embedding.
        
        Args:
            category_ids: Main category IDs [batch, seq]
            subcategory_ids: Subcategory IDs [batch, seq] (optional)
        
        Returns:
            Merchant category embeddings [batch, seq, category_dim]
        """
        # Get main category embeddings
        cat_embedded = self.category_embedding(category_ids)
        
        if self.use_hierarchy and subcategory_ids is not None:
            # Get subcategory embeddings
            subcat_embedded = self.subcategory_embedding(subcategory_ids)
            
            # Project subcategory to main category space
            subcat_projected = self.hierarchy_projection(subcat_embedded)
            
            # Combine main and projected subcategory
            combined = torch.cat([cat_embedded, subcat_projected], dim=-1)
            final = self.combine_layer(combined)
            
            return final
        
        return cat_embedded


# Utility function to create a complete transaction embedding pipeline
def create_transaction_embedding_pipeline(embedding_dim: int = 128,
                                          use_rotary: bool = True,
                                          use_amount_aware: bool = True,
                                          **kwargs) -> nn.Module:
    """
    Factory function to create a complete transaction embedding pipeline.
    
    Args:
        embedding_dim: Dimension of final embeddings
        use_rotary: Whether to use rotary positional encodings
        use_amount_aware: Whether to use amount-aware embeddings
        **kwargs: Additional arguments for specific embedders
    
    Returns:
        Configured embedding module
    """
    
    # Define feature lists (in practice, these would come from config)
    numerical_features = [
        'transaction_velocity',
        'account_balance',
        'credit_score',
        'distance_from_home',
        'device_risk_score'
    ]
    
    time_features = [
        'hour_of_day',
        'day_of_week',
        'days_since_last_tx',
        'hour_sin',
        'hour_cos'
    ]
    
    if use_amount_aware:
        embedder = AmountAwareTransactionEmbedding(
            numerical_features=numerical_features,
            time_features=time_features,
            amount_feature='amount',
            embedding_dim=embedding_dim,
            use_rotary=use_rotary,
            **kwargs
        )
    else:
        embedder = TemporalTransactionEmbedding(
            numerical_features=numerical_features,
            time_features=time_features,
            embedding_dim=embedding_dim,
            use_rotary=use_rotary,
            **kwargs
        )
    
    return embedder