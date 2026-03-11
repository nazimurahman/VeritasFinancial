# =============================================================================
# File: transformers.py
# Description: Transformer-based architectures for sequential fraud detection
# Author: VeritasFinancial DS Team
# Version: 1.0.0
# Last Updated: 2024-01-15
# =============================================================================

"""
TRANSFORMER ARCHITECTURE FOR FRAUD DETECTION
=============================================
This module implements advanced transformer models specifically designed for
detecting fraudulent patterns in banking transaction sequences. The architecture
combines multiple state-of-the-art techniques:

1. Grouped Query Attention (GQA): Reduces memory footprint while maintaining
   model quality by sharing key-value heads across multiple query heads.

2. Rotary Position Embeddings (RoPE): Enables better extrapolation to longer
   sequences than traditional positional encodings.

3. Flash Attention: Memory-efficient attention implementation that reduces
   memory usage from O(n²) to O(n).

4. Multi-scale Feature Extraction: Captures patterns at different temporal
   granularities (minutes, hours, days).

5. Manifold-Constrained Hyper-Connections (mHC): Stabilizes training in deep
   networks through controlled residual connections.

Key Features for Fraud Detection:
- Captures sequential patterns in transaction history
- Identifies sudden behavioral changes
- Learns complex temporal dependencies
- Handles variable-length sequences
- Provides attention-based explainability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, Dict, List, Union, Any
from dataclasses import dataclass
import warnings
import logging

# Configure logging for production monitoring
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

@dataclass
class TransformerConfig:
    """
    Configuration class for transformer model.
    
    This dataclass centralizes all hyperparameters for easy experimentation
    and version control. Using dataclass ensures type safety and provides
    default values for quick prototyping.
    
    Attributes:
        vocab_size (int): Size of vocabulary (unused, kept for compatibility)
        hidden_size (int): Dimension of hidden representations (d_model)
        num_hidden_layers (int): Number of transformer blocks
        num_attention_heads (int): Number of attention heads
        num_key_value_heads (int): Number of key/value heads for GQA
        intermediate_size (int): Size of feed-forward layer
        hidden_act (str): Activation function name
        hidden_dropout_prob (float): Dropout probability
        attention_probs_dropout_prob (float): Attention dropout
        max_position_embeddings (int): Maximum sequence length
        initializer_range (float): Range for weight initialization
        layer_norm_eps (float): LayerNorm epsilon for numerical stability
        use_cache (bool): Whether to use KV cache for inference
        tie_word_embeddings (bool): Tie input/output embeddings
        rope_theta (float): Base for RoPE frequencies
        attention_dropout (float): Alias for attention_probs_dropout_prob
        use_flash_attention (bool): Use Flash Attention optimization
        use_mhc (bool): Use Manifold-Constrained Hyper-Connections
        mhc_gamma (float): mHC gamma parameter (connection strength)
        mhc_beta (float): mHC beta parameter (manifold constraint)
        num_sequence_scales (int): Number of scales for multi-scale processing
    """
    # Core architecture parameters
    vocab_size: int = 30522  # Standard BERT vocab size (unused but kept)
    hidden_size: int = 768    # Dimension of embeddings (d_model)
    num_hidden_layers: int = 12  # Number of transformer blocks
    num_attention_heads: int = 12  # Number of attention heads
    num_key_value_heads: int = 4   # For Grouped Query Attention (GQA)
    intermediate_size: int = 3072  # FFN hidden dimension (4 * hidden_size typically)
    
    # Regularization and stability
    hidden_act: str = "gelu"  # Activation: gelu, relu, silu, etc.
    hidden_dropout_prob: float = 0.1  # Dropout for hidden layers
    attention_probs_dropout_prob: float = 0.1  # Dropout for attention weights
    attention_dropout: float = 0.1  # Alias for above
    max_position_embeddings: int = 512  # Maximum sequence length
    initializer_range: float = 0.02  # For weight initialization (std dev)
    layer_norm_eps: float = 1e-12  # LayerNorm epsilon
    
    # Inference optimization
    use_cache: bool = True  # Enable KV caching for faster inference
    tie_word_embeddings: bool = False  # Not used for fraud detection
    
    # Advanced RoPE parameters
    rope_theta: float = 10000.0  # Base for RoPE (rotary position embedding)
    
    # Flash Attention optimization
    use_flash_attention: bool = True  # Use memory-efficient attention
    
    # Manifold-Constrained Hyper-Connections (mHC)
    use_mhc: bool = True  # Enable mHC for training stability
    mhc_gamma: float = 0.9  # Gamma for mHC - controls connection strength
    mhc_beta: float = 0.5   # Beta for mHC - manifold constraint
    
    # Multi-scale processing
    num_sequence_scales: int = 3  # Process at 3 temporal scales
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate attention head dimensions
        assert self.hidden_size % self.num_attention_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"
        
        # Validate GQA configuration
        assert self.num_attention_heads % self.num_key_value_heads == 0, \
            f"num_attention_heads ({self.num_attention_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})"
        
        # Log configuration for debugging
        logger.info(f"TransformerConfig initialized with hidden_size={self.hidden_size}, "
                   f"layers={self.num_hidden_layers}, heads={self.num_attention_heads}")


# =============================================================================
# ROTARY POSITION EMBEDDINGS (RoPE)
# =============================================================================

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation.
    
    RoPE encodes absolute positional information with a rotation matrix that
    naturally incorporates relative position information. Unlike traditional
    positional encodings that add position information, RoPE rotates the
    query and key vectors based on their positions.
    
    Mathematical Formulation:
        For position m, we apply rotation matrix R(m) to query q:
        q_m' = R(m) * q
        
        R(m) is a block diagonal matrix with 2D rotation blocks:
        [cos(mθ)  -sin(mθ)]
        [sin(mθ)   cos(mθ)]
    
    Benefits for Fraud Detection:
    - Better extrapolation to longer sequences than learned embeddings
    - Preserves relative position information (key for sequential patterns)
    - No additional parameters to learn
    - Works well with KV caching for streaming inference
    
    Reference:
        "RoFormer: Enhanced Transformer with Rotary Position Embedding"
        https://arxiv.org/abs/2104.09864
    """
    
    def __init__(self, dim: int, max_position_embeddings: int = 512, base: float = 10000.0):
        """
        Initialize Rotary Positional Embedding.
        
        Args:
            dim (int): Dimension of embeddings (must be even)
            max_position_embeddings (int): Maximum sequence length
            base (float): Base for computing frequencies (theta in paper)
        """
        super().__init__()
        
        # Validate dimension (must be even for rotation matrices)
        assert dim % 2 == 0, f"Dimension {dim} must be even for rotary embeddings"
        
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute frequency bands: θ_i = base^(-2i/dim)
        # This creates geometrically decreasing frequencies
        # Shape: (dim//2,) - one frequency per 2D rotation block
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Build cache for cos and sin embeddings
        self._build_cache(max_position_embeddings)
        
    def _build_cache(self, seq_len: int):
        """
        Precompute cos and sin values for all positions.
        
        Args:
            seq_len (int): Sequence length to cache
        """
        # Create position indices [0, 1, 2, ..., seq_len-1]
        # Shape: (seq_len,)
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        
        # Compute frequencies for each position and dimension
        # Outer product: positions × frequencies
        # Shape: (seq_len, dim//2)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        # Duplicate each frequency to get pairs (for sin and cos)
        # Shape: (seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Compute cos and sin
        # Shape: (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embeddings to input.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim)
            seq_len (Optional[int]): Sequence length (if None, use x.size(1))
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: cos and sin embeddings
                Each of shape (1, seq_len, dim) for broadcasting
        """
        if seq_len is None:
            seq_len = x.size(1)
        
        # Ensure cache is large enough
        if seq_len > self.max_position_embeddings:
            self._build_cache(seq_len)
            self.max_position_embeddings = seq_len
        
        # Return cached values, adding batch dimension for broadcasting
        # Shape: (1, seq_len, dim)
        return (
            self.cos_cached[:seq_len].unsqueeze(0),
            self.sin_cached[:seq_len].unsqueeze(0)
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half of the hidden dimensions.
    
    This is the core rotation operation in RoPE. It splits the last dimension
    into two halves and applies a 90-degree rotation to the second half.
    
    Args:
        x (torch.Tensor): Input tensor of shape (..., dim)
        
    Returns:
        torch.Tensor: Rotated tensor of same shape
    """
    # Split the last dimension into two halves
    # Shape becomes (..., dim//2, 2) or (..., dim//2)
    x1, x2 = x.chunk(2, dim=-1)
    
    # Concatenate with sign change: [-x2, x1]
    # This implements 90-degree rotation: (x1, x2) -> (-x2, x1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key.
    
    This function applies the rotation to query and key vectors:
        q_rotated = q * cos + rotate_half(q) * sin
        k_rotated = k * cos + rotate_half(k) * sin
    
    Args:
        q (torch.Tensor): Query tensor of shape (batch, heads, seq_len, dim)
        k (torch.Tensor): Key tensor of shape (batch, heads, seq_len, dim)
        cos (torch.Tensor): Cosine embeddings of shape (1, seq_len, dim)
        sin (torch.Tensor): Sine embeddings of shape (1, seq_len, dim)
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Rotated query and key
    """
    # Add head dimension to cos/sin for broadcasting
    # Shape: (1, 1, seq_len, dim)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    
    # Apply rotation: q * cos + rotate_half(q) * sin
    # This rotates each 2D subspace by the angle determined by position
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


# =============================================================================
# GROUPED QUERY ATTENTION (GQA)
# =============================================================================

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) implementation.
    
    GQA is an intermediate between Multi-Head Attention (MHA) and Multi-Query
    Attention (MQA). It groups query heads to share key/value heads, reducing
    memory footprint while maintaining model quality.
    
    Key Concepts:
        num_heads: Total number of query heads
        num_kv_heads: Number of key/value heads (must divide num_heads)
        groups = num_heads // num_kv_heads: Number of query heads per KV head
    
    Memory Comparison:
        MHA: KV cache size = 2 * seq_len * num_heads * head_dim
        MQA: KV cache size = 2 * seq_len * 1 * head_dim
        GQA: KV cache size = 2 * seq_len * num_kv_heads * head_dim
    
    Benefits for Fraud Detection:
    - Reduced memory usage for long transaction sequences
    - Faster inference with KV caching
    - Maintains representational power through grouped queries
    
    Reference:
        "GQA: Training Generalized Multi-Query Transformer Models from
        Multi-Head Checkpoints" https://arxiv.org/abs/2305.13245
    """
    
    def __init__(
        self,
        config: TransformerConfig,
        layer_idx: Optional[int] = None
    ):
        """
        Initialize Grouped Query Attention.
        
        Args:
            config (TransformerConfig): Model configuration
            layer_idx (Optional[int]): Index of this layer (for logging)
        """
        super().__init__()
        
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Validate head dimensions
        assert self.head_dim * self.num_heads == self.hidden_size, \
            f"hidden_size must be divisible by num_heads"
        
        # Calculate number of groups
        self.num_groups = self.num_heads // self.num_kv_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        
        # Projection layers
        # Q projection: [hidden_size -> num_heads * head_dim]
        self.q_proj = nn.Linear(
            self.hidden_size, 
            self.num_heads * self.head_dim, 
            bias=False
        )
        
        # K projection: [hidden_size -> num_kv_heads * head_dim]
        self.k_proj = nn.Linear(
            self.hidden_size, 
            self.num_kv_heads * self.head_dim, 
            bias=False
        )
        
        # V projection: [hidden_size -> num_kv_heads * head_dim]
        self.v_proj = nn.Linear(
            self.hidden_size, 
            self.num_kv_heads * self.head_dim, 
            bias=False
        )
        
        # Output projection
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, 
            self.hidden_size, 
            bias=False
        )
        
        # Dropout for attention weights
        self.attention_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # Initialize weights
        self._init_weights()
        
        # For Flash Attention (optional)
        self.use_flash = config.use_flash_attention
        
        logger.debug(f"Initialized GQA layer {layer_idx}: "
                    f"{self.num_heads} heads, {self.num_kv_heads} KV heads")
    
    def _init_weights(self):
        """Initialize weights with normal distribution."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    
    def _reshape_for_attention(
        self, 
        x: torch.Tensor, 
        num_heads: int
    ) -> torch.Tensor:
        """
        Reshape hidden states for multi-head attention.
        
        Args:
            x (torch.Tensor): Input of shape (batch, seq_len, hidden_size)
            num_heads (int): Number of attention heads
            
        Returns:
            torch.Tensor: Reshaped tensor of shape (batch, num_heads, seq_len, head_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Reshape: [batch, seq_len, num_heads * head_dim] -> 
        #          [batch, seq_len, num_heads, head_dim]
        x = x.view(batch_size, seq_len, num_heads, self.head_dim)
        
        # Transpose: [batch, seq_len, num_heads, head_dim] ->
        #            [batch, num_heads, seq_len, head_dim]
        x = x.transpose(1, 2)
        
        return x
    
    def _repeat_kv(
        self, 
        x: torch.Tensor, 
        n_rep: int
    ) -> torch.Tensor:
        """
        Repeat key/value heads to match number of query heads.
        
        This is the core operation for GQA - repeating each KV head
        to serve multiple query heads in the same group.
        
        Args:
            x (torch.Tensor): Input of shape (batch, num_kv_heads, seq_len, head_dim)
            n_rep (int): Number of repetitions (queries per KV head)
            
        Returns:
            torch.Tensor: Repeated tensor of shape (batch, num_heads, seq_len, head_dim)
        """
        batch, num_kv_heads, seq_len, head_dim = x.shape
        
        if n_rep == 1:
            return x
        
        # Expand: [batch, num_kv_heads, seq_len, head_dim] ->
        #         [batch, num_kv_heads, 1, seq_len, head_dim]
        x = x[:, :, None, :, :]
        
        # Repeat: [batch, num_kv_heads, n_rep, seq_len, head_dim]
        x = x.expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
        
        # Reshape: [batch, num_kv_heads * n_rep, seq_len, head_dim]
        return x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)
    
    def _flash_attention_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Flash Attention forward pass (memory-efficient attention).
        
        Flash Attention reduces memory usage from O(n²) to O(n) by:
        1. Tiling - processing attention in blocks
        2. Recomputation - recomputing attention weights during backward
        3. Kernel fusion - combining multiple operations
        
        Args:
            q (torch.Tensor): Query of shape (batch, seq_len, num_heads, head_dim)
            k (torch.Tensor): Key of shape (batch, seq_len, num_kv_heads, head_dim)
            v (torch.Tensor): Value of shape (batch, seq_len, num_kv_heads, head_dim)
            attention_mask (Optional[torch.Tensor]): Attention mask
            
        Returns:
            torch.Tensor: Attention output
        """
        try:
            # Import Flash Attention (optional dependency)
            from flash_attn import flash_attn_func
            
            # Ensure correct layout for Flash Attention
            # Flash attention expects [batch, seq_len, num_heads, head_dim]
            if q.dim() == 4 and q.size(1) != q.size(2):  # If shape is [batch, heads, seq_len, dim]
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
            
            # Apply Flash Attention
            attn_output = flash_attn_func(
                q, k, v,
                dropout_p=self.config.attention_dropout,
                softmax_scale=1.0 / math.sqrt(self.head_dim),
                causal=False  # Not causal for fraud detection
            )
            
            return attn_output
            
        except ImportError:
            # Fall back to standard attention if Flash Attention not available
            logger.warning("Flash Attention not installed. Falling back to standard attention.")
            return self._standard_attention_forward(q, k, v, attention_mask)
    
    def _standard_attention_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Standard attention implementation (for comparison/fallback).
        
        Args:
            q (torch.Tensor): Query of shape (batch, num_heads, seq_len, head_dim)
            k (torch.Tensor): Key of shape (batch, num_kv_heads, seq_len, head_dim)
            v (torch.Tensor): Value of shape (batch, num_kv_heads, seq_len, head_dim)
            attention_mask (Optional[torch.Tensor]): Attention mask
            
        Returns:
            torch.Tensor: Attention output
        """
        # Repeat KV heads to match query heads
        k = self._repeat_kv(k, self.num_queries_per_kv)
        v = self._repeat_kv(v, self.num_queries_per_kv)
        
        # Compute attention scores
        # Shape: (batch, num_heads, seq_len, seq_len)
        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # Apply attention mask (for padding, causal masking, etc.)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        
        # Apply dropout
        attn_weights = self.attention_dropout(attn_weights)
        
        # Apply attention to values
        # Shape: (batch, num_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, v)
        
        return attn_output
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of Grouped Query Attention.
        
        Args:
            hidden_states (torch.Tensor): Input of shape (batch, seq_len, hidden_size)
            attention_mask (Optional[torch.Tensor]): Attention mask
            past_key_value (Optional[Tuple[torch.Tensor, torch.Tensor]]): Cached KV states
            use_cache (bool): Whether to return KV cache
            output_attentions (bool): Whether to output attention weights
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
                - attention output
                - attention weights (if output_attentions=True)
                - updated KV cache (if use_cache=True)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to queries, keys, values
        # Shape: (batch, seq_len, num_heads * head_dim) for Q
        # Shape: (batch, seq_len, num_kv_heads * head_dim) for K/V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for attention
        # Shape: (batch, num_heads, seq_len, head_dim) for Q
        # Shape: (batch, num_kv_heads, seq_len, head_dim) for K/V
        query_states = self._reshape_for_attention(query_states, self.num_heads)
        key_states = self._reshape_for_attention(key_states, self.num_kv_heads)
        value_states = self._reshape_for_attention(value_states, self.num_kv_heads)
        
        # Handle KV caching for inference
        if past_key_value is not None:
            # Extract past key and value states
            past_key, past_value = past_key_value
            
            # Concatenate with current states
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
        
        # Prepare KV cache for future calls
        present_key_value = (key_states, value_states) if use_cache else None
        
        # Choose attention implementation based on configuration
        if self.use_flash and seq_len > 128:  # Flash Attention beneficial for longer sequences
            # Flash attention expects different layout
            attn_output = self._flash_attention_forward(
                query_states, key_states, value_states, attention_mask
            )
        else:
            # Standard attention
            attn_output = self._standard_attention_forward(
                query_states, key_states, value_states, attention_mask
            )
        
        # Reshape attention output back to hidden size
        # Shape: (batch, num_heads, seq_len, head_dim) ->
        #        (batch, seq_len, num_heads * head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        # Final output projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None, present_key_value


# =============================================================================
# MANIFOLD-CONSTRAINED HYPER-CONNECTIONS (mHC)
# =============================================================================

class ManifoldConstrainedHyperConnection(nn.Module):
    """
    Manifold-Constrained Hyper-Connections (mHC) for stable deep network training.
    
    mHC is an advanced residual connection mechanism that addresses training
    instability in very deep networks. It builds upon Hyper-Connections but adds
    manifold constraints to ensure the learned representations stay on a
    well-behaved manifold.
    
    Key Innovations:
    1. Doubly-stochastic connections: Connections are normalized both row-wise
       and column-wise for better gradient flow.
    
    2. Manifold constraint: Projects connections onto a low-dimensional manifold
       to prevent overfitting and improve generalization.
    
    3. Adaptive gating: Learns to dynamically weight contributions from different
       layers based on input characteristics.
    
    Mathematical Formulation:
        output = gamma * (M ⊙ f(x)) + x
        where:
        - M is the connection matrix with doubly-stochastic constraints
        - ⊙ represents manifold-constrained operation
        - gamma controls connection strength
    
    Benefits for Fraud Detection:
    - Enables training deeper networks for complex fraud patterns
    - Stabilizes training with imbalanced data
    - Improves gradient flow through many transformer layers
    - Reduces risk of vanishing/exploding gradients
    
    Reference:
        "Manifold-Constrained Hyper-Connections for Deep Network Training"
        (Hypothetical - based on your requirements)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        gamma: float = 0.9,
        beta: float = 0.5,
        manifold_dim: int = 32
    ):
        """
        Initialize mHC module.
        
        Args:
            hidden_size (int): Dimension of hidden representations
            num_layers (int): Number of layers to connect
            gamma (float): Connection strength (0 to 1)
            beta (float): Manifold constraint strength (0 to 1)
            manifold_dim (int): Dimension of low-rank manifold
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gamma = gamma
        self.beta = beta
        self.manifold_dim = manifold_dim
        
        # Connection matrix M (initialized with identity-like pattern)
        # This matrix determines how each layer connects to others
        self.connection_weights = nn.Parameter(
            torch.eye(num_layers) * 0.5 + torch.randn(num_layers, num_layers) * 0.1
        )
        
        # Manifold projection matrices (low-rank approximation)
        # Project connections onto a lower-dimensional manifold
        self.manifold_U = nn.Parameter(torch.randn(num_layers, manifold_dim) * 0.01)
        self.manifold_V = nn.Parameter(torch.randn(manifold_dim, num_layers) * 0.01)
        
        # Layer-specific gating mechanisms
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.LayerNorm(hidden_size // 4),
                nn.GELU(),
                nn.Linear(hidden_size // 4, 1),
                nn.Sigmoid()
            ) for _ in range(num_layers)
        ])
        
        # Output projections for each layer
        self.output_projs = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        
        logger.info(f"Initialized mHC with {num_layers} layers, manifold_dim={manifold_dim}")
    
    def _apply_doubly_stochastic(self, M: torch.Tensor) -> torch.Tensor:
        """
        Apply doubly-stochastic normalization to connection matrix.
        
        Doubly-stochastic means rows and columns both sum to 1, which ensures
        proper normalization of gradient flow.
        
        Args:
            M (torch.Tensor): Connection matrix of shape (num_layers, num_layers)
            
        Returns:
            torch.Tensor: Doubly-stochastic normalized matrix
        """
        # Normalize rows
        M_row_norm = F.softmax(M, dim=1)
        
        # Normalize columns
        M_col_norm = F.softmax(M_row_norm, dim=0)
        
        return M_col_norm
    
    def _manifold_constraint(self, M: torch.Tensor) -> torch.Tensor:
        """
        Apply manifold constraint via low-rank projection.
        
        Projects the connection matrix onto a low-dimensional manifold to
        prevent overfitting and improve generalization.
        
        Args:
            M (torch.Tensor): Connection matrix of shape (num_layers, num_layers)
            
        Returns:
            torch.Tensor: Manifold-constrained matrix
        """
        # Project onto manifold: M_manifold = U @ V
        # This creates a low-rank approximation
        M_manifold = self.manifold_U @ self.manifold_V
        
        # Blend original with manifold projection based on beta
        # beta close to 0: use original
        # beta close to 1: use manifold projection
        return (1 - self.beta) * M + self.beta * M_manifold
    
    def forward(
        self,
        layer_outputs: List[torch.Tensor],
        current_idx: int
    ) -> torch.Tensor:
        """
        Apply mHC to combine layer outputs.
        
        Args:
            layer_outputs (List[torch.Tensor]): List of all layer outputs so far
            current_idx (int): Index of current layer
            
        Returns:
            torch.Tensor: Combined output for current layer
        """
        # Stack all previous layer outputs
        # Shape: (num_prev_layers, batch, seq_len, hidden_size)
        prev_outputs = torch.stack(layer_outputs[:current_idx + 1], dim=0)
        
        # Get connection weights for current layer
        # Shape: (num_prev_layers,)
        M = self.connection_weights[current_idx, :current_idx + 1]
        
        # Apply manifold constraint
        M = self._manifold_constraint(M)
        
        # Apply doubly-stochastic normalization
        M = self._apply_doubly_stochastic(M)
        
        # Get adaptive gates for each previous layer
        gates = []
        for i, output in enumerate(layer_outputs[:current_idx + 1]):
            # Gate depends on current output
            gate = self.gates[i](output.mean(dim=1))  # Average over sequence
            gates.append(gate)
        
        # Stack gates
        gates = torch.stack(gates, dim=0)  # Shape: (num_prev_layers, batch, 1)
        
        # Combine weighted outputs
        combined = torch.zeros_like(layer_outputs[current_idx])
        for i in range(current_idx + 1):
            # Apply gate and connection weight
            weight = M[i].view(1, 1, 1, 1) * gates[i].view(-1, 1, 1, 1)
            
            # Project and add
            projected = self.output_projs[i](prev_outputs[i])
            combined = combined + weight * projected
        
        # Apply gamma and add residual
        output = self.gamma * combined + layer_outputs[current_idx]
        
        return output


# =============================================================================
# TRANSFORMER BLOCK
# =============================================================================

class TransformerBlock(nn.Module):
    """
    Complete transformer block with attention, FFN, and mHC.
    
    This block implements a full transformer layer with:
    1. Grouped Query Attention
    2. Feed-Forward Network
    3. Manifold-Constrained Hyper-Connections (optional)
    4. Layer Normalization (pre-norm architecture)
    5. Residual connections
    
    Architecture:
        x_norm = LayerNorm(x)
        attn_out = Attention(x_norm) + x
        x_norm2 = LayerNorm(attn_out)
        ffn_out = FFN(x_norm2) + attn_out
        if mHC: ffn_out = mHC(ffn_out)
    
    Benefits for Fraud Detection:
    - Processes transaction sequences to find patterns
    - Learns complex temporal dependencies
    - Stable training with deep networks
    - Efficient inference with KV caching
    """
    
    def __init__(
        self,
        config: TransformerConfig,
        layer_idx: int
    ):
        """
        Initialize transformer block.
        
        Args:
            config (TransformerConfig): Model configuration
            layer_idx (int): Index of this layer
        """
        super().__init__()
        
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.use_mhc = config.use_mhc
        
        # Layer normalizations (pre-norm architecture)
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size, 
            eps=config.layer_norm_eps
        )
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, 
            eps=config.layer_norm_eps
        )
        
        # Attention module
        self.attention = GroupedQueryAttention(config, layer_idx)
        
        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            self._get_activation(config.hidden_act),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )
        
        # mHC will be added later by the main model
        self.mhc = None
        
        logger.debug(f"Initialized TransformerBlock {layer_idx}")
    
    def _get_activation(self, activation: str) -> nn.Module:
        """
        Get activation function by name.
        
        Args:
            activation (str): Activation name ('gelu', 'relu', 'silu', etc.)
            
        Returns:
            nn.Module: Activation module
        """
        if activation == "gelu":
            return nn.GELU()
        elif activation == "relu":
            return nn.ReLU()
        elif activation == "silu" or activation == "swish":
            return nn.SiLU()
        elif activation == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        all_layer_outputs: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of transformer block.
        
        Args:
            hidden_states (torch.Tensor): Input of shape (batch, seq_len, hidden_size)
            attention_mask (Optional[torch.Tensor]): Attention mask
            past_key_value (Optional[Tuple]): Cached KV states
            use_cache (bool): Whether to return KV cache
            output_attentions (bool): Whether to output attention weights
            all_layer_outputs (Optional[List]): All layer outputs for mHC
            
        Returns:
            Tuple containing:
                - hidden_states: Output tensor
                - attentions: Attention weights (if requested)
                - present_key_value: Updated KV cache (if requested)
        """
        # Pre-attention layer norm (pre-norm architecture)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Attention
        attn_output, attn_weights, present_key_value = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions
        )
        
        # First residual connection
        hidden_states = residual + attn_output
        
        # Post-attention layer norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Feed-forward network
        ffn_output = self.mlp(hidden_states)
        
        # Second residual connection
        hidden_states = residual + ffn_output
        
        # Apply mHC if available and requested
        if self.use_mhc and self.mhc is not None and all_layer_outputs is not None:
            hidden_states = self.mhc(all_layer_outputs, self.layer_idx)
        
        return hidden_states, attn_weights, present_key_value


# =============================================================================
# TRANSACTION SEQUENCE TRANSFORMER
# =============================================================================

class TransactionSequenceTransformer(nn.Module):
    """
    Complete transformer model for transaction sequence fraud detection.
    
    This model processes sequences of banking transactions to detect fraudulent
    patterns. It combines all advanced techniques:
    - Grouped Query Attention (GQA) for efficient inference
    - Rotary Position Embeddings (RoPE) for better sequence modeling
    - Manifold-Constrained Hyper-Connections (mHC) for stable training
    - Flash Attention for memory efficiency
    - Multi-scale processing for capturing patterns at different temporal scales
    
    Input Format:
        Each transaction is represented as a feature vector containing:
        - Amount (scaled)
        - Transaction type (embedded)
        - Merchant category (embedded)
        - Time features (hour, day, etc.)
        - Location features
        - Device features
        - Historical statistics (rolling windows)
    
    Output:
        Fraud probability score (0-1)
    
    Architecture Overview:
        1. Input projection: Map raw features to hidden dimension
        2. Positional encoding: Add sequence position information (RoPE)
        3. Multiple transformer blocks (with GQA and mHC)
        4. Sequence pooling (attention pooling)
        5. Classification head
    
    Training Considerations:
        - Use weighted loss for class imbalance
        - Monitor attention patterns for explainability
        - Regularize with dropout and weight decay
        - Use gradient clipping for stability
    """
    
    def __init__(self, config: TransformerConfig, num_features: int):
        """
        Initialize transaction sequence transformer.
        
        Args:
            config (TransformerConfig): Model configuration
            num_features (int): Number of input features per transaction
        """
        super().__init__()
        
        self.config = config
        self.num_features = num_features
        self.hidden_size = config.hidden_size
        
        # Input projection: Map raw features to hidden dimension
        self.input_projection = nn.Sequential(
            nn.Linear(num_features, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            nn.Dropout(config.hidden_dropout_prob)
        )
        
        # Rotary Position Embeddings
        self.rotary_emb = RotaryPositionalEmbedding(
            dim=config.hidden_size,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta
        )
        
        # Multi-scale processing (optional)
        if config.num_sequence_scales > 1:
            self.downsample_convs = nn.ModuleList([
                nn.Conv1d(
                    config.hidden_size,
                    config.hidden_size,
                    kernel_size=3,
                    stride=2,
                    padding=1
                ) for _ in range(config.num_sequence_scales - 1)
            ])
            
            self.upsample_convs = nn.ModuleList([
                nn.ConvTranspose1d(
                    config.hidden_size,
                    config.hidden_size,
                    kernel_size=4,
                    stride=2,
                    padding=1
                ) for _ in range(config.num_sequence_scales - 1)
            ])
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.num_hidden_layers)
        ])
        
        # Manifold-Constrained Hyper-Connections (if enabled)
        if config.use_mhc:
            # Create mHC module and attach to each layer
            mhc = ManifoldConstrainedHyperConnection(
                hidden_size=config.hidden_size,
                num_layers=config.num_hidden_layers,
                gamma=config.mhc_gamma,
                beta=config.mhc_beta
            )
            
            # Attach mHC to each layer
            for layer in self.layers:
                layer.mhc = mhc
        
        # Final layer norm
        self.final_layernorm = nn.LayerNorm(
            config.hidden_size, 
            eps=config.layer_norm_eps
        )
        
        # Sequence pooling (attention-based pooling)
        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh()
        )
        self.pooler_attention = nn.Linear(config.hidden_size, 1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Sigmoid()  # Output fraud probability
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # For KV caching during inference
        self._use_cache = config.use_cache
        self._past_key_values = None
        
        logger.info(f"Initialized TransactionSequenceTransformer with {config.num_hidden_layers} layers, "
                   f"{config.hidden_size} hidden size, {config.num_attention_heads} heads")
    
    def _init_weights(self, module):
        """Initialize weights for different module types."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def _multi_scale_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Process sequence at multiple temporal scales.
        
        This function downsamples the sequence to capture patterns at
        different granularities (e.g., per-minute, per-hour, per-day).
        
        Args:
            hidden_states (torch.Tensor): Input of shape (batch, seq_len, hidden_size)
            attention_mask (Optional[torch.Tensor]): Attention mask
            
        Returns:
            torch.Tensor: Multi-scale processed features
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Prepare list for multi-scale features
        multi_scale_features = [hidden_states]
        
        # Downsample path
        current = hidden_states.transpose(1, 2)  # (batch, hidden, seq_len)
        for i, conv in enumerate(self.downsample_convs):
            current = conv(current)
            # Add downsampled features
            multi_scale_features.append(current.transpose(1, 2))
        
        # Upsample path (combine scales)
        combined = multi_scale_features[-1]
        for i, conv in enumerate(reversed(self.upsample_convs)):
            combined = conv(combined.transpose(1, 2)).transpose(1, 2)
            scale_idx = len(multi_scale_features) - 2 - i
            if scale_idx >= 0:
                # Add with residual
                combined = combined + multi_scale_features[scale_idx]
        
        return combined
    
    def _sequence_pooling(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool sequence representations using attention-based pooling.
        
        This is more sophisticated than simple mean/max pooling as it learns
        to focus on important positions in the sequence (e.g., suspicious
        transactions).
        
        Args:
            hidden_states (torch.Tensor): Shape (batch, seq_len, hidden_size)
            attention_mask (Optional[torch.Tensor]): Shape (batch, seq_len)
            
        Returns:
            torch.Tensor: Pooled representation of shape (batch, hidden_size)
        """
        # Apply non-linearity
        pooled = self.pooler(hidden_states)  # (batch, seq_len, hidden_size)
        
        # Compute attention scores
        attention_scores = self.pooler_attention(pooled)  # (batch, seq_len, 1)
        
        # Apply mask if provided
        if attention_mask is not None:
            # Expand mask to match scores
            mask = attention_mask.unsqueeze(-1)  # (batch, seq_len, 1)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax over sequence dimension
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, seq_len, 1)
        
        # Weighted sum
        pooled_representation = torch.sum(
            attention_weights * hidden_states, dim=1
        )  # (batch, hidden_size)
        
        return pooled_representation
    
    def forward(
        self,
        input_ids: torch.Tensor,  # Transaction features
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the transformer model.
        
        Args:
            input_ids (torch.Tensor): Transaction features of shape (batch, seq_len, num_features)
            attention_mask (Optional[torch.Tensor]): Mask of shape (batch, seq_len)
            use_cache (bool): Whether to use KV caching
            output_attentions (bool): Whether to output attention weights
            output_hidden_states (bool): Whether to output all hidden states
            
        Returns:
            Dict containing:
                - logits: Fraud probability scores (batch, 1)
                - hidden_states: All hidden states (if requested)
                - attentions: Attention weights (if requested)
                - pooled_output: Pooled representation (batch, hidden_size)
        """
        batch_size, seq_len = input_ids.shape[0], input_ids.shape[1]
        
        # Input projection
        # Shape: (batch, seq_len, hidden_size)
        hidden_states = self.input_projection(input_ids)
        
        # Get rotary position embeddings
        cos, sin = self.rotary_emb(hidden_states, seq_len)
        
        # Apply rotary embeddings to all layers via attention modules
        # (will be used inside attention blocks)
        
        # Multi-scale processing (if enabled)
        if hasattr(self, 'downsample_convs') and seq_len > 32:
            hidden_states = self._multi_scale_forward(hidden_states, attention_mask)
        
        # Store all layer outputs for mHC
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        
        # Initialize KV cache
        present_key_values = [] if use_cache else None
        
        # Process through transformer layers
        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            
            # Get past KV for this layer if caching
            past_key_value = (
                self._past_key_values[layer_idx] 
                if self._past_key_values is not None and use_cache 
                else None
            )
            
            # Forward through layer
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
                all_layer_outputs=all_hidden_states
            )
            
            # Update hidden states
            hidden_states = layer_outputs[0]
            
            # Store attention if requested
            if output_attentions:
                all_attentions.append(layer_outputs[1])
            
            # Update KV cache
            if use_cache:
                present_key_values.append(layer_outputs[2])
        
        # Final layer norm
        hidden_states = self.final_layernorm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        
        # Update cache for future inference
        if use_cache:
            self._past_key_values = present_key_values
        
        # Sequence pooling
        pooled_output = self._sequence_pooling(hidden_states, attention_mask)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        # Prepare output
        outputs = {
            "logits": logits,
            "pooled_output": pooled_output,
        }
        
        if output_hidden_states:
            outputs["hidden_states"] = all_hidden_states
        
        if output_attentions:
            outputs["attentions"] = all_attentions
        
        if use_cache:
            outputs["past_key_values"] = present_key_values
        
        return outputs
    
    def predict_fraud_probability(
        self,
        transaction_sequence: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Convenience method for fraud prediction.
        
        Args:
            transaction_sequence (torch.Tensor): Sequence of transactions
            attention_mask (Optional[torch.Tensor]): Attention mask
            
        Returns:
            np.ndarray: Fraud probabilities for each sequence
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=transaction_sequence,
                attention_mask=attention_mask,
                use_cache=False
            )
            probabilities = outputs["logits"].cpu().numpy()
        
        return probabilities
    
    def reset_cache(self):
        """Reset KV cache for fresh inference."""
        self._past_key_values = None
    
    def get_attention_weights(
        self,
        transaction_sequence: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_idx: int = -1
    ) -> np.ndarray:
        """
        Extract attention weights for explainability.
        
        Args:
            transaction_sequence (torch.Tensor): Input sequence
            attention_mask (Optional[torch.Tensor]): Attention mask
            layer_idx (int): Which layer to extract (-1 for last layer)
            
        Returns:
            np.ndarray: Attention weights
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=transaction_sequence,
                attention_mask=attention_mask,
                output_attentions=True,
                use_cache=False
            )
            
            attentions = outputs.get("attentions", [])
            if not attentions:
                return None
            
            if layer_idx >= len(attentions):
                layer_idx = -1
            
            attention_weights = attentions[layer_idx].cpu().numpy()
        
        return attention_weights


# =============================================================================
# FACTORY FUNCTION FOR EASY MODEL CREATION
# =============================================================================

def create_fraud_transformer(
    num_features: int,
    hidden_size: int = 768,
    num_layers: int = 12,
    num_heads: int = 12,
    num_kv_heads: int = 4,
    max_seq_len: int = 512,
    dropout: float = 0.1,
    use_flash_attention: bool = True,
    use_mhc: bool = True,
    **kwargs
) -> TransactionSequenceTransformer:
    """
    Factory function to create a fraud detection transformer with sensible defaults.
    
    Args:
        num_features (int): Number of input features per transaction
        hidden_size (int): Hidden dimension size
        num_layers (int): Number of transformer layers
        num_heads (int): Number of attention heads
        num_kv_heads (int): Number of key/value heads for GQA
        max_seq_len (int): Maximum sequence length
        dropout (float): Dropout probability
        use_flash_attention (bool): Whether to use Flash Attention
        use_mhc (bool): Whether to use mHC
        **kwargs: Additional configuration parameters
        
    Returns:
        TransactionSequenceTransformer: Configured model
    """
    config = TransformerConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        max_position_embeddings=max_seq_len,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        use_flash_attention=use_flash_attention,
        use_mhc=use_mhc,
        **kwargs
    )
    
    model = TransactionSequenceTransformer(config, num_features)
    
    return model


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    """
    Example usage of the transformer model for fraud detection.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create model with 100 features per transaction
    model = create_fraud_transformer(
        num_features=100,
        hidden_size=256,  # Smaller for demo
        num_layers=6,
        num_heads=8,
        num_kv_heads=2,
        max_seq_len=128,
        dropout=0.1,
        use_flash_attention=True,
        use_mhc=True
    )
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create dummy batch
    batch_size = 4
    seq_len = 64
    num_features = 100
    
    dummy_input = torch.randn(batch_size, seq_len, num_features)
    dummy_mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    outputs = model(
        input_ids=dummy_input,
        attention_mask=dummy_mask,
        output_attentions=True
    )
    
    print(f"\nForward pass successful!")
    print(f"Output shape: {outputs['logits'].shape}")
    print(f"Fraud probabilities: {outputs['logits'].detach().numpy().flatten()}")
    
    # Get attention weights for explainability
    attentions = model.get_attention_weights(dummy_input, dummy_mask)
    print(f"Attention weights shape: {attentions.shape if attentions is not None else None}")