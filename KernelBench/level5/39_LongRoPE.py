import torch
import torch.nn as nn
import torch.nn.functional as F
import math


TASK_CONFIG = {
    'comparison_mode': 'default',
    'atol': 1e-4,
    'rtol': 1e-4,
}

class Model(nn.Module):
    """
    LongRoPE: Extended Rotary Position Embedding for Long Context.
    
    Extends RoPE to much longer contexts through progressive
    dimension interpolation and non-uniform rescaling.
    
    Based on: "LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens"
    """
    def __init__(self, dim, base=10000, original_max_seq=4096, target_max_seq=65536):
        """
        :param dim: Dimension of embeddings (must be even)
        :param base: Base for frequency computation
        :param original_max_seq: Original training context length
        :param target_max_seq: Target extended context length
        """
        super(Model, self).__init__()
        assert dim % 2 == 0, "Dimension must be even"
        
        self.dim = dim
        self.base = base
        self.original_max_seq = original_max_seq
        self.target_max_seq = target_max_seq
        
        # Compute extension ratio
        self.extension_ratio = target_max_seq / original_max_seq
        
        # Non-uniform rescaling factors for different dimension groups
        # Lower dimensions get smaller scaling (preserve high-freq info)
        # Higher dimensions get larger scaling (extend low-freq reach)
        dim_half = dim // 2
        self.register_buffer('rescale_factors', 
            self._compute_rescale_factors(dim_half))
        
        # Compute adjusted inverse frequencies
        inv_freq = self._compute_inv_freq()
        self.register_buffer('inv_freq', inv_freq)
        
        # Build cache
        self._build_cache(target_max_seq)
    
    def _compute_rescale_factors(self, dim_half):
        """Compute non-uniform rescaling factors."""
        # Split dimensions into groups with different scaling
        factors = torch.ones(dim_half)
        
        # Lower dimensions (high frequency) - less scaling
        low_dim = dim_half // 4
        factors[:low_dim] = 1.0
        
        # Middle dimensions - moderate scaling
        mid_dim = dim_half // 2
        factors[low_dim:mid_dim] = math.sqrt(self.extension_ratio)
        
        # Higher dimensions (low frequency) - full scaling
        factors[mid_dim:] = self.extension_ratio
        
        return factors
    
    def _compute_inv_freq(self):
        """Compute adjusted inverse frequencies with non-uniform scaling."""
        # Base inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        
        # Apply non-uniform rescaling
        inv_freq = inv_freq / self.rescale_factors
        
        return inv_freq
    
    def _build_cache(self, seq_len):
        """Build cos/sin cache for extended positions."""
        positions = torch.arange(seq_len, dtype=torch.float)
        
        # Outer product
        freqs = torch.outer(positions, self.inv_freq)
        
        # Duplicate for complex rotation
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self.register_buffer('cos_cache', emb.cos(), persistent=False)
        self.register_buffer('sin_cache', emb.sin(), persistent=False)
    
    def _rotate_half(self, x):
        """Rotate half the hidden dims."""
        x1 = x[..., :self.dim // 2]
        x2 = x[..., self.dim // 2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, q, k, position_ids=None):
        """
        Apply LongRoPE to query and key tensors.
        
        :param q: Query tensor (batch, num_heads, seq_len, head_dim)
        :param k: Key tensor (batch, num_heads, seq_len, head_dim)
        :param position_ids: Optional position indices
        :return: Tuple of (rotated_q, rotated_k)
        """
        seq_len = q.shape[2]
        
        # Extend cache if needed
        if seq_len > self.cos_cache.shape[0]:
            self._build_cache(seq_len)
        
        if position_ids is None:
            cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)
            sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        else:
            cos = self.cos_cache[position_ids].unsqueeze(1)
            sin = self.sin_cache[position_ids].unsqueeze(1)
        
        # Apply rotation
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_embed, k_embed


# Test parameters
batch_size = 8
num_heads = 32
seq_len = 8192  # Extended context
head_dim = 128
original_max_seq = 4096
target_max_seq = 65536

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    return [q, k]

def get_init_inputs():
    return [head_dim, 10000, original_max_seq, target_max_seq]

