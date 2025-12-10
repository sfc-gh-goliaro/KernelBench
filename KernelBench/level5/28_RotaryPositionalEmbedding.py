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
    Rotary Positional Embedding (RoPE).
    
    Encodes position information through rotation of query and key vectors,
    enabling relative position awareness with linear complexity.
    
    Based on: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """
    def __init__(self, dim, max_seq_len, base=10000, scaling_factor=1.0):
        """
        :param dim: Dimension of embeddings (must be even)
        :param max_seq_len: Maximum sequence length
        :param base: Base for frequency computation
        :param scaling_factor: Factor for position scaling (for extended context)
        """
        super(Model, self).__init__()
        assert dim % 2 == 0, "Dimension must be even for RoPE"
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_factor = scaling_factor
        
        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos/sin cache
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len):
        """Build sin/cos cache for given sequence length."""
        positions = torch.arange(seq_len, device=self.inv_freq.device).float()
        positions = positions / self.scaling_factor
        
        # Outer product of positions and inverse frequencies
        freqs = torch.outer(positions, self.inv_freq)  # (seq_len, dim/2)
        
        # Duplicate for interleaving
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        
        self.register_buffer('cos_cache', emb.cos(), persistent=False)
        self.register_buffer('sin_cache', emb.sin(), persistent=False)
    
    def _rotate_half(self, x):
        """Rotate half the hidden dims of the input."""
        x1 = x[..., :self.dim // 2]
        x2 = x[..., self.dim // 2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, q, k, position_ids=None):
        """
        Apply rotary embeddings to query and key tensors.
        
        :param q: Query tensor (batch, num_heads, seq_len, head_dim)
        :param k: Key tensor (batch, num_heads, seq_len, head_dim)
        :param position_ids: Optional position indices (batch, seq_len)
        :return: Tuple of (rotated_q, rotated_k)
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Extend cache if needed
        if seq_len > self.cos_cache.shape[0]:
            self._build_cache(seq_len)
        
        if position_ids is None:
            # Use default positions
            cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)
            sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        else:
            # Gather cos/sin for specific positions
            cos = self.cos_cache[position_ids].unsqueeze(1)  # (batch, 1, seq, dim)
            sin = self.sin_cache[position_ids].unsqueeze(1)
        
        # Apply rotation
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_embed, k_embed
    
    def forward_single(self, x, position_ids=None):
        """
        Apply rotary embeddings to a single tensor.
        
        :param x: Input tensor (batch, num_heads, seq_len, head_dim)
        :param position_ids: Optional position indices
        :return: Rotated tensor
        """
        seq_len = x.shape[2]
        
        if seq_len > self.cos_cache.shape[0]:
            self._build_cache(seq_len)
        
        if position_ids is None:
            cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)
            sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        else:
            cos = self.cos_cache[position_ids].unsqueeze(1)
            sin = self.sin_cache[position_ids].unsqueeze(1)
        
        return (x * cos) + (self._rotate_half(x) * sin)


# Test parameters
batch_size = 32
num_heads = 32
seq_len = 2048
head_dim = 128

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
    return [head_dim, seq_len * 2]  # dim, max_seq_len

