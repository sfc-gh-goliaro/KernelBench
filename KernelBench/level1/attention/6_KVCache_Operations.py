import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    KV Cache Operations
    
    Used by: All autoregressive LLMs during inference
    
    Manages key-value cache for efficient autoregressive generation.
    Supports appending new KV pairs and reading from cache.
    
    Shapes:
        new_k, new_v: (batch_size, num_heads, new_seq_len, head_dim)
        cached_k, cached_v: (batch_size, num_heads, cached_seq_len, head_dim)
        Output k, v: (batch_size, num_heads, total_seq_len, head_dim)
    """
    
    def __init__(self, num_heads: int, head_dim: int, max_seq_len: int = 8192):
        """
        Initialize KV cache operations.
        
        Args:
            num_heads: Number of attention heads
            head_dim: Dimension of each head
            max_seq_len: Maximum sequence length for cache
        """
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
    
    def forward(self, new_k: torch.Tensor, new_v: torch.Tensor, 
                cached_k: torch.Tensor = None, cached_v: torch.Tensor = None) -> tuple:
        """
        Append new KV to cache and return full KV tensors.
        
        Args:
            new_k: New key tensor (batch, num_heads, new_seq_len, head_dim)
            new_v: New value tensor (batch, num_heads, new_seq_len, head_dim)
            cached_k: Cached keys or None (batch, num_heads, cached_len, head_dim)
            cached_v: Cached values or None (batch, num_heads, cached_len, head_dim)
            
        Returns:
            Tuple of (full_k, full_v) concatenated tensors
        """
        if cached_k is None or cached_v is None:
            # No cache yet, just return new KV
            return new_k, new_v
        
        # Concatenate along sequence dimension
        full_k = torch.cat([cached_k, new_k], dim=2)
        full_v = torch.cat([cached_v, new_v], dim=2)
        
        return full_k, full_v


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "num_heads": 32, "head_dim": 128, "cached_seq_len": 1024, "new_seq_len": 1},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("attention", "6_KVCache_Operations")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    new_shape = (p["batch_size"], p["num_heads"], p["new_seq_len"], p["head_dim"])
    cached_shape = (p["batch_size"], p["num_heads"], p["cached_seq_len"], p["head_dim"])
    new_k = DISTRIBUTIONS[dist_name](new_shape, dtype=dtype, device=device)
    new_v = DISTRIBUTIONS[dist_name](new_shape, dtype=dtype, device=device)
    cached_k = DISTRIBUTIONS[dist_name](cached_shape, dtype=dtype, device=device)
    cached_v = DISTRIBUTIONS[dist_name](cached_shape, dtype=dtype, device=device)
    return [new_k, new_v, cached_k, cached_v]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["num_heads"], p["head_dim"]]
