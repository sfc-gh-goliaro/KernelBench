import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Dual Chunk Attention RoPE
    
    Used by: Long-context models
    
    Dual chunk attention RoPE for processing very long contexts in chunks.
    Uses relative positions within chunks combined with inter-chunk positions.
    
    Shapes:
        Input: (batch_size, seq_len, num_heads, head_dim) for q and k
        Output: (batch_size, seq_len, num_heads, head_dim) for q and k
    """
    
    def __init__(self, head_dim: int, max_seq_len: int = 8192, base: float = 10000.0,
                 chunk_size: int = 2048):
        """
        Initialize Dual Chunk RoPE.
        
        Args:
            head_dim: Dimension of each attention head
            max_seq_len: Maximum sequence length
            base: Base for frequency computation
            chunk_size: Size of each chunk
        """
        super(Model, self).__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.chunk_size = chunk_size
        
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute for max sequence length
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def _compute_dual_chunk_positions(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute dual chunk position IDs."""
        positions = torch.arange(seq_len, device=device)
        
        # Intra-chunk positions (0 to chunk_size-1, repeating)
        intra_chunk = positions % self.chunk_size
        
        # Inter-chunk positions (chunk index * some factor)
        inter_chunk = (positions // self.chunk_size) * self.chunk_size
        
        # Combine: use intra-chunk for local attention, inter-chunk for global
        return intra_chunk + inter_chunk
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor = None) -> tuple:
        """
        Apply dual chunk RoPE to q and k.
        
        Args:
            q: Query tensor (batch, seq, num_heads, head_dim)
            k: Key tensor (batch, seq, num_heads, head_dim)
            position_ids: Optional position indices (overrides dual chunk computation)
            
        Returns:
            Tuple of rotated (q, k)
        """
        seq_len = q.shape[1]
        
        if position_ids is None:
            position_ids = self._compute_dual_chunk_positions(seq_len, q.device)
            cos = self.cos_cached[position_ids].unsqueeze(0).unsqueeze(2)
            sin = self.sin_cached[position_ids].unsqueeze(0).unsqueeze(2)
        else:
            cos = self.cos_cached[position_ids].unsqueeze(2)
            sin = self.sin_cached[position_ids].unsqueeze(2)
        
        q_rotated = self._apply_rotary(q, cos, sin)
        k_rotated = self._apply_rotary(k, cos, sin)
        
        return q_rotated, k_rotated
    
    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding."""
        x1 = x[..., :self.head_dim // 2]
        x2 = x[..., self.head_dim // 2:]
        rotated = torch.cat((-x2, x1), dim=-1)
        return x * cos + rotated * sin


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "seq_length": 8192, "num_heads": 32, "head_dim": 128, "chunk_size": 2048},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("embeddings", "8_DualChunk_RoPE")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["seq_length"], p["num_heads"], p["head_dim"])
    q = DISTRIBUTIONS["normal"](shape, dtype=dtype, device=device) if dist_name == "indices" else DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    k = DISTRIBUTIONS["normal"](shape, dtype=dtype, device=device) if dist_name == "indices" else DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [q, k]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["head_dim"], p["seq_length"], 10000.0, p["chunk_size"]]
