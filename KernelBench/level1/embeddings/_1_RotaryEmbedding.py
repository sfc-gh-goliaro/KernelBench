import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn
from typing import Optional, Literal

class Model(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    
    Used by: Llama, Qwen, Mistral, Gemma, Yi, DeepSeek, Phi
    
    Applies rotary position embeddings to query and key tensors by rotating
    pairs of dimensions using precomputed cos/sin values based on position.
    
    Shapes (depends on layout parameter):
        layout="bshd": (batch_size, seq_len, num_heads, head_dim) - default
        layout="bhsd": (batch_size, num_heads, seq_len, head_dim) - Llama attention style
    """
    
    def __init__(
        self, 
        head_dim: int, 
        max_seq_len: int = 8192, 
        base: float = 10000.0,
        layout: Literal["bshd", "bhsd"] = "bshd"
    ):
        """
        Initialize RoPE.
        
        Args:
            head_dim: Dimension of each attention head (must be even)
            max_seq_len: Maximum sequence length for precomputed embeddings
            base: Base for the frequency computation
            layout: Input tensor layout. 
                    "bshd" = (batch, seq, heads, head_dim) - default
                    "bhsd" = (batch, heads, seq, head_dim) - Llama attention style
        """
        super(Model, self).__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.layout = layout
        
        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos and sin for all positions
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> tuple:
        """
        Apply rotary embeddings to q and k.
        
        Args:
            q: Query tensor. Shape depends on layout:
               - "bshd": (batch_size, seq_len, num_heads, head_dim)
               - "bhsd": (batch_size, num_heads, seq_len, head_dim)
            k: Key tensor with same layout as q
            position_ids: Optional position indices of shape (batch_size, seq_len)
            
        Returns:
            Tuple of (rotated_q, rotated_k) with same shapes as inputs
        """
        # Get seq_len based on layout
        if self.layout == "bhsd":
            seq_len = q.shape[2]  # (batch, heads, seq, head_dim)
        else:
            seq_len = q.shape[1]  # (batch, seq, heads, head_dim)
        
        if position_ids is None:
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]
        else:
            cos = self.cos_cached[position_ids]
            sin = self.sin_cached[position_ids]
        
        # Reshape for broadcasting based on layout
        if self.layout == "bhsd":
            # For (batch, heads, seq, head_dim): broadcast shape is (1, 1, seq, head_dim)
            if position_ids is None:
                cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, head_dim)
                sin = sin.unsqueeze(0).unsqueeze(0)
            else:
                cos = cos.unsqueeze(1)  # (batch, 1, seq, head_dim)
                sin = sin.unsqueeze(1)
        else:
            # For (batch, seq, heads, head_dim): broadcast shape is (1, seq, 1, head_dim)
            if position_ids is None:
                cos = cos.unsqueeze(0).unsqueeze(2)  # (1, seq, 1, head_dim)
                sin = sin.unsqueeze(0).unsqueeze(2)
            else:
                cos = cos.unsqueeze(2)  # (batch, seq, 1, head_dim)
                sin = sin.unsqueeze(2)
        
        q_rotated = self._apply_rotary(q, cos, sin)
        k_rotated = self._apply_rotary(k, cos, sin)
        
        return q_rotated, k_rotated
    
    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding to a single tensor."""
        # Split into two halves
        x1 = x[..., :self.head_dim // 2]
        x2 = x[..., self.head_dim // 2:]
        
        # Rotate
        rotated = torch.cat((-x2, x1), dim=-1)
        
        return x * cos + rotated * sin


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "num_heads": 32, "head_dim": 128},
    # Llama-3.1-8B: hidden_size=4096, num_heads=32, head_dim=128
    {"batch_size": 8, "seq_length": 4096, "num_heads": 32, "head_dim": 128},
    # Llama-3.1-70B: hidden_size=8192, num_heads=64, head_dim=128
    {"batch_size": 4, "seq_length": 4096, "num_heads": 64, "head_dim": 128},
    # Mistral-7B-v0.3: hidden_size=4096, num_heads=32, head_dim=128
    {"batch_size": 8, "seq_length": 4096, "num_heads": 32, "head_dim": 128},
    # Qwen2-VL-7B: num_heads=28, head_dim=128
    {"batch_size": 8, "seq_length": 2048, "num_heads": 28, "head_dim": 128},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("embeddings", "1_RotaryEmbedding")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    q = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_length"], p["num_heads"], p["head_dim"]), dtype=dtype, device=device)
    k = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_length"], p["num_heads"], p["head_dim"]), dtype=dtype, device=device)
    return [q, k]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["head_dim"]]
