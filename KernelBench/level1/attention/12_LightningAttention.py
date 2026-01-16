import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Lightning Attention
    
    Used by: Lightning attention models
    
    Linear attention with kernel feature maps for O(n) complexity.
    Uses feature map φ(x) to approximate softmax attention.
    
    Shapes:
        Input: (batch_size, seq_len, hidden_size)
        Output: (batch_size, seq_len, hidden_size)
    """
    
    def __init__(self, hidden_size: int, num_heads: int, feature_dim: int = None):
        """
        Initialize lightning attention.
        
        Args:
            hidden_size: Model hidden dimension
            num_heads: Number of attention heads
            feature_dim: Dimension of kernel features (default: head_dim)
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.feature_dim = feature_dim or self.head_dim
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Apply kernel feature map (ELU + 1 for positivity)."""
        return F.elu(x) + 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lightning attention forward pass.
        
        Uses linear attention: O = φ(Q)(φ(K)^T V) / φ(Q)(φ(K)^T 1)
        
        Args:
            x: Input tensor (batch, seq_len, hidden_size)
            
        Returns:
            Output tensor (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to (batch, num_heads, seq, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply feature map for linear attention
        q = self._feature_map(q * self.scale)
        k = self._feature_map(k)
        
        # Causal linear attention with cumulative sum
        # For causal: KV = cumsum(K^T V), K_sum = cumsum(K^T 1)
        kv = torch.einsum('bhsd,bhsv->bhdv', k, v)  # (batch, heads, head_dim, head_dim)
        k_sum = k.sum(dim=2)  # (batch, heads, head_dim)
        
        # For full sequence (non-causal version)
        # O = Q @ KV / (Q @ k_sum)
        numerator = torch.einsum('bhsd,bhdv->bhsv', q, kv)
        denominator = torch.einsum('bhsd,bhd->bhs', q, k_sum).unsqueeze(-1) + 1e-6
        
        attn_output = numerator / denominator
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        return self.o_proj(attn_output)


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "seq_length": 4096, "hidden_size": 4096, "num_heads": 32},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("attention", "12_LightningAttention")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["seq_length"], p["hidden_size"])
    x = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["hidden_size"], p["num_heads"]]
