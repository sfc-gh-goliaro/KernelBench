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
    Attention with Linear Biases (ALiBi)
    
    Used by: BLOOM, MPT
    
    ALiBi adds a linear bias to attention scores based on the distance
    between query and key positions. This provides position information
    without explicit position embeddings and extrapolates well to longer
    sequences.
    
    Shapes:
        Input: (batch_size, seq_len, hidden_size)
        Output: (batch_size, seq_len, hidden_size)
    """
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        """
        Initialize ALiBi attention.
        
        Args:
            hidden_size: Model hidden dimension
            num_heads: Number of attention heads
            dropout: Attention dropout probability
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Compute ALiBi slopes for each head
        # Slopes are powers of 2^(-8/num_heads)
        slopes = self._get_alibi_slopes(num_heads)
        self.register_buffer('slopes', slopes)
    
    def _get_alibi_slopes(self, num_heads: int) -> torch.Tensor:
        """Compute ALiBi slopes for each head."""
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]
        
        if math.log2(num_heads).is_integer():
            slopes = get_slopes_power_of_2(num_heads)
        else:
            # For non-power-of-2, interpolate
            closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            slopes = slopes + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:num_heads - closest_power_of_2]
        
        return torch.tensor(slopes, dtype=torch.float32)
    
    def _get_alibi_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute ALiBi bias matrix."""
        # Create distance matrix: position i attending to position j
        # bias[i,j] = -slope * |i - j| (but only for j <= i in causal)
        positions = torch.arange(seq_len, device=device)
        distance = positions.unsqueeze(0) - positions.unsqueeze(1)  # (seq, seq)
        
        # Apply slopes: (num_heads, 1, 1) * (1, seq, seq)
        alibi = self.slopes.to(device).unsqueeze(1).unsqueeze(1) * distance.unsqueeze(0)
        
        return alibi  # (num_heads, seq, seq)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with ALiBi attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
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
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Add ALiBi bias
        alibi_bias = self._get_alibi_bias(seq_len, x.device)
        scores = scores + alibi_bias.unsqueeze(0)  # (batch, num_heads, seq, seq)
        
        # Apply causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        if self.dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        return self.o_proj(attn_output)


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "hidden_size": 4096, "num_heads": 32},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("attention", "9_ALiBi")

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
