import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    RetNet Retention
    
    Used by: RetNet
    
    RetNet retention mechanism with decay matrix. Supports both
    parallel and recurrent computation modes.
    
    Shapes:
        Input: (batch, seq_len, hidden_size)
        Output: (batch, seq_len, hidden_size)
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 8, double_v_dim: bool = True):
        """
        Initialize retention.
        
        Args:
            hidden_size: Model hidden dimension
            num_heads: Number of retention heads
            double_v_dim: Whether to double value dimension
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.v_dim = self.head_dim * 2 if double_v_dim else self.head_dim
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * self.v_dim, bias=False)
        self.g_proj = nn.Linear(hidden_size, num_heads * self.v_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.v_dim, hidden_size, bias=False)
        
        # Per-head decay rates (gamma)
        # Different heads have different decay rates for multi-scale retention
        decay_rates = 1 - torch.pow(2, -5 - torch.arange(num_heads, dtype=torch.float))
        self.register_buffer('decay_rates', decay_rates)
        
        self.scale = self.head_dim ** -0.5
    
    def _parallel_retention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Parallel retention computation."""
        batch_size, seq_len, num_heads, head_dim = q.shape
        
        # Build decay matrix D[i,j] = gamma^(i-j) for i >= j, 0 otherwise
        positions = torch.arange(seq_len, device=q.device)
        decay_mask = positions.unsqueeze(0) - positions.unsqueeze(1)  # (seq, seq)
        decay_mask = decay_mask.float()
        
        # Apply causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device))
        
        # Build D matrix for each head
        D = self.decay_rates.view(1, num_heads, 1, 1) ** decay_mask.unsqueeze(0).unsqueeze(0)
        D = D * causal_mask  # (1, num_heads, seq, seq)
        
        # Retention: (Q @ K^T) * D @ V
        # Q, K: (batch, seq, heads, head_dim) -> (batch, heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        qk = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch, heads, seq, seq)
        qk = qk * D  # Apply decay
        
        output = torch.matmul(qk, v)  # (batch, heads, seq, v_dim)
        return output.transpose(1, 2)  # (batch, seq, heads, v_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retention forward pass (parallel mode).
        
        Args:
            x: Input tensor (batch, seq_len, hidden_size)
            
        Returns:
            Output tensor (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # Projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.v_dim)
        g = self.g_proj(x).view(batch_size, seq_len, self.num_heads, self.v_dim)
        
        # Parallel retention
        retention_out = self._parallel_retention(q, k, v)
        
        # Apply swish gate
        output = retention_out * F.silu(g)
        
        # Reshape and project output
        output = output.reshape(batch_size, seq_len, -1)
        return self.o_proj(output)


# ============================================================================
# Benchmark Configuration
# ============================================================================
