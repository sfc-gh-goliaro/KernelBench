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
    RetNet (Retentive Network) retention mechanism.
    
    RetNet achieves training parallelism, low-cost inference, and good performance
    by replacing attention with a retention mechanism that supports both parallel
    and recurrent computation.
    
    Based on: "Retentive Network: A Successor to Transformer for Large Language Models"
    """
    def __init__(self, dim, num_heads, seq_len, double_v_dim=False):
        """
        :param dim: Model dimension
        :param num_heads: Number of retention heads
        :param seq_len: Maximum sequence length
        :param double_v_dim: Whether to double the value dimension
        """
        super(Model, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.seq_len = seq_len
        self.v_dim = self.head_dim * 2 if double_v_dim else self.head_dim
        
        # Decay rates (gamma) for each head - different heads have different decay
        # These are learned or set based on head index
        angles = 1.0 / (10000 ** (torch.arange(0, num_heads) / num_heads))
        self.register_buffer('decay', angles)
        
        # Projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, num_heads * self.v_dim, bias=False)
        
        # Group normalization (applied per head)
        self.group_norm = nn.GroupNorm(num_heads, num_heads * self.v_dim)
        
        # Output projection
        self.out_proj = nn.Linear(num_heads * self.v_dim, dim, bias=False)
        
        # xPos-like relative position encoding
        self.register_buffer('xpos_scale', self._build_xpos_scale(seq_len))
        
    def _build_xpos_scale(self, seq_len):
        """Build xPos scaling factors for relative position encoding."""
        scale = (torch.arange(0, self.head_dim, 2) + 0.4 * self.head_dim) / (1.4 * self.head_dim)
        scale = scale.unsqueeze(0).expand(seq_len, -1)
        pos = torch.arange(0, seq_len).unsqueeze(1)
        scale = scale ** pos
        scale = torch.stack([scale, scale], dim=-1).reshape(seq_len, self.head_dim)
        return scale
        
    def _build_decay_mask(self, seq_len, device):
        """Build the causal decay mask D for parallel training."""
        # D[i,j] = gamma^(i-j) if i >= j else 0
        positions = torch.arange(seq_len, device=device)
        distance = positions.unsqueeze(0) - positions.unsqueeze(1)  # (seq, seq)
        
        # Clamp negative distances (future positions)
        decay_mask = self.decay.unsqueeze(1).unsqueeze(2) ** distance.unsqueeze(0).clamp(min=0)
        
        # Apply causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        decay_mask = decay_mask * causal_mask.unsqueeze(0)
        
        return decay_mask
    
    def forward(self, x):
        """
        Forward pass for RetNet retention (parallel mode).
        
        :param x: Input tensor of shape (batch_size, seq_len, dim)
        :return: Output tensor of shape (batch_size, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.v_dim).transpose(1, 2)
        
        # Apply xPos scaling to Q and K
        xpos = self.xpos_scale[:seq_len].unsqueeze(0).unsqueeze(0)
        q = q * xpos
        k = k / xpos
        
        # Build decay mask
        decay_mask = self._build_decay_mask(seq_len, x.device)  # (heads, seq, seq)
        
        # Retention computation: R = (Q @ K^T * D) @ V
        # This is done in parallel mode for training
        retention = torch.matmul(q, k.transpose(-2, -1))  # (batch, heads, seq, seq)
        retention = retention * decay_mask.unsqueeze(0)
        
        # Apply to values
        out = torch.matmul(retention, v)  # (batch, heads, seq, v_dim)
        
        # Reshape and apply group norm
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.num_heads * self.v_dim)
        out = self.group_norm(out.transpose(-1, -2)).transpose(-1, -2)
        
        # Output projection
        return self.out_proj(out)


# Test parameters
batch_size = 16
seq_len = 512
dim = 512
num_heads = 8

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    return [torch.randn(batch_size, seq_len, dim)]

def get_init_inputs():
    return [dim, num_heads, seq_len]

