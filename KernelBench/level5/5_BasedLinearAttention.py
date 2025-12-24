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
    Based: Simple Linear Attention with Taylor Expansion.
    
    Uses second-order Taylor expansion of softmax for linear attention
    with improved expressiveness compared to standard linear attention.
    
    Based on: "Simple linear attention language models balance the recall-throughput tradeoff"
    """
    def __init__(self, dim, num_heads, seq_len, feature_dim=16):
        """
        :param dim: Model dimension
        :param num_heads: Number of attention heads
        :param seq_len: Sequence length
        :param feature_dim: Dimension of Taylor expansion features
        """
        super(Model, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        # Projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        # Learnable scaling
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(self.head_dim))
        
    def taylor_expansion(self, x):
        """
        Second-order Taylor expansion feature map.
        φ(x) ≈ [1, x, x⊗x / sqrt(2)]
        
        For efficiency, we use a simplified version.
        """
        # Normalize
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-9)
        
        # First order term
        first_order = x
        
        # Second order term (outer product approximated)
        # For efficiency, we use element-wise square
        second_order = x * x / math.sqrt(2)
        
        # Concatenate constant (1), first order, and second order
        batch, heads, seq, d = x.shape
        ones = torch.ones(batch, heads, seq, 1, device=x.device, dtype=x.dtype)
        
        return torch.cat([ones, first_order, second_order], dim=-1)
    
    def causal_linear_attention(self, q, k, v):
        """
        Causal linear attention using cumsum trick.
        """
        batch, heads, seq, d = v.shape
        feature_dim = q.shape[-1]
        
        # Compute KV cumsum for causal attention
        kv = torch.einsum('bhsd,bhsv->bhsdv', k, v)  # (batch, heads, seq, feature, value)
        kv_cumsum = torch.cumsum(kv, dim=2)  # Causal cumsum
        
        # Compute output
        out = torch.einsum('bhsd,bhsdv->bhsv', q, kv_cumsum)
        
        # Normalize
        k_cumsum = torch.cumsum(k, dim=2)  # (batch, heads, seq, feature)
        normalizer = torch.einsum('bhsd,bhsd->bhs', q, k_cumsum).unsqueeze(-1)
        normalizer = normalizer.clamp(min=1e-9)
        
        return out / normalizer
    
    def forward(self, x):
        """
        Forward pass for Based linear attention.
        
        :param x: Input tensor of shape (batch_size, seq_len, dim)
        :return: Output tensor of shape (batch_size, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scale Q and K
        q = q * self.scale
        
        # Apply Taylor expansion feature map
        q_features = self.taylor_expansion(q)
        k_features = self.taylor_expansion(k)
        
        # Causal linear attention
        out = self.causal_linear_attention(q_features, k_features, v)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        return self.out_proj(out)


# Test parameters
batch_size = 16
seq_len = 1024
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

