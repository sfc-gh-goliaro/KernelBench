import torch
import torch.nn as nn
import torch.nn.functional as F


TASK_CONFIG = {
    'comparison_mode': 'default',
    'atol': 1e-4,
    'rtol': 1e-4,
}

class Model(nn.Module):
    """
    Linear Attention mechanism that replaces softmax with kernel feature maps.
    Achieves O(N) complexity instead of O(N^2) for standard attention.
    
    Based on: "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
    """
    def __init__(self, dim, num_heads, seq_len, eps=1e-6):
        """
        :param dim: Model dimension
        :param num_heads: Number of attention heads
        :param seq_len: Sequence length
        :param eps: Small epsilon for numerical stability
        """
        super(Model, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.seq_len = seq_len
        self.eps = eps
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
    def feature_map(self, x):
        """ELU-based feature map for linear attention."""
        return F.elu(x) + 1
    
    def forward(self, x):
        """
        Forward pass for linear attention.
        
        :param x: Input tensor of shape (batch_size, seq_len, dim)
        :return: Output tensor of shape (batch_size, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply feature map (kernel trick)
        q = self.feature_map(q)
        k = self.feature_map(k)
        
        # Linear attention: (Q @ K^T) @ V becomes Q @ (K^T @ V)
        # This is O(N*d^2) instead of O(N^2*d)
        kv = torch.einsum('bhnd,bhnm->bhdm', k, v)
        qkv = torch.einsum('bhnd,bhdm->bhnm', q, kv)
        
        # Normalization
        k_sum = k.sum(dim=2, keepdim=True)
        normalizer = torch.einsum('bhnd,bhkd->bhnk', q, k_sum).clamp(min=self.eps)
        
        out = qkv / normalizer
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        return self.out_proj(out)


# Test parameters
batch_size = 32
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

