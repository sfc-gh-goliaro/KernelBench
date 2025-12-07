import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Grouped Query Attention (GQA).
    
    Uses fewer key-value heads than query heads, with each KV head
    shared across multiple query heads. Reduces memory bandwidth
    while maintaining quality.
    
    Based on: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
    """
    def __init__(self, dim, num_heads, num_kv_heads, head_dim=None, dropout=0.0):
        """
        :param dim: Model dimension
        :param num_heads: Number of query heads
        :param num_kv_heads: Number of key-value heads (< num_heads)
        :param head_dim: Dimension per head (default: dim // num_heads)
        :param dropout: Attention dropout rate
        """
        super(Model, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim if head_dim else dim // num_heads
        self.num_groups = num_heads // num_kv_heads
        self.scale = self.head_dim ** -0.5
        
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        
        # Projections
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x, attention_mask=None, kv_cache=None, use_cache=False):
        """
        Forward pass for Grouped Query Attention.
        
        :param x: Input tensor (batch, seq_len, dim)
        :param attention_mask: Optional causal mask (batch, 1, seq_q, seq_k)
        :param kv_cache: Optional tuple of (cached_keys, cached_values)
        :param use_cache: Whether to return updated KV cache
        :return: Output tensor (batch, seq_len, dim) and optionally KV cache
        """
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape Q: (batch, seq, num_heads, head_dim) -> (batch, num_heads, seq, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Reshape K, V: (batch, seq, num_kv_heads, head_dim) -> (batch, num_kv_heads, seq, head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Handle KV cache
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
        
        kv_seq_len = k.shape[2]
        
        # Repeat KV heads to match query heads
        # (batch, num_kv_heads, seq, head_dim) -> (batch, num_heads, seq, head_dim)
        k = k.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1)
        k = k.reshape(batch_size, self.num_heads, kv_seq_len, self.head_dim)
        
        v = v.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1)
        v = v.reshape(batch_size, self.num_heads, kv_seq_len, self.head_dim)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        # Softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply to values
        out = torch.matmul(attn_probs, v)
        
        # Reshape output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.out_proj(out)
        
        if use_cache:
            # Return original (non-repeated) K, V for caching
            new_k = k.view(batch_size, self.num_kv_heads, self.num_groups, kv_seq_len, self.head_dim)[:, :, 0]
            new_v = v.view(batch_size, self.num_kv_heads, self.num_groups, kv_seq_len, self.head_dim)[:, :, 0]
            return out, (new_k, new_v)
        
        return out


# Test parameters
batch_size = 8
seq_len = 1024
dim = 4096
num_heads = 32
num_kv_heads = 8  # 4 query heads per KV head
head_dim = 128

def get_inputs():
    return [torch.randn(batch_size, seq_len, dim)]

def get_init_inputs():
    return [dim, num_heads, num_kv_heads, head_dim]

