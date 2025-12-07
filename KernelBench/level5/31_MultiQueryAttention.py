import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Multi-Query Attention (MQA).
    
    Uses a single key-value head shared across all query heads.
    Maximizes memory bandwidth efficiency at inference time.
    
    Based on: "Fast Transformer Decoding: One Write-Head is All You Need"
    """
    def __init__(self, dim, num_heads, head_dim=None, dropout=0.0):
        """
        :param dim: Model dimension
        :param num_heads: Number of query heads
        :param head_dim: Dimension per head (default: dim // num_heads)
        :param dropout: Attention dropout rate
        """
        super(Model, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim else dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query projection: full multi-head
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        
        # Key-Value projection: single head
        self.k_proj = nn.Linear(dim, self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.head_dim, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x, attention_mask=None, kv_cache=None, use_cache=False):
        """
        Forward pass for Multi-Query Attention.
        
        :param x: Input tensor (batch, seq_len, dim)
        :param attention_mask: Optional causal mask
        :param kv_cache: Optional tuple of (cached_keys, cached_values)
        :param use_cache: Whether to return updated KV cache
        :return: Output tensor and optionally KV cache
        """
        batch_size, seq_len, _ = x.shape
        
        # Project queries (multi-head)
        q = self.q_proj(x)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Project keys and values (single head)
        k = self.k_proj(x)  # (batch, seq, head_dim)
        v = self.v_proj(x)  # (batch, seq, head_dim)
        
        # Handle KV cache
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=1)
            v = torch.cat([cached_v, v], dim=1)
        
        kv_seq_len = k.shape[1]
        
        # Expand K, V to match query heads
        # (batch, seq, head_dim) -> (batch, num_heads, seq, head_dim)
        k = k.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        v = v.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        else:
            # Default causal mask for autoregressive
            causal_mask = torch.triu(
                torch.full((seq_len, kv_seq_len), float('-inf'), device=x.device),
                diagonal=kv_seq_len - seq_len + 1
            )
            attn_scores = attn_scores + causal_mask
        
        # Softmax
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply to values
        out = torch.matmul(attn_probs, v)
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.out_proj(out)
        
        if use_cache:
            # Return single-head K, V for caching
            new_k = k[:, 0]  # (batch, seq, head_dim)
            new_v = v[:, 0]
            return out, (new_k, new_v)
        
        return out


# Test parameters
batch_size = 8
seq_len = 1024
dim = 2048
num_heads = 16
head_dim = 128

def get_inputs():
    return [torch.randn(batch_size, seq_len, dim)]

def get_init_inputs():
    return [dim, num_heads, head_dim]

