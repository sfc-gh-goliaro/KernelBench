import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Fused Attention: QKV Projection + Attention + Output Projection.
    
    Combines the complete attention block:
    1. QKV projection (optionally packed)
    2. Scaled dot-product attention
    3. Output projection
    
    This is the full "attention layer" fusion for maximum efficiency.
    
    Reference: FlashAttention, xFormers, Triton attention
    """
    def __init__(self, dim, num_heads, num_kv_heads=None, head_dim=None,
                 dropout=0.0, bias=False, causal=True):
        """
        :param dim: Model dimension
        :param num_heads: Number of query heads
        :param num_kv_heads: Number of KV heads (for GQA/MQA)
        :param head_dim: Head dimension
        :param dropout: Attention dropout
        :param bias: Use bias in projections
        :param causal: Use causal masking
        """
        super(Model, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads else num_heads
        self.head_dim = head_dim if head_dim else dim // num_heads
        self.dropout = dropout
        self.causal = causal
        self.scale = self.head_dim ** -0.5
        
        # Packed QKV projection
        self.qkv_dim = (num_heads + 2 * self.num_kv_heads) * self.head_dim
        self.qkv_proj = nn.Linear(dim, self.qkv_dim, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(num_heads * self.head_dim, dim, bias=bias)
    
    def forward(self, x, attention_mask=None, kv_cache=None):
        """
        Fused attention forward.
        
        :param x: Input (batch, seq, dim)
        :param attention_mask: Optional attention mask
        :param kv_cache: Optional KV cache for incremental decoding
        :return: Attention output, optional updated KV cache
        """
        batch_size, seq_len, _ = x.shape
        
        # === FUSED KERNEL START ===
        # QKV projection (single matmul)
        qkv = self.qkv_proj(x)
        
        # Split Q, K, V
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        
        q = qkv[..., :q_dim]
        k = qkv[..., q_dim:q_dim + kv_dim]
        v = qkv[..., q_dim + kv_dim:]
        
        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Handle KV cache
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        
        new_kv_cache = (k, v)
        
        # GQA: expand KV to match Q heads
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(
                batch_size, self.num_heads, -1, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(
                batch_size, self.num_heads, -1, self.head_dim)
        
        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Causal mask
        if self.causal:
            query_len, key_len = q.shape[2], k.shape[2]
            causal_mask = torch.triu(
                torch.ones(query_len, key_len, device=x.device, dtype=torch.bool),
                diagonal=key_len - query_len + 1
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # Additional attention mask
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        # Softmax
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # Dropout
        if self.training and self.dropout > 0:
            attn_probs = F.dropout(attn_probs, p=self.dropout, training=True)
        
        # Apply attention to values
        attn_out = torch.matmul(attn_probs, v)
        
        # Reshape and output projection
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(attn_out)
        # === FUSED KERNEL END ===
        
        return output, new_kv_cache


# With RoPE fusion
class FusedAttentionWithRoPE(nn.Module):
    """
    Fused Attention with RoPE embedded.
    
    Includes rotary position embedding in the fused kernel.
    """
    def __init__(self, dim, num_heads, num_kv_heads=None, head_dim=None,
                 max_position=8192, rope_theta=10000.0):
        super(FusedAttentionWithRoPE, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads else num_heads
        self.head_dim = head_dim if head_dim else dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv_dim = (num_heads + 2 * self.num_kv_heads) * self.head_dim
        self.qkv_proj = nn.Linear(dim, self.qkv_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)
        
        # RoPE
        self._init_rope(max_position, rope_theta)
    
    def _init_rope(self, max_position, theta):
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer('inv_freq', inv_freq)
        
        t = torch.arange(max_position).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def _apply_rope(self, x, position_ids):
        cos = self.cos_cached[position_ids].unsqueeze(1)
        sin = self.sin_cached[position_ids].unsqueeze(1)
        
        x1, x2 = x[..., :self.head_dim//2], x[..., self.head_dim//2:]
        rotated = torch.cat([-x2, x1], dim=-1)
        
        return x * cos + rotated * sin
    
    def forward(self, x, position_ids=None):
        batch_size, seq_len, _ = x.shape
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Fused: QKV proj + RoPE + attention + output proj
        qkv = self.qkv_proj(x)
        
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        
        q = qkv[..., :q_dim].view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = qkv[..., q_dim:q_dim+kv_dim].view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = qkv[..., q_dim+kv_dim:].view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        q = self._apply_rope(q, position_ids)
        k = self._apply_rope(k, position_ids)
        
        # GQA expansion
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(batch_size, self.num_heads, -1, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(batch_size, self.num_heads, -1, self.head_dim)
        
        # Causal attention
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale + 
                        torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1),
                        dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, seq_len, -1)
        
        return self.out_proj(out)


# Test parameters
batch_size = 8
seq_len = 2048
dim = 4096
num_heads = 32
num_kv_heads = 8

def get_inputs():
    return [torch.randn(batch_size, seq_len, dim)]

def get_init_inputs():
    return [dim, num_heads, num_kv_heads]

