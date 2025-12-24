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
    Multi-head Latent Attention (MLA) from DeepSeek-V3.
    
    Compresses KV cache using low-rank projection while maintaining
    full attention expressiveness. Reduces memory by 93.75%.
    
    Based on: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"
    """
    def __init__(self, dim, num_heads, kv_lora_rank=512, qk_rope_dim=64, 
                 v_head_dim=128, qk_nope_dim=128):
        """
        :param dim: Model dimension
        :param num_heads: Number of attention heads
        :param kv_lora_rank: Rank for KV compression
        :param qk_rope_dim: Dimension for rotary position encoding
        :param v_head_dim: Value head dimension
        :param qk_nope_dim: Query/Key dimension without position encoding
        """
        super(Model, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_dim = qk_rope_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_dim = qk_nope_dim
        self.qk_head_dim = qk_rope_dim + qk_nope_dim
        
        self.scale = self.qk_head_dim ** -0.5
        
        # Query projection (full dimension)
        self.q_proj = nn.Linear(dim, num_heads * self.qk_head_dim, bias=False)
        self.q_norm = nn.RMSNorm(self.qk_head_dim) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(self.qk_head_dim)
        
        # Compressed KV projection (low rank)
        self.kv_down_proj = nn.Linear(dim, kv_lora_rank, bias=False)
        self.kv_norm = nn.RMSNorm(kv_lora_rank) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(kv_lora_rank)
        
        # Up-projections from compressed representation
        self.k_up_proj = nn.Linear(kv_lora_rank, num_heads * self.qk_head_dim, bias=False)
        self.v_up_proj = nn.Linear(kv_lora_rank, num_heads * v_head_dim, bias=False)
        
        # RoPE components
        self.k_rope_proj = nn.Linear(kv_lora_rank, qk_rope_dim, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(num_heads * v_head_dim, dim, bias=False)
        
        # Precompute RoPE frequencies
        self._init_rope(max_seq_len=4096)
    
    def _init_rope(self, max_seq_len, base=10000):
        """Initialize rotary position embeddings."""
        inv_freq = 1.0 / (base ** (torch.arange(0, self.qk_rope_dim, 2).float() / self.qk_rope_dim))
        self.register_buffer('inv_freq', inv_freq)
        
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cache', emb.cos())
        self.register_buffer('sin_cache', emb.sin())
    
    def _apply_rope(self, x, position_ids=None):
        """Apply rotary position embedding."""
        seq_len = x.shape[2]
        
        if position_ids is None:
            cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)
            sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        else:
            cos = self.cos_cache[position_ids].unsqueeze(1)
            sin = self.sin_cache[position_ids].unsqueeze(1)
        
        x1 = x[..., :self.qk_rope_dim // 2]
        x2 = x[..., self.qk_rope_dim // 2:]
        
        x_rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + x_rotated * sin
    
    def forward(self, x, attention_mask=None, position_ids=None):
        """
        Forward pass for Multi-head Latent Attention.
        
        :param x: Input tensor (batch, seq_len, dim)
        :param attention_mask: Optional attention mask
        :param position_ids: Optional position indices
        :return: Output tensor (batch, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Query projection
        q = self.q_proj(x)
        q = q.view(batch_size, seq_len, self.num_heads, self.qk_head_dim)
        q = self.q_norm(q)
        q = q.transpose(1, 2)  # (batch, heads, seq, qk_head_dim)
        
        # Compressed KV
        kv_compressed = self.kv_down_proj(x)  # (batch, seq, kv_lora_rank)
        kv_compressed = self.kv_norm(kv_compressed)
        
        # Up-project K and V
        k = self.k_up_proj(kv_compressed)
        k = k.view(batch_size, seq_len, self.num_heads, self.qk_head_dim).transpose(1, 2)
        
        v = self.v_up_proj(kv_compressed)
        v = v.view(batch_size, seq_len, self.num_heads, self.v_head_dim).transpose(1, 2)
        
        # Apply RoPE to the rope dimensions of Q and K
        q_rope = q[..., :self.qk_rope_dim]
        k_rope = k[..., :self.qk_rope_dim]
        
        q_rope = self._apply_rope(q_rope, position_ids)
        k_rope = self._apply_rope(k_rope, position_ids)
        
        q = torch.cat([q_rope, q[..., self.qk_rope_dim:]], dim=-1)
        k = torch.cat([k_rope, k[..., self.qk_rope_dim:]], dim=-1)
        
        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_probs = F.softmax(attn_weights, dim=-1)
        
        # Apply to values
        out = torch.matmul(attn_probs, v)
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(out)


# Test parameters
batch_size = 4
seq_len = 2048
dim = 4096
num_heads = 32
kv_lora_rank = 512  # Compression rank

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
    return [dim, num_heads, kv_lora_rank]

