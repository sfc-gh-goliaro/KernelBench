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
    GLM-style Prefix Language Model Attention.
    
    Implements bidirectional attention over prefix and autoregressive
    attention for generation. Used in GLM-4 and ChatGLM.
    
    Based on: "GLM: General Language Model Pretraining with Autoregressive Blank Infilling"
    """
    def __init__(self, dim, num_heads, num_kv_heads=None, head_dim=None, 
                 rope_base=10000, max_seq_len=8192):
        """
        :param dim: Model dimension
        :param num_heads: Number of query heads
        :param num_kv_heads: Number of KV heads (for GQA)
        :param head_dim: Dimension per head
        :param rope_base: Base for RoPE
        :param max_seq_len: Maximum sequence length
        """
        super(Model, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads else num_heads
        self.head_dim = head_dim if head_dim else dim // num_heads
        self.num_groups = num_heads // self.num_kv_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV projection (with GQA support)
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)
        
        # RoPE
        self.rope_base = rope_base
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Build RoPE cache
        self._build_rope_cache(max_seq_len)
        
        # Layer norms (GLM uses post-norm in attention)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
    
    def _build_rope_cache(self, seq_len):
        """Build RoPE sin/cos cache."""
        positions = torch.arange(seq_len).float()
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cache', emb.cos(), persistent=False)
        self.register_buffer('sin_cache', emb.sin(), persistent=False)
    
    def _apply_rope(self, x, position_ids):
        """Apply rotary position embedding."""
        cos = self.cos_cache[position_ids].unsqueeze(1)
        sin = self.sin_cache[position_ids].unsqueeze(1)
        
        x1 = x[..., :self.head_dim // 2]
        x2 = x[..., self.head_dim // 2:]
        rotated = torch.cat([-x2, x1], dim=-1)
        
        return x * cos + rotated * sin
    
    def forward(self, x, prefix_length, position_ids=None, kv_cache=None):
        """
        Forward pass with prefix LM attention.
        
        :param x: Input tensor (batch, seq_len, dim)
        :param prefix_length: Length of bidirectional prefix
        :param position_ids: Optional position indices
        :param kv_cache: Optional KV cache for generation
        :return: Output tensor (batch, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        
        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply QK layer norms (GLM-4 style)
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Apply RoPE
        q = self._apply_rope(q, position_ids)
        k = self._apply_rope(k, position_ids)
        
        # Handle KV cache
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        
        kv_seq_len = k.shape[2]
        
        # Expand KV for GQA
        if self.num_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1)
            k = k.reshape(batch_size, self.num_heads, kv_seq_len, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1)
            v = v.reshape(batch_size, self.num_heads, kv_seq_len, self.head_dim)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Build GLM attention mask
        # Prefix: bidirectional (can attend to all prefix tokens)
        # Generation: causal (can only attend to past)
        attn_mask = torch.zeros(seq_len, kv_seq_len, device=x.device)
        
        # Prefix region: full attention
        attn_mask[:prefix_length, :prefix_length] = 0
        
        # Generation region: causal
        for i in range(prefix_length, seq_len):
            # Can attend to all prefix and past generation tokens
            attn_mask[i, prefix_length + (kv_seq_len - seq_len):i + (kv_seq_len - seq_len) + 1] = 0
            attn_mask[i, i + (kv_seq_len - seq_len) + 1:] = float('-inf')
        
        # Generation tokens can attend to prefix
        attn_mask[prefix_length:, :prefix_length] = 0
        
        attn_scores = attn_scores + attn_mask.unsqueeze(0).unsqueeze(0)
        
        # Softmax and apply
        attn_probs = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_probs, v)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.out_proj(out)
        
        return out


# Test parameters
batch_size = 4
seq_len = 1024
prefix_length = 256  # Bidirectional prefix
dim = 4096
num_heads = 32
num_kv_heads = 2  # MQA style

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    x = torch.randn(batch_size, seq_len, dim)
    return [x, prefix_length]

def get_init_inputs():
    return [dim, num_heads, num_kv_heads]

