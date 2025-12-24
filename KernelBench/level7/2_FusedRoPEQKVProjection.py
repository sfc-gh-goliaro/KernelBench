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
    Fused Rotary Positional Embedding + QKV Projection.
    
    Combines the QKV linear projection with rotary embedding application
    in a single fused kernel to eliminate intermediate memory access.
    
    Critical optimization for LLM attention:
    - Single read of input tensor
    - Apply RoPE during/after projection
    - Single write of rotated Q, K, V
    
    Reference: FlashAttention RoPE fusion, vLLM
    """
    def __init__(self, dim, num_heads, num_kv_heads=None, head_dim=None,
                 max_position=8192, rope_theta=10000.0):
        """
        :param dim: Model dimension
        :param num_heads: Number of query heads
        :param num_kv_heads: Number of KV heads (for GQA)
        :param head_dim: Dimension per head
        :param max_position: Maximum sequence position for RoPE
        :param rope_theta: RoPE theta base
        """
        super(Model, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads else num_heads
        self.head_dim = head_dim if head_dim else dim // num_heads
        self.rope_theta = rope_theta
        
        # QKV projections
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        
        # Precompute RoPE frequencies
        self._init_rope(max_position)
    
    def _init_rope(self, max_position):
        """Initialize RoPE cos/sin cache."""
        inv_freq = 1.0 / (self.rope_theta ** (
            torch.arange(0, self.head_dim, 2).float() / self.head_dim
        ))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute for max positions
        t = torch.arange(max_position).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def _rotate_half(self, x):
        """Rotate half the hidden dims of the input."""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def _apply_rope(self, x, cos, sin):
        """Apply rotary positional embedding."""
        return (x * cos) + (self._rotate_half(x) * sin)
    
    def forward(self, hidden_states, position_ids=None):
        """
        Fused RoPE + QKV projection.
        
        :param hidden_states: Input (batch, seq, dim)
        :param position_ids: Position indices (batch, seq)
        :return: Tuple of (Q, K, V) with RoPE applied to Q and K
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # === FUSED KERNEL START ===
        # Step 1: QKV projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Step 2: Apply RoPE to Q and K
        cos = self.cos_cached[position_ids].unsqueeze(1)  # (batch, 1, seq, head_dim)
        sin = self.sin_cached[position_ids].unsqueeze(1)
        
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)
        # === FUSED KERNEL END ===
        
        return q, k, v


# Variant with packed QKV weights
class FusedRoPEPackedQKV(nn.Module):
    """
    Fused RoPE with packed QKV projection.
    
    Single matrix multiplication for Q, K, V with inline RoPE.
    """
    def __init__(self, dim, num_heads, num_kv_heads=None, head_dim=None,
                 max_position=8192, rope_theta=10000.0):
        super(FusedRoPEPackedQKV, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads else num_heads
        self.head_dim = head_dim if head_dim else dim // num_heads
        self.rope_theta = rope_theta
        
        # Packed QKV
        qkv_dim = (num_heads + 2 * self.num_kv_heads) * self.head_dim
        self.qkv_proj = nn.Linear(dim, qkv_dim, bias=False)
        
        self._init_rope(max_position)
    
    def _init_rope(self, max_position):
        inv_freq = 1.0 / (self.rope_theta ** (
            torch.arange(0, self.head_dim, 2).float() / self.head_dim
        ))
        self.register_buffer('inv_freq', inv_freq)
        t = torch.arange(max_position).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, hidden_states, position_ids=None):
        batch_size, seq_len, _ = hidden_states.shape
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Single fused QKV projection
        qkv = self.qkv_proj(hidden_states)
        
        # Split and reshape
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        
        q = qkv[..., :q_dim].view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = qkv[..., q_dim:q_dim+kv_dim].view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = qkv[..., q_dim+kv_dim:].view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos = self.cos_cached[position_ids].unsqueeze(1)
        sin = self.sin_cached[position_ids].unsqueeze(1)
        
        x1_q, x2_q = q[..., :self.head_dim//2], q[..., self.head_dim//2:]
        x1_k, x2_k = k[..., :self.head_dim//2], k[..., self.head_dim//2:]
        
        q = torch.cat([x1_q * cos[..., :self.head_dim//2] - x2_q * sin[..., :self.head_dim//2],
                       x2_q * cos[..., self.head_dim//2:] + x1_q * sin[..., self.head_dim//2:]], dim=-1)
        k = torch.cat([x1_k * cos[..., :self.head_dim//2] - x2_k * sin[..., :self.head_dim//2],
                       x2_k * cos[..., self.head_dim//2:] + x1_k * sin[..., self.head_dim//2:]], dim=-1)
        
        return q, k, v


# Test parameters
batch_size = 8
seq_len = 2048
dim = 4096
num_heads = 32
num_kv_heads = 8  # GQA

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
    return [dim, num_heads, num_kv_heads]

