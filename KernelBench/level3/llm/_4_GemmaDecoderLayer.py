import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GemmaRMSNorm(nn.Module):
    """Gemma-style RMSNorm with +1 offset on weights."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))  # Note: zeros, then +1 in forward
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * (1 + self.weight)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding."""
    def __init__(self, head_dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())

    def forward(self, seq_len: int):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    q_rotated = torch.stack([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1).flatten(-2)
    k_rotated = torch.stack([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1).flatten(-2)
    return q_rotated, k_rotated


class GemmaAttention(nn.Module):
    """Gemma attention with optional sliding window and soft-capping."""
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, 
                 head_dim: int, max_seq_len: int, use_sliding_window: bool = False,
                 sliding_window_size: int = 4096, attn_logit_softcap: float = 50.0):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window_size = sliding_window_size
        self.attn_logit_softcap = attn_logit_softcap
        
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(head_dim, max_seq_len)
        self.scale = head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary_emb(seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Soft-capping (Gemma-2 feature)
        attn = self.attn_logit_softcap * torch.tanh(attn / self.attn_logit_softcap)
        
        # Causal mask (with optional sliding window)
        if self.use_sliding_window:
            mask = torch.ones(seq_len, seq_len, device=x.device)
            mask = torch.triu(mask, diagonal=1)  # Causal
            mask += torch.tril(torch.ones_like(mask), diagonal=-self.sliding_window_size)  # Sliding window
            mask = mask.bool()
        else:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(out)


class GemmaMLP(nn.Module):
    """Gemma-style GeGLU MLP."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GeGLU: GELU(gate) * up
        return self.down_proj(F.gelu(self.gate_proj(x), approximate='tanh') * self.up_proj(x))


class Model(nn.Module):
    """
    Gemma Decoder Layer
    
    The core repeated block in Gemma-style transformers.
    Used by: Gemma, Gemma-2, Gemma-3
    
    Key differences from Llama:
    - 4 RMSNorms per layer (pre/post attention, pre/post FFN)
    - GeGLU activation instead of SwiGLU
    - Attention logit soft-capping
    - Alternating sliding window attention (Gemma-2)
    - Weight offset in RMSNorm (+1)
    """
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, 
                 intermediate_size: int, max_seq_len: int = 8192,
                 use_sliding_window: bool = False):
        super().__init__()
        self.head_dim = hidden_size // num_heads
        
        # Pre-attention norm
        self.input_layernorm = GemmaRMSNorm(hidden_size)
        self.self_attn = GemmaAttention(
            hidden_size, num_heads, num_kv_heads, self.head_dim, 
            max_seq_len, use_sliding_window
        )
        # Post-attention norm (Gemma-2 addition)
        self.post_attention_layernorm = GemmaRMSNorm(hidden_size)
        
        # Pre-FFN norm
        self.pre_feedforward_layernorm = GemmaRMSNorm(hidden_size)
        self.mlp = GemmaMLP(hidden_size, intermediate_size)
        # Post-FFN norm (Gemma-2 addition)
        self.post_feedforward_layernorm = GemmaRMSNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention block with pre/post norms
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = self.post_attention_layernorm(x)
        x = residual + x
        
        # FFN block with pre/post norms
        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        x = residual + x
        
        return x


# Benchmark configuration (Gemma-2-9B dimensions)
batch_size = 8
seq_len = 2048
hidden_size = 3584
num_heads = 16
num_kv_heads = 8
intermediate_size = 14336

def get_inputs():
    return [torch.randn(batch_size, seq_len, hidden_size)]

def get_init_inputs():
    return [hidden_size, num_heads, num_kv_heads, intermediate_size]

