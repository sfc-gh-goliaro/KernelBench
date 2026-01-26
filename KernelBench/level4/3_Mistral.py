"""
Mistral Dense Decoder Model

A decoder-only transformer implementing Mistral architecture:
- Grouped-Query Attention (GQA)
- Sliding Window Attention
- Rotary Position Embeddings (RoPE)
- SwiGLU MLP

Variants from Table 5:
- Mistral-7B-v0.3: hidden=4096, heads=32, kv_heads=8, layers=32, sliding_window=4096
- Mistral-Nemo-12B: hidden=5120, heads=32, kv_heads=8, layers=40

This model uses level1 operators from KernelBench.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any

# Import level1 operators
# Operators that need wrapping (different interface in level4)
from ..level1.normalization._4_RMSNorm import Model as RMSNormL1
from ..level1.embeddings._1_RotaryEmbedding import Model as RotaryEmbeddingL1
# Operators used directly (no wrapping needed)
from ..level1.activations._7_Swish import Model as Swish
from ..level1.attention._7_SlidingWindowAttention import Model as SlidingWindowAttn
from ..level1.matmul._1_MatMul import Model as MatMul


# ============================================================================
# Model Variants - configs loaded from HuggingFace
# ============================================================================

VARIANTS: Dict[str, str] = {
    "7B": "mistralai/Mistral-7B-v0.3",
    "Nemo-12B": "mistralai/Mistral-Nemo-12B",
}


# ============================================================================
# Wrapper classes for level1 operators that need adaptation
# ============================================================================

class RMSNorm(nn.Module):
    """RMS Normalization with learnable weight, using level1 operator.
    
    Wrapping needed because:
    - Level1 RMSNorm has no learnable weight parameter
    - Level1 expects (batch, features, *) but we use (batch, seq, hidden)
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self._rms_norm = RMSNormL1(hidden_size, eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self._rms_norm(x.transpose(1, -1)).transpose(1, -1)
        return normalized * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding using level1 operator.
    
    Wrapping needed because:
    - Level1 expects (batch, seq, heads, head_dim)
    - Level4 uses (batch, heads, seq, head_dim)
    """
    def __init__(self, head_dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self._rope = RotaryEmbeddingL1(head_dim, max_seq_len, base)
        self.head_dim = head_dim

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple:
        q_reshaped = q.transpose(1, 2)
        k_reshaped = k.transpose(1, 2)
        q_rotated, k_rotated = self._rope(q_reshaped, k_reshaped)
        return q_rotated.transpose(1, 2), k_rotated.transpose(1, 2)


# ============================================================================
# Component Modules (using level1 operators)
# ============================================================================

class SlidingWindowAttention(nn.Module):
    """Grouped-Query Attention with optional Sliding Window using level1 operators."""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 8192,
        sliding_window: Optional[int] = None,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.sliding_window = sliding_window

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # Use level1 operators
        self.rotary_emb = RotaryEmbedding(head_dim, max_seq_len, rope_theta)
        self.matmul = MatMul()

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE using level1 operator
        q, k = self.rotary_emb(q, k)

        # Expand KV heads for GQA
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Attention scores using level1 matmul
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = self.matmul(q, k.transpose(-2, -1)) * scale

        # Create causal + sliding window mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        
        if self.sliding_window is not None:
            window_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=x.device),
                diagonal=-self.sliding_window
            )
            causal_mask = causal_mask + window_mask
        
        attn_weights = attn_weights.masked_fill(causal_mask.bool(), float('-inf'))

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = self.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP using level1 operators."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        # Use level1 Swish operator directly
        self.swish = Swish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.swish(self.gate_proj(x)) * self.up_proj(x))


class MistralDecoderLayer(nn.Module):
    """Single Mistral decoder layer using level1 operators."""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        max_seq_len: int = 8192,
        sliding_window: Optional[int] = None,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        # Use level1 RMSNorm
        self.input_layernorm = RMSNorm(hidden_size, rms_norm_eps)
        self.self_attn = SlidingWindowAttention(
            hidden_size, num_heads, num_kv_heads, head_dim,
            max_seq_len, sliding_window, rope_theta
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, rms_norm_eps)
        self.mlp = SwiGLUMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, attention_mask)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    Mistral decoder-only transformer with sliding window attention.
    
    Uses level1 operators from KernelBench:
    - RMSNorm from level1/normalization/4_RMSNorm
    - RotaryEmbedding (wrapped - handles shape transpose)
    - Swish (used directly)
    - MatMul (used directly)
    
    For HuggingFace integration (weight loading, validation), use MistralAdapter
    from hf_adapters.py.
    """
    
    def __init__(
        self,
        vocab_size: int = 32768,
        hidden_size: int = 4096,
        num_layers: int = 32,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        intermediate_size: int = 14336,
        max_seq_len: int = 8192,
        sliding_window: int = 4096,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-5,
        **kwargs  # Accept and ignore extra kwargs for flexibility
    ):
        super().__init__()
        
        # Store config values
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.max_seq_len = max_seq_len
        self.sliding_window = sliding_window
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        self.layers = nn.ModuleList([
            MistralDecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                max_seq_len=max_seq_len,
                sliding_window=sliding_window,
                rope_theta=rope_theta,
                rms_norm_eps=rms_norm_eps,
            )
            for _ in range(num_layers)
        ])
        
        # Use level1 RMSNorm
        self.norm = RMSNorm(hidden_size, rms_norm_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embed_tokens(input_ids)

        for layer in self.layers:
            x = layer(x, attention_mask)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 2
sequence_length = 512
vocab_size = 32768
hidden_size = 4096
num_layers = 8
num_heads = 32
num_kv_heads = 8
head_dim = 128
intermediate_size = 14336


def get_inputs():
    return [torch.randint(0, vocab_size, (batch_size, sequence_length))]


def get_init_inputs():
    return [{
        'vocab_size': vocab_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'num_kv_heads': num_kv_heads,
        'head_dim': head_dim,
        'intermediate_size': intermediate_size,
        'sliding_window': 4096,
    }]
