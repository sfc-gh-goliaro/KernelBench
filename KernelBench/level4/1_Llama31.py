"""
Llama-3.1 Dense Decoder Model

A decoder-only transformer implementing Llama-3.1 architecture:
- RMSNorm normalization
- Grouped-Query Attention (GQA)
- Rotary Position Embeddings (RoPE)
- SwiGLU MLP (gate/up projection fused)

Variants from Table 5:
- Llama-3.1-8B: hidden=4096, heads=32, kv_heads=8, layers=32
- Llama-3.1-70B: hidden=8192, heads=64, kv_heads=8, layers=80

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
from ..level1.matmul._1_MatMul import Model as MatMul


# ============================================================================
# Model Variants - configs loaded from HuggingFace
# ============================================================================

VARIANTS: Dict[str, str] = {
    "8B": "meta-llama/Llama-3.1-8B",
    "70B": "meta-llama/Llama-3.1-70B",
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
        # Apply RMSNorm from level1 (transpose to match expected shape)
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
        """Apply RoPE to q and k tensors."""
        # q, k: (batch, heads, seq, head_dim)
        # Reshape for level1 RoPE: (batch, seq, heads, head_dim)
        q_reshaped = q.transpose(1, 2)
        k_reshaped = k.transpose(1, 2)
        
        q_rotated, k_rotated = self._rope(q_reshaped, k_reshaped)
        
        # Reshape back: (batch, heads, seq, head_dim)
        return q_rotated.transpose(1, 2), k_rotated.transpose(1, 2)


# ============================================================================
# Component Modules (using level1 operators)
# ============================================================================

class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA) using level1 operators."""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 8192,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # Use level1 RoPE operator (wrapped for shape adaptation)
        self.rotary_emb = RotaryEmbedding(head_dim, max_seq_len, rope_theta)
        
        # Use level1 MatMul operator directly
        self.matmul = MatMul()

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = self.rotary_emb(q, k)

        # Expand KV heads for GQA
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Scaled dot-product attention using level1 MatMul
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = self.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

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
        gate = self.swish(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class LlamaDecoderLayer(nn.Module):
    """Single Llama decoder layer using level1 operators."""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        max_seq_len: int = 8192,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        # Use level1 RMSNorm operator
        self.input_layernorm = RMSNorm(hidden_size, rms_norm_eps)
        self.self_attn = GroupedQueryAttention(
            hidden_size, num_heads, num_kv_heads, head_dim, max_seq_len, rope_theta
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, rms_norm_eps)
        self.mlp = SwiGLUMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, attention_mask)
        x = residual + x

        # MLP with residual
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
    Llama-3.1 style decoder-only transformer.
    
    Uses level1 operators from KernelBench:
    - RMSNorm (wrapped - adds learnable weight, handles shape)
    - RotaryEmbedding (wrapped - handles shape transpose)
    - Swish (used directly)
    - MatMul (used directly)
    
    For HuggingFace integration (weight loading, validation), use LlamaAdapter
    from hf_adapters.py.
    """
    
    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_size: int = 4096,
        num_layers: int = 32,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: Optional[int] = None,
        intermediate_size: int = 14336,
        max_seq_len: int = 8192,
        rope_theta: float = 500000.0,
        rms_norm_eps: float = 1e-5,
        **kwargs  # Accept and ignore extra kwargs for flexibility
    ):
        super().__init__()
        
        # Store config values
        if head_dim is None:
            head_dim = hidden_size // num_heads
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        
        # Token embedding
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        # Decoder layers using level1 operators
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                max_seq_len=max_seq_len,
                rope_theta=rope_theta,
                rms_norm_eps=rms_norm_eps,
            )
            for _ in range(num_layers)
        ])
        
        # Final normalization using level1 RMSNorm
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

# Default reduced config for benchmarking
batch_size = 2
sequence_length = 512
vocab_size = 128256
hidden_size = 4096
num_layers = 8  # Reduced
num_heads = 32
num_kv_heads = 8
head_dim = 128
intermediate_size = 14336


def get_inputs():
    """Get benchmark inputs."""
    return [torch.randint(0, vocab_size, (batch_size, sequence_length))]


def get_init_inputs():
    """Get model initialization inputs."""
    return [{
        'vocab_size': vocab_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'num_kv_heads': num_kv_heads,
        'head_dim': head_dim,
        'intermediate_size': intermediate_size,
    }]
