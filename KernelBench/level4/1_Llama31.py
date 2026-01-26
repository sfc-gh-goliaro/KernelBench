"""
Llama-3.1 Dense Decoder Model

A decoder-only transformer implementing Llama-3.1 architecture:
- RMSNorm normalization
- Grouped-Query Attention (GQA) with Paged KV Cache
- Rotary Position Embeddings (RoPE)
- SwiGLU MLP (gate/up projection fused)

Variants from Table 5:
- Llama-3.1-8B: hidden=4096, heads=32, kv_heads=8, layers=32
- Llama-3.1-70B: hidden=8192, heads=64, kv_heads=8, layers=80

This model uses level1 operators from KernelBench directly (no wrappers needed).
All operators are used directly without any custom implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, Tuple

# Import level1 operators - all used directly without wrappers
from ..level1.normalization._4_RMSNorm import Model as RMSNorm
from ..level1.embeddings._1_RotaryEmbedding import Model as RotaryEmbedding
from ..level1.activations._7_Swish import Model as Swish
from ..level1.matmul._10_Linear import Model as Linear
from ..level1.attention._3_GroupedQueryAttention import Model as GroupedQueryAttention


# ============================================================================
# Model Variants - configs loaded from HuggingFace
# ============================================================================

VARIANTS: Dict[str, str] = {
    "8B": "meta-llama/Llama-3.1-8B",
    "70B": "meta-llama/Llama-3.1-70B",
}


# ============================================================================
# Component Modules (using level1 operators directly)
# ============================================================================

class LlamaAttention(nn.Module):
    """
    Llama-style attention block using level1 operators.
    
    Uses:
    - Linear for Q/K/V/O projections
    - RotaryEmbedding for position encoding
    - GroupedQueryAttention for attention with paged KV cache
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 8192,
        rope_theta: float = 10000.0,
        block_size: int = 16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # Q/K/V/O projections using level1 Linear
        self.q_proj = Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = Linear(num_heads * head_dim, hidden_size, bias=False)

        # RoPE using level1 RotaryEmbedding with bhsd layout
        self.rotary_emb = RotaryEmbedding(head_dim, max_seq_len, rope_theta, layout="bhsd")
        
        # Attention using level1 GroupedQueryAttention with paged KV cache
        self.attn = GroupedQueryAttention(num_heads, num_kv_heads, head_dim, block_size)

    def forward(
        self, 
        x: torch.Tensor, 
        kv_cache_pool: torch.Tensor,
        block_table: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with paged KV cache.
        
        Args:
            x: Input tensor (batch_size, seq_len, hidden_size)
            kv_cache_pool: Paged KV cache (num_blocks, block_size, num_kv_heads, head_dim, 2)
            block_table: Block table (batch_size, max_blocks_per_seq)
            context_lens: Context lengths (batch_size,)
            
        Returns:
            Output tensor (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape

        # Q/K/V projections using level1 Linear
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE using level1 RotaryEmbedding
        q, k = self.rotary_emb(q, k)

        # Attention using level1 GroupedQueryAttention with paged KV cache
        attn_output = self.attn(q, k, v, kv_cache_pool, block_table, context_lens)

        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP using level1 operators."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)
        
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
        block_size: int = 16,
    ):
        super().__init__()
        # Use level1 RMSNorm operator directly with learnable weight and dim=-1 for (batch, seq, hidden)
        self.input_layernorm = RMSNorm(hidden_size, rms_norm_eps, learnable_weight=True, dim=-1)
        self.self_attn = LlamaAttention(
            hidden_size, num_heads, num_kv_heads, head_dim, max_seq_len, rope_theta, block_size
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, rms_norm_eps, learnable_weight=True, dim=-1)
        self.mlp = SwiGLUMLP(hidden_size, intermediate_size)

    def forward(
        self, 
        x: torch.Tensor, 
        kv_cache_pool: torch.Tensor,
        block_table: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        # Self-attention with residual
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, kv_cache_pool, block_table, context_lens)
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
    Llama-3.1 style decoder-only transformer with Paged KV Cache.
    
    Uses level1 operators from KernelBench directly (no wrappers):
    - RMSNorm (with learnable_weight=True, dim=-1)
    - RotaryEmbedding (with layout="bhsd")
    - GroupedQueryAttention (with paged KV cache)
    - Linear
    - Swish
    
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
        block_size: int = 16,
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
        self.block_size = block_size
        
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
                block_size=block_size,
            )
            for _ in range(num_layers)
        ])
        
        # Final normalization using level1 RMSNorm directly
        self.norm = RMSNorm(hidden_size, rms_norm_eps, learnable_weight=True, dim=-1)
        self.lm_head = Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self, 
        input_ids: torch.Tensor, 
        kv_cache_pool: torch.Tensor,
        block_table: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with paged KV cache.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            kv_cache_pool: Paged KV cache (num_blocks, block_size, num_kv_heads, head_dim, 2)
            block_table: Block table (batch_size, max_blocks_per_seq)
            context_lens: Context lengths (batch_size,)
            
        Returns:
            logits: Output logits (batch_size, seq_len, vocab_size)
        """
        x = self.embed_tokens(input_ids)

        for layer in self.layers:
            x = layer(x, kv_cache_pool, block_table, context_lens)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


# ============================================================================
# Benchmark Configuration
# ============================================================================

# Default reduced config for benchmarking
batch_size = 2
sequence_length = 512
context_length = 1024  # Tokens already in KV cache
vocab_size = 128256
hidden_size = 4096
num_layers = 8  # Reduced
num_heads = 32
num_kv_heads = 8
head_dim = 128
intermediate_size = 14336
block_size = 16
max_blocks_per_seq = (context_length + sequence_length) // block_size + 1
num_blocks = batch_size * max_blocks_per_seq + 100  # Extra blocks for safety


def get_inputs():
    """Get benchmark inputs including paged KV cache."""
    input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
    kv_cache_pool = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, 2)
    block_table = torch.randint(0, num_blocks, (batch_size, max_blocks_per_seq))
    context_lens = torch.full((batch_size,), context_length, dtype=torch.long)
    return [input_ids, kv_cache_pool, block_table, context_lens]


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
        'block_size': block_size,
    }]
