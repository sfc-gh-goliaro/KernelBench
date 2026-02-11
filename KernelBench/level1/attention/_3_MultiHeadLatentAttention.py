"""
Multi-Head Latent Attention (MLA)

Used by: DeepSeek-V2, DeepSeek-V2-Lite, DeepSeek-V3

MLA compresses KV into a low-rank latent space before caching,
reducing KV cache memory while maintaining model quality.

Key insight: Instead of caching full K,V tensors, we cache the compressed
latent representation. The K,V are recomputed on-the-fly via up-projections.

Architecture:
    - Q projection (either direct or low-rank via q_lora_rank)
    - KV compression (kv_a_proj_with_mqa) to latent space
    - KV up-projection (kv_b_proj) to full K_nope and V
    - RoPE applied to rope portions of Q and K (complex-polar mode)
    - SDPA for attention computation
    - Output projection

Shapes:
    Input: (batch_size, seq_len, hidden_size)
    Output: (batch_size, seq_len, hidden_size)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from ._1_PagedKVCache import AttentionMetadata, create_attention_metadata
from ._2_Attention import ScaledDotProductAttention
from ..normalization._4_RMSNorm import Model as RMSNorm
from ..embeddings._1_RotaryEmbedding import Model as RotaryEmbedding


class Model(nn.Module):
    """
    Multi-Head Latent Attention (MLA).

    Complete attention module with:
    - Q projection (with optional low-rank)
    - KV compression and up-projection
    - RoPE (complex-polar mode with YARN scaling support)
    - ScaledDotProductAttention for attention computation
    - Simple KV caching for decode phase
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        kv_lora_rank: int,
        q_lora_rank: Optional[int] = None,
        max_seq_len: int = 163840,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
        block_size: int = 16,
        num_blocks: int = 1024,
        layer_idx: int = 0,
    ):
        """
        Initialize Multi-Head Latent Attention.

        Args:
            hidden_size: Model hidden dimension
            num_heads: Number of attention heads
            qk_nope_head_dim: Non-RoPE dimension for Q/K
            qk_rope_head_dim: RoPE dimension for Q/K
            v_head_dim: Value head dimension
            kv_lora_rank: Rank for KV compression (latent dimension)
            q_lora_rank: Optional rank for Q low-rank projection
            max_seq_len: Maximum sequence length
            rope_theta: RoPE theta parameter
            rope_scaling: Optional YARN scaling config
            block_size: Block size for paged cache (unused currently)
            num_blocks: Number of cache blocks (unused currently)
            layer_idx: Layer index (for debugging)
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.layer_idx = layer_idx
        self.scaling = self.qk_head_dim ** (-0.5)

        # Q projection (either direct or low-rank)
        if q_lora_rank is None:
            self.q_proj = nn.Linear(hidden_size, num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
            self.q_a_layernorm = RMSNorm(q_lora_rank, eps=1e-6, learnable_weight=True, dim=-1)
            self.q_b_proj = nn.Linear(q_lora_rank, num_heads * self.qk_head_dim, bias=False)

        # KV projection with MQA-style fusion (compressed + rope)
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size,
            kv_lora_rank + qk_rope_head_dim,
            bias=False
        )
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=1e-6, learnable_weight=True, dim=-1)

        # KV up-projection: from compressed to full K_nope and V
        self.kv_b_proj = nn.Linear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False
        )

        # Output projection
        self.o_proj = nn.Linear(num_heads * v_head_dim, hidden_size, bias=False)

        # RoPE with complex-polar mode (matches HuggingFace DeepSeek-V2)
        self.rotary_emb = RotaryEmbedding(
            head_dim=qk_rope_head_dim,
            max_seq_len=max_seq_len,
            base=rope_theta,
            layout="bhsd",
            rope_scaling=rope_scaling,
            mode="complex",
        )

        # Attention math (parameter-free)
        self.sdpa = ScaledDotProductAttention()

        # Simple KV cache for decode phase
        self.k_cache = None
        self.v_cache = None

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """
        Forward pass with KV caching support.

        Args:
            x: Input tensor (batch_size, seq_len, hidden_size)
            position_ids: Position indices (batch_size, seq_len)
            attn_metadata: Attention metadata

        Returns:
            Output tensor (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape

        # Q projection (with optional low-rank)
        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))

        q = q.view(batch_size, seq_len, self.num_heads, self.qk_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # KV projection
        compressed_kv = self.kv_a_proj_with_mqa(x)
        k_nope_compressed, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        # Up-project compressed KV to get k_nope and v
        kv_proj = self.kv_b_proj(self.kv_a_layernorm(k_nope_compressed))
        kv_proj = kv_proj.view(batch_size, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, value_states = torch.split(kv_proj, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Reshape k_pe for RoPE: (batch, 1, seq, rope_dim) — single KV head
        k_pe = k_pe.view(batch_size, 1, seq_len, self.qk_rope_head_dim)

        # Apply RoPE using complex-polar multiplication
        q_pe, k_pe = self.rotary_emb(q_pe, k_pe, position_ids)

        # Expand k_pe to all heads
        k_pe = k_pe.expand(*k_nope.shape[:-1], -1)

        # Concatenate nope and pe components
        query_states = torch.cat((q_nope, q_pe), dim=-1)
        key_states = torch.cat((k_nope, k_pe), dim=-1)

        # Handle KV caching for decode phase
        if attn_metadata.is_prefill:
            self.k_cache = key_states
            self.v_cache = value_states
        else:
            if self.k_cache is not None:
                key_states = torch.cat([self.k_cache, key_states], dim=2)
                value_states = torch.cat([self.v_cache, value_states], dim=2)
            self.k_cache = key_states
            self.v_cache = value_states

        # Scaled dot-product attention
        attn_output = self.sdpa(
            query_states,
            key_states,
            value_states,
            scale=self.scaling,
            is_causal=attn_metadata.is_prefill,
        )

        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)

    def reset_cache(self):
        """Reset this layer's KV cache."""
        self.k_cache = None
        self.v_cache = None
