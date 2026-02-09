"""
Attention with Linear Biases (ALiBi) with Paged KV Cache

Used by: BLOOM, MPT

ALiBi adds a linear bias to attention scores based on position.
This provides position information without explicit position embeddings
and extrapolates well to longer sequences.

This implementation composes:
- PagedKVCache (_1_PagedKVCache) for cache management
- ScaledDotProductAttention (_2_Attention) for attention math
  with ALiBi slopes passed as attn_bias

This implementation matches HuggingFace BLOOM exactly for numerical consistency.
BLOOM uses manual matmul + softmax (not F.scaled_dot_product_attention) for its
attention computation, so we replicate that pattern here via baddbmm.

NOTE: Q/K/V projections are done separately using Linear operators.
This operator takes pre-projected Q, K, V tensors.

Shapes:
    q: (batch_size, num_heads, seq_len, head_dim) - projected query
    k: (batch_size, num_heads, seq_len, head_dim) - projected key (new tokens)
    v: (batch_size, num_heads, seq_len, head_dim) - projected value (new tokens)
    Output: (batch_size, num_heads, seq_len, head_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from ._1_PagedKVCache import (
    Model as PagedKVCache,
    AttentionMetadata,
    create_attention_metadata,
)



def get_alibi_slopes(num_heads: int) -> torch.Tensor:
    """
    Compute ALiBi slopes for each head.
    Matches HuggingFace BLOOM's implementation exactly.
    """
    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]

    if math.log2(num_heads).is_integer():
        slopes = get_slopes_power_of_2(num_heads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        slopes = get_slopes_power_of_2(closest_power_of_2)
        slopes = slopes + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:num_heads - closest_power_of_2]

    return torch.tensor(slopes, dtype=torch.float32)


class Model(nn.Module):
    """
    Attention with Linear Biases (ALiBi) with Paged KV Cache.

    Composes PagedKVCache for cache management.
    Uses manual baddbmm + softmax to match HuggingFace BLOOM exactly.
    """

    def __init__(self, num_heads: int, head_dim: int, block_size: int = 16,
                 num_blocks: int = 1024):
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Compute ALiBi slopes for each head (matching HuggingFace)
        slopes = get_alibi_slopes(num_heads)
        self.register_buffer('alibi_slopes', slopes)

        # Paged KV cache
        self.kv_cache = PagedKVCache(
            num_kv_heads=num_heads,
            head_dim=head_dim,
            block_size=block_size,
            num_blocks=num_blocks,
        )

    def reset_cache(self):
        """Reset the KV cache."""
        self.kv_cache.reset()

    def _build_alibi_tensor(self, seq_len: int, batch_size: int,
                             device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Build ALiBi tensor matching HuggingFace's implementation.

        Returns:
            alibi: (batch_size * num_heads, 1, seq_len) - added to attention scores
        """
        arange_tensor = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(0)
        alibi = self.alibi_slopes.to(device=device, dtype=dtype).unsqueeze(1) * arange_tensor
        alibi = alibi.unsqueeze(0).expand(batch_size, -1, -1)
        alibi = alibi.reshape(batch_size * self.num_heads, 1, seq_len)
        return alibi

    def _prefill_attention(self, q: torch.Tensor, k: torch.Tensor,
                           v: torch.Tensor) -> torch.Tensor:
        """Compute attention during prefill phase (matches HF BLOOM exactly)."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        device = q.device
        dtype = q.dtype

        q_reshaped = q.reshape(batch_size * num_heads, seq_len, head_dim)
        k_reshaped = k.reshape(batch_size * num_heads, seq_len, head_dim).transpose(-1, -2)
        v_reshaped = v.reshape(batch_size * num_heads, seq_len, head_dim)

        alibi = self._build_alibi_tensor(seq_len, batch_size, device, dtype)

        scores = torch.baddbmm(
            alibi, q_reshaped, k_reshaped,
            beta=1.0, alpha=self.scale,
        )

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))

        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype)
        attn_output = torch.bmm(attn_weights, v_reshaped)

        attn_output = attn_output.view(batch_size, num_heads, seq_len, head_dim)
        return attn_output

    def _decode_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          block_table: torch.Tensor,
                          context_lens: torch.Tensor) -> torch.Tensor:
        """Compute attention during decode phase (matches HF BLOOM exactly)."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        device = q.device
        dtype = q.dtype

        max_context_len = context_lens.max().item()

        if max_context_len > 0:
            k_cache, v_cache = self.kv_cache.gather(block_table, context_lens)
            k_full = torch.cat([k_cache, k], dim=2)
            v_full = torch.cat([v_cache, v], dim=2)
        else:
            k_full = k
            v_full = v

        total_len = k_full.shape[2]

        q_reshaped = q.reshape(batch_size * num_heads, seq_len, head_dim)
        k_reshaped = k_full.reshape(batch_size * num_heads, total_len, head_dim).transpose(-1, -2)
        v_reshaped = v_full.reshape(batch_size * num_heads, total_len, head_dim)

        alibi = self._build_alibi_tensor(total_len, batch_size, device, dtype)

        scores = torch.baddbmm(
            alibi, q_reshaped, k_reshaped,
            beta=1.0, alpha=self.scale,
        )

        scores = scores.view(batch_size, num_heads, seq_len, total_len)

        query_positions = context_lens.view(batch_size, 1, 1) + torch.arange(seq_len, device=device).view(1, seq_len, 1)
        key_positions = torch.arange(total_len, device=device).view(1, 1, total_len)
        causal_mask = key_positions > query_positions

        if max_context_len > 0:
            cache_positions = torch.arange(max_context_len, device=device).unsqueeze(0)
            padding_mask = cache_positions >= context_lens.unsqueeze(1)
            padding_mask = torch.cat([
                padding_mask,
                torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
            ], dim=1)
        else:
            padding_mask = torch.zeros(batch_size, total_len, dtype=torch.bool, device=device)

        attn_mask = causal_mask | padding_mask.unsqueeze(1)
        attn_mask = attn_mask.unsqueeze(1)
        scores = scores.masked_fill(attn_mask, float('-inf'))

        scores = scores.view(batch_size * num_heads, seq_len, total_len)
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype)
        attn_output = torch.bmm(attn_weights, v_reshaped)

        attn_output = attn_output.view(batch_size, num_heads, seq_len, head_dim)
        return attn_output

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                attn_metadata: AttentionMetadata) -> torch.Tensor:
        """
        Forward pass with paged KV cache and ALiBi attention.

        Args:
            q: Projected query (batch_size, num_heads, seq_len, head_dim)
            k: Projected key for new tokens (batch_size, num_heads, seq_len, head_dim)
            v: Projected value for new tokens (batch_size, num_heads, seq_len, head_dim)
            attn_metadata: Attention metadata containing slot_mapping, block_table, etc.

        Returns:
            Output tensor (batch_size, num_heads, seq_len, head_dim)
        """
        # Write new K,V to cache
        self.kv_cache.write(k, v, attn_metadata.slot_mapping)

        # Choose prefill or decode path
        if attn_metadata.is_prefill:
            return self._prefill_attention(q, k, v)
        else:
            return self._decode_attention(
                q, k, v,
                attn_metadata.block_table,
                attn_metadata.context_lens
            )
