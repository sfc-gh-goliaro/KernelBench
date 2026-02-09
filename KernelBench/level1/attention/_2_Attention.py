"""
Scaled Dot-Product Attention + Multi-Head Attention with Paged KV Cache

This file provides two operators:

1. ScaledDotProductAttention (Model)
   Parameter-free attention math supporting MHA, GQA, MQA, cross-attention,
   additive bias (ALiBi, T5 relative position bias), causal masking, and
   configurable scaling. Does NOT own any weights or KV cache.
   Used by: all models (Llama, Falcon, BLOOM, Mixtral, T5, etc.)

2. MultiHeadAttention
   Attention with paged KV cache for autoregressive generation.
   Composes PagedKVCache + ScaledDotProductAttention.
   Handles MHA (num_kv_heads == num_heads), GQA (num_kv_heads < num_heads),
   and MQA (num_kv_heads == 1) transparently via the num_kv_heads parameter.
   Used by: Llama, Falcon, Mixtral, and any decoder-only model with paged KV cache.

Input shapes (BHSD layout):
    q: (batch_size, num_heads, seq_len_q, head_dim)
    k: (batch_size, num_kv_heads, seq_len_k, head_dim)
    v: (batch_size, num_kv_heads, seq_len_k, head_dim)
    attn_mask: Optional float mask added to scores
    attn_bias: Optional additive bias (ALiBi, T5 relative position bias)

Output:
    (batch_size, num_heads, seq_len_q, head_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from ._1_PagedKVCache import Model as PagedKVCache, AttentionMetadata


# ============================================================================
# ScaledDotProductAttention — parameter-free attention math
# ============================================================================

class ScaledDotProductAttention(nn.Module):
    """
    Generic Scaled Dot-Product Attention (parameter-free).

    Handles GQA/MQA by repeating K/V heads to match Q heads before calling
    F.scaled_dot_product_attention. Supports optional additive bias
    (ALiBi, T5 relative position bias) and attention masks.
    """

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        scale: Optional[float] = None,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute scaled dot-product attention.

        Args:
            q: Query tensor (batch, num_heads, seq_q, head_dim)
            k: Key tensor (batch, num_kv_heads, seq_k, head_dim)
            v: Value tensor (batch, num_kv_heads, seq_k, head_dim)
            attn_mask: Optional float mask added to attention scores before softmax.
                       Shape: broadcastable to (batch, num_heads, seq_q, seq_k).
                       Use 0 for positions to attend, -inf for positions to mask.
            attn_bias: Optional additive bias added to attention scores (e.g. ALiBi,
                       T5 relative position bias). Shape: broadcastable to
                       (batch, num_heads, seq_q, seq_k).
            is_causal: If True, apply causal (lower-triangular) masking.
                       Cannot be used together with attn_mask.
            scale: Scaling factor for attention scores. If None, uses 1/sqrt(head_dim).
            dropout_p: Dropout probability for attention weights.

        Returns:
            Attention output (batch, num_heads, seq_q, head_dim)
        """
        enable_gqa = k.shape[1] != q.shape[1]

        # Combine attn_mask and attn_bias into a single mask for SDPA
        combined_mask = None
        if attn_bias is not None and attn_mask is not None:
            combined_mask = attn_bias + attn_mask
        elif attn_bias is not None:
            combined_mask = attn_bias
        elif attn_mask is not None:
            combined_mask = attn_mask

        # When we have a combined_mask, we cannot use is_causal=True with SDPA
        # (SDPA doesn't allow both attn_mask and is_causal). In that case,
        # we fold the causal mask into combined_mask.
        use_causal = is_causal and combined_mask is None

        if is_causal and combined_mask is not None:
            # Build explicit causal mask and combine
            seq_q = q.shape[2]
            seq_k = k.shape[2]
            device = q.device
            dtype = q.dtype
            causal = torch.triu(
                torch.full((seq_q, seq_k), float('-inf'), device=device, dtype=dtype),
                diagonal=seq_k - seq_q + 1,
            )
            combined_mask = combined_mask + causal

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=combined_mask,
            dropout_p=dropout_p if self.training else 0.0,
            is_causal=use_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
        return out


# ============================================================================
# MultiHeadAttention — unified MHA/GQA/MQA with Paged KV Cache
# ============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with Paged KV Cache.

    Handles MHA, GQA, and MQA transparently via num_kv_heads:
    - MHA:  num_kv_heads == num_heads  (standard multi-head attention)
    - GQA:  num_kv_heads < num_heads   (grouped-query attention, e.g. Llama-3)
    - MQA:  num_kv_heads == 1          (multi-query attention, e.g. Falcon-7B)

    Composes PagedKVCache + ScaledDotProductAttention.

    NOTE: Q/K/V projections are done externally by the level4 model.
    This operator takes pre-projected Q, K, V tensors.
    """

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int,
                 block_size: int = 16, num_blocks: int = 1024,
                 max_seq_len: int = 8192, dropout: float = 0.0):
        """
        Args:
            num_heads: Number of query heads
            num_kv_heads: Number of key/value heads (1 for MQA, num_heads for MHA,
                          or any divisor of num_heads for GQA)
            head_dim: Dimension of each attention head
            block_size: Number of tokens per cache block/page
            num_blocks: Total number of blocks in the cache pool
            max_seq_len: Maximum sequence length (informational)
            dropout: Attention dropout probability
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dropout = dropout

        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Paged KV cache
        self.kv_cache = PagedKVCache(
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            num_blocks=num_blocks,
        )

        # Attention math
        self.attn = ScaledDotProductAttention()

    def reset_cache(self):
        """Reset the KV cache to zeros."""
        self.kv_cache.reset()

    def _build_decode_mask(
        self,
        q: torch.Tensor,
        k_full: torch.Tensor,
        context_lens: torch.Tensor,
        max_context_len: int,
    ) -> torch.Tensor:
        """Build combined causal + padding mask for decode phase."""
        batch_size = q.shape[0]
        seq_len = q.shape[2]
        total_len = k_full.shape[2]
        device = q.device
        dtype = q.dtype

        # Query positions (relative to start of sequence)
        query_positions = (
            torch.arange(seq_len, device=device).unsqueeze(0)
            + context_lens.unsqueeze(1)
        )
        key_positions = torch.arange(total_len, device=device).unsqueeze(0)

        # Causal mask
        causal_mask = key_positions > query_positions.unsqueeze(-1)

        # Padding mask for cached K/V
        if max_context_len > 0:
            cache_positions = torch.arange(max_context_len, device=device).unsqueeze(0)
            padding_mask = cache_positions >= context_lens.unsqueeze(1)
            padding_mask = torch.cat([
                padding_mask,
                torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
            ], dim=1)
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
        else:
            padding_mask = torch.zeros(
                batch_size, 1, 1, total_len, dtype=torch.bool, device=device
            )

        # Combine
        causal_mask = causal_mask.unsqueeze(1)
        bool_mask = causal_mask | padding_mask

        attn_mask = torch.zeros(
            batch_size, 1, seq_len, total_len, dtype=dtype, device=device
        )
        attn_mask = attn_mask.masked_fill(bool_mask, float('-inf'))
        return attn_mask

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                attn_metadata: AttentionMetadata) -> torch.Tensor:
        """
        Forward pass with paged KV cache.

        Args:
            q: Projected query (batch_size, num_heads, seq_len, head_dim)
            k: Projected key for new tokens (batch_size, num_kv_heads, seq_len, head_dim)
            v: Projected value for new tokens (batch_size, num_kv_heads, seq_len, head_dim)
            attn_metadata: Attention metadata containing slot_mapping, block_table, etc.

        Returns:
            Output tensor (batch_size, num_heads, seq_len, head_dim)
        """
        # Write new K/V to cache and gather full K/V
        k_full, v_full = self.kv_cache(k, v, attn_metadata)

        if attn_metadata.is_prefill:
            # Prefill: all tokens are new, use causal masking
            return self.attn(
                q, k_full, v_full,
                is_causal=True,
                scale=self.scale,
                dropout_p=self.dropout,
            )
        else:
            # Decode: need explicit mask for variable context lengths
            max_context_len = attn_metadata.context_lens.max().item()
            attn_mask = self._build_decode_mask(
                q, k_full, attn_metadata.context_lens, max_context_len
            )
            return self.attn(
                q, k_full, v_full,
                attn_mask=attn_mask,
                scale=self.scale,
                dropout_p=self.dropout,
            )
