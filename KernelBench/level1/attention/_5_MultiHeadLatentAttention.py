"""
Multi-Head Latent Attention (MLA) with Paged KV Cache

Used by: DeepSeek-V2, DeepSeek-V2-Lite, DeepSeek-V3

MLA compresses KV into a low-rank latent space before caching,
reducing KV cache memory while maintaining model quality.
Uses paged cache where the compressed latent is stored in blocks.

Key insight: Instead of caching full K,V tensors, we cache the compressed
latent representation. The K,V are recomputed on-the-fly via up-projections.

This implementation owns its latent cache (per-layer cache pattern from vLLM).

Shapes:
    q: (batch_size, num_heads, seq_len, qk_head_dim) - projected query
       where qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    kv_c: (batch_size, seq_len, kv_lora_rank) - compressed KV latent (new tokens)
    k_pe: (batch_size, 1, seq_len, qk_rope_head_dim) - rope-encoded K component
    Output: (batch_size, num_heads, seq_len, v_head_dim)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


# ============================================================================
# Attention Metadata (vLLM-style)
# ============================================================================

@dataclass
class AttentionMetadata:
    """
    Metadata for paged attention operations.
    
    Aligned with vLLM's CommonAttentionMetadata pattern.
    """
    slot_mapping: torch.Tensor      # (num_tokens,) - where to write new cache entries
    block_table: torch.Tensor       # (batch_size, max_blocks_per_seq) - block assignments
    context_lens: torch.Tensor      # (batch_size,) - tokens already in cache
    seq_lens: torch.Tensor          # (batch_size,) - total sequence lengths (context + new)
    is_prefill: bool                # True if this is prefill phase


def create_attention_metadata(
    batch_size: int,
    seq_lens: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    device: torch.device,
) -> AttentionMetadata:
    """
    Create attention metadata for paged attention.
    """
    slot_mappings = []
    
    for batch_idx in range(batch_size):
        context_len = context_lens[batch_idx].item()
        seq_len = seq_lens[batch_idx].item()
        
        for pos in range(context_len, seq_len):
            block_idx = pos // block_size
            block_offset = pos % block_size
            physical_block = block_table[batch_idx, block_idx].item()
            slot = physical_block * block_size + block_offset
            slot_mappings.append(slot)
    
    slot_mapping = torch.tensor(slot_mappings, dtype=torch.long, device=device)
    
    is_prefill = (context_lens == 0).all().item() and (seq_lens > 1).any().item()
    
    return AttentionMetadata(
        slot_mapping=slot_mapping,
        block_table=block_table,
        context_lens=context_lens,
        seq_lens=seq_lens,
        is_prefill=is_prefill,
    )


class Model(nn.Module):
    """
    Multi-Head Latent Attention (MLA) with Paged KV Cache.
    
    This implementation stores compressed latent (kv_c concatenated with k_pe)
    in the cache, similar to vLLM's approach.
    
    The cache stores: [kv_c (kv_lora_rank), k_pe (qk_rope_head_dim)]
    
    This operator takes:
    - Pre-projected Q with RoPE already applied to the rope portion
    - kv_c: compressed KV latent (for up-projection to K_nope and V)
    - k_pe: the rope-encoded key component
    
    The KV up-projections (kv_b_proj) are done inside this operator.
    """

    def __init__(
        self,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        kv_lora_rank: int,
        block_size: int = 16,
        num_blocks: int = 1024,
        max_seq_len: int = 8192,
        dropout: float = 0.0,
        softmax_scale: Optional[float] = None,
    ):
        """
        Initialize MLA with paged latent cache.

        Args:
            num_heads: Number of attention heads
            qk_nope_head_dim: Non-RoPE dimension for Q/K
            qk_rope_head_dim: RoPE dimension for Q/K
            v_head_dim: Value head dimension
            kv_lora_rank: Rank for KV compression (stored in cache)
            block_size: Number of tokens per cache block/page
            num_blocks: Total number of blocks in the cache pool
            max_seq_len: Maximum sequence length
            dropout: Attention dropout probability
            softmax_scale: Optional scaling factor (default: 1/sqrt(qk_head_dim))
        """
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        # Cache stores: [kv_c, k_pe] = kv_lora_rank + qk_rope_head_dim
        self.cache_dim = kv_lora_rank + qk_rope_head_dim

        if softmax_scale is not None:
            self.scale = softmax_scale
        else:
            self.scale = 1.0 / math.sqrt(self.qk_head_dim)

        # KV up-projection: kv_c -> [k_nope, v]
        # Output: num_heads * (qk_nope_head_dim + v_head_dim)
        self.kv_b_proj = nn.Linear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False
        )

        # Initialize paged latent cache as buffer
        # Shape: (num_blocks, block_size, cache_dim)
        self.register_buffer('latent_cache', torch.zeros(
            num_blocks, block_size, self.cache_dim
        ))

    def reset_cache(self):
        """Reset the latent cache to zeros."""
        self.latent_cache.zero_()

    def _write_to_cache(
        self,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        slot_mapping: torch.Tensor
    ) -> None:
        """
        Write new latent to cache using slot_mapping.
        
        Args:
            kv_c: (batch_size, seq_len, kv_lora_rank)
            k_pe: (batch_size, 1, seq_len, qk_rope_head_dim)
            slot_mapping: (num_tokens,) - absolute positions in cache
        """
        if slot_mapping.numel() == 0:
            return
        
        # Flatten kv_c to (num_tokens, kv_lora_rank)
        kv_c_flat = kv_c.reshape(-1, self.kv_lora_rank)
        
        # k_pe is (batch, 1, seq, rope_dim) - squeeze and flatten
        k_pe_flat = k_pe.squeeze(1).reshape(-1, self.qk_rope_head_dim)
        
        # Concatenate to form cache entry
        cache_entry = torch.cat([kv_c_flat, k_pe_flat], dim=-1)
        
        # Compute block indices and offsets
        block_indices = slot_mapping // self.block_size
        block_offsets = slot_mapping % self.block_size
        
        # Write to cache
        self.latent_cache[block_indices, block_offsets] = cache_entry

    def _gather_from_cache_truncated(
        self,
        block_table: torch.Tensor,
        context_lens: torch.Tensor,
        max_context_len: int
    ) -> tuple:
        """
        Gather latent from paged cache, truncated to max_context_len.

        Returns:
            kv_c_cache: (batch_size, max_context_len, kv_lora_rank)
            k_pe_cache: (batch_size, 1, max_context_len, qk_rope_head_dim)
        """
        batch_size = block_table.shape[0]
        device = self.latent_cache.device
        
        num_blocks_needed = (max_context_len + self.block_size - 1) // self.block_size
        block_table_truncated = block_table[:, :num_blocks_needed]
        
        gathered_blocks = self.latent_cache[block_table_truncated.flatten()]
        gathered_blocks = gathered_blocks.view(
            batch_size, num_blocks_needed, self.block_size, self.cache_dim
        )
        
        gathered = gathered_blocks.view(
            batch_size, num_blocks_needed * self.block_size, self.cache_dim
        )
        gathered = gathered[:, :max_context_len, :]
        
        # Split into kv_c and k_pe
        kv_c_cache = gathered[..., :self.kv_lora_rank]
        k_pe_cache = gathered[..., self.kv_lora_rank:].unsqueeze(1)  # Add head dim
        
        return kv_c_cache, k_pe_cache

    def _prefill_attention(
        self,
        q: torch.Tensor,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attention during prefill phase.
        
        Args:
            q: (batch_size, num_heads, seq_len, qk_head_dim)
            kv_c: (batch_size, seq_len, kv_lora_rank)
            k_pe: (batch_size, 1, seq_len, qk_rope_head_dim)
        """
        batch_size, num_heads, seq_len, _ = q.shape
        device = q.device
        
        # Up-project kv_c to get k_nope and v
        kv = self.kv_b_proj(kv_c)  # (batch, seq, num_heads * (qk_nope + v))
        kv = kv.view(batch_size, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        kv = kv.transpose(1, 2)  # (batch, heads, seq, qk_nope + v)
        
        k_nope = kv[..., :self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim:]
        
        # Construct full K by concatenating k_nope and k_pe
        # k_pe is (batch, 1, seq, rope_dim) - broadcast to all heads
        k = torch.cat([k_nope, k_pe.expand(-1, num_heads, -1, -1)], dim=-1)
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Softmax in float32 for numerical stability, then convert to value dtype
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32)
        if self.dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)
        
        attn_output = torch.matmul(attn_weights.to(v.dtype), v)
        return attn_output

    def _decode_attention(
        self,
        q: torch.Tensor,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        block_table: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attention during decode phase.
        """
        batch_size, num_heads, seq_len, _ = q.shape
        device = q.device
        
        max_context_len = context_lens.max().item()
        
        if max_context_len > 0:
            kv_c_cache, k_pe_cache = self._gather_from_cache_truncated(
                block_table, context_lens, max_context_len
            )
            
            # Concatenate cached and new
            kv_c_full = torch.cat([kv_c_cache, kv_c], dim=1)
            k_pe_full = torch.cat([k_pe_cache, k_pe], dim=2)
        else:
            kv_c_full = kv_c
            k_pe_full = k_pe
        
        total_len = kv_c_full.shape[1]
        
        # Up-project to get k_nope and v
        kv = self.kv_b_proj(kv_c_full)
        kv = kv.view(batch_size, total_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        kv = kv.transpose(1, 2)
        
        k_nope = kv[..., :self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim:]
        
        # Construct full K
        k = torch.cat([k_nope, k_pe_full.expand(-1, num_heads, -1, -1)], dim=-1)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Causal + padding mask
        query_positions = torch.arange(seq_len, device=device).unsqueeze(0) + context_lens.unsqueeze(1)
        key_positions = torch.arange(total_len, device=device).unsqueeze(0)
        
        causal_mask = key_positions > query_positions.unsqueeze(-1)
        
        if max_context_len > 0:
            cache_positions = torch.arange(max_context_len, device=device).unsqueeze(0)
            padding_mask = cache_positions >= context_lens.unsqueeze(1)
            padding_mask = torch.cat([
                padding_mask,
                torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
            ], dim=1)
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
        else:
            padding_mask = torch.zeros(batch_size, 1, 1, total_len, dtype=torch.bool, device=device)
        
        causal_mask = causal_mask.unsqueeze(1)
        attn_mask = causal_mask | padding_mask
        
        scores = scores.masked_fill(attn_mask, float('-inf'))
        
        # Softmax in float32 for numerical stability, then convert to value dtype
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32)
        if self.dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)
        
        attn_output = torch.matmul(attn_weights.to(v.dtype), v)
        return attn_output

    def forward(
        self,
        q: torch.Tensor,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """
        Forward pass with paged latent cache and MLA.

        Args:
            q: Projected query with RoPE applied (batch_size, num_heads, seq_len, qk_head_dim)
            kv_c: Compressed KV latent (batch_size, seq_len, kv_lora_rank)
            k_pe: RoPE-encoded K component (batch_size, 1, seq_len, qk_rope_head_dim)
            attn_metadata: Attention metadata

        Returns:
            Output tensor (batch_size, num_heads, seq_len, v_head_dim)
        """
        # Ensure cache is on same device/dtype as inputs
        if self.latent_cache.device != q.device or self.latent_cache.dtype != q.dtype:
            # Must use register_buffer to properly update the buffer in the module
            new_cache = self.latent_cache.to(device=q.device, dtype=q.dtype)
            self.register_buffer('latent_cache', new_cache, persistent=False)
        
        # Write new latent to cache
        self._write_to_cache(kv_c, k_pe, attn_metadata.slot_mapping)
        
        # Choose prefill or decode path
        if attn_metadata.is_prefill:
            return self._prefill_attention(q, kv_c, k_pe)
        else:
            return self._decode_attention(
                q, kv_c, k_pe,
                attn_metadata.block_table,
                attn_metadata.context_lens
            )


# ============================================================================
# Benchmark Configuration
# ============================================================================
