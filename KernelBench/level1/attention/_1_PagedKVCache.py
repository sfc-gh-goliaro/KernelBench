import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple


# ============================================================================
# Attention Metadata (vLLM-style)
# ============================================================================

@dataclass
class AttentionMetadata:
    """
    Metadata for paged attention operations.

    Aligned with vLLM's CommonAttentionMetadata pattern.
    """
    slot_mapping: torch.Tensor      # (num_tokens,) - where to write new K,V
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

    Args:
        batch_size: Number of sequences in batch
        seq_lens: (batch_size,) total sequence lengths (context + new tokens)
        context_lens: (batch_size,) tokens already in cache
        block_table: (batch_size, max_blocks_per_seq) block assignments
        block_size: Number of tokens per block
        device: Device for tensors

    Returns:
        AttentionMetadata with computed slot_mapping
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
    Paged KV Cache

    Used by: All autoregressive models (Llama, Falcon, BLOOM, Mistral, Mixtral, etc.)

    Manages a paged KV cache for efficient autoregressive decoding.
    Cache entries are stored in non-contiguous blocks accessed via a page table.
    Supports MHA, GQA, and MQA transparently via num_kv_heads parameter.

    This operator is cache-only — it does NOT compute attention.
    The model layer orchestrates: write → gather → attention (separate operator).

    Operations:
        write(k, v, slot_mapping): Store new K/V entries into cache slots
        gather(block_table, context_lens): Read cached K/V for decode

    Cache shape: (num_blocks, block_size, num_kv_heads, head_dim, 2)
        where the last dimension indexes [K=0, V=1]
    """

    def __init__(
        self,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        num_blocks: int = 1024,
    ):
        """
        Initialize paged KV cache.

        Args:
            num_kv_heads: Number of key/value heads (1 for MQA, num_heads for MHA,
                          or any divisor of num_heads for GQA)
            head_dim: Dimension of each attention head
            block_size: Number of tokens per cache block/page
            num_blocks: Total number of blocks in the cache pool
        """
        super(Model, self).__init__()
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks

        # Initialize paged KV cache as buffer (not parameter — no gradients)
        self.register_buffer('kv_cache', torch.zeros(
            num_blocks, block_size, num_kv_heads, head_dim, 2
        ))

    def reset(self):
        """Reset the KV cache to zeros."""
        self.kv_cache.zero_()

    def write(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """
        Write new K, V entries to cache using slot_mapping.

        Args:
            k: (batch_size, num_kv_heads, seq_len, head_dim)
            v: (batch_size, num_kv_heads, seq_len, head_dim)
            slot_mapping: (num_tokens,) — absolute slot positions in cache
        """
        if slot_mapping.numel() == 0:
            return

        # Ensure cache is on same device/dtype as inputs
        if self.kv_cache.device != k.device or self.kv_cache.dtype != k.dtype:
            self.kv_cache = self.kv_cache.to(device=k.device, dtype=k.dtype)

        # Flatten K, V to (num_tokens, num_kv_heads, head_dim)
        k_flat = k.transpose(1, 2).reshape(-1, self.num_kv_heads, self.head_dim)
        v_flat = v.transpose(1, 2).reshape(-1, self.num_kv_heads, self.head_dim)

        # Compute block indices and offsets
        block_indices = slot_mapping // self.block_size
        block_offsets = slot_mapping % self.block_size

        # Write to cache
        self.kv_cache[block_indices, block_offsets, :, :, 0] = k_flat
        self.kv_cache[block_indices, block_offsets, :, :, 1] = v_flat

    def gather(
        self,
        block_table: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gather K and V tensors from paged cache, truncated to max context length.

        Args:
            block_table: (batch_size, max_blocks_per_seq)
            context_lens: (batch_size,)

        Returns:
            k_cache: (batch_size, num_kv_heads, max_context_len, head_dim)
            v_cache: (batch_size, num_kv_heads, max_context_len, head_dim)
        """
        max_context_len = context_lens.max().item()
        if max_context_len == 0:
            batch_size = block_table.shape[0]
            device = self.kv_cache.device
            dtype = self.kv_cache.dtype
            empty_k = torch.zeros(batch_size, self.num_kv_heads, 0, self.head_dim,
                                  device=device, dtype=dtype)
            empty_v = torch.zeros(batch_size, self.num_kv_heads, 0, self.head_dim,
                                  device=device, dtype=dtype)
            return empty_k, empty_v

        batch_size = block_table.shape[0]
        device = self.kv_cache.device

        # Calculate how many blocks we need
        num_blocks_needed = (max_context_len + self.block_size - 1) // self.block_size

        # Gather only the blocks we need
        block_table_truncated = block_table[:, :num_blocks_needed]

        gathered_blocks = self.kv_cache[block_table_truncated.flatten()]
        gathered_blocks = gathered_blocks.view(
            batch_size, num_blocks_needed, self.block_size,
            self.num_kv_heads, self.head_dim, 2
        )

        # Reshape to (batch, num_blocks * block_size, kv_heads, head_dim, 2)
        gathered = gathered_blocks.view(
            batch_size, num_blocks_needed * self.block_size,
            self.num_kv_heads, self.head_dim, 2
        )

        # Truncate to exactly max_context_len
        gathered = gathered[:, :max_context_len, :, :, :]

        # Split K and V, transpose to (batch, kv_heads, context, head_dim)
        k_cache = gathered[..., 0].transpose(1, 2)
        v_cache = gathered[..., 1].transpose(1, 2)

        return k_cache, v_cache

    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Write new K/V to cache and gather full K/V for attention.

        This is a convenience method that combines write + gather + concat.

        Args:
            k: New key (batch_size, num_kv_heads, seq_len, head_dim)
            v: New value (batch_size, num_kv_heads, seq_len, head_dim)
            attn_metadata: Attention metadata

        Returns:
            k_full: (batch_size, num_kv_heads, total_len, head_dim)
            v_full: (batch_size, num_kv_heads, total_len, head_dim)
        """
        # Write new K/V to cache
        self.write(k, v, attn_metadata.slot_mapping)

        if attn_metadata.is_prefill:
            # During prefill, all tokens are new — no need to gather from cache
            return k, v
        else:
            # During decode, gather cached K/V and concatenate with new
            k_cache, v_cache = self.gather(
                attn_metadata.block_table, attn_metadata.context_lens
            )
            k_full = torch.cat([k_cache, k], dim=2)
            v_full = torch.cat([v_cache, v], dim=2)
            return k_full, v_full
