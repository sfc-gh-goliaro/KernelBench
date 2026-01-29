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
    # Compute slot_mapping: for each new token, where in the cache to write it
    # New tokens are at positions [context_len, context_len + 1, ..., seq_len - 1]
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
    
    # Determine if this is prefill (all context_lens are 0 and seq_len > 1)
    # or decode (context_lens > 0 or seq_len == 1)
    is_prefill = (context_lens == 0).all().item() and (seq_lens > 1).any().item()
    
    return AttentionMetadata(
        slot_mapping=slot_mapping,
        block_table=block_table,
        context_lens=context_lens,
        seq_lens=seq_lens,
        is_prefill=is_prefill,
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
    Attention with Linear Biases (ALiBi) with Paged KV Cache
    
    Used by: BLOOM, MPT
    
    ALiBi adds a linear bias to attention scores based on position.
    This provides position information without explicit position embeddings
    and extrapolates well to longer sequences.
    
    This implementation matches HuggingFace BLOOM exactly for numerical consistency.
    Uses paged KV cache for efficient memory management (vLLM-style).
    
    NOTE: Q/K/V projections are done separately using Linear operators.
    This operator takes pre-projected Q, K, V tensors.
    
    Shapes:
        q: (batch_size, num_heads, seq_len, head_dim) - projected query
        k: (batch_size, num_heads, seq_len, head_dim) - projected key (new tokens)
        v: (batch_size, num_heads, seq_len, head_dim) - projected value (new tokens)
        Output: (batch_size, num_heads, seq_len, head_dim)
    """

    def __init__(self, num_heads: int, head_dim: int, block_size: int = 16,
                 num_blocks: int = 1024):
        """
        Initialize ALiBi attention with paged KV cache.
        
        Args:
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            block_size: Number of tokens per cache block/page
            num_blocks: Total number of blocks in the cache pool
        """
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks

        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Compute ALiBi slopes for each head (matching HuggingFace)
        slopes = get_alibi_slopes(num_heads)
        self.register_buffer('alibi_slopes', slopes)

        # Initialize paged KV cache as buffer (not parameter - no gradients)
        # Shape: (num_blocks, block_size, num_heads, head_dim, 2) where 2 = [K, V]
        self.register_buffer('kv_cache', torch.zeros(
            num_blocks, block_size, num_heads, head_dim, 2
        ))

    def reset_cache(self):
        """Reset the KV cache."""
        self.kv_cache.zero_()

    def _write_to_cache(self, k: torch.Tensor, v: torch.Tensor,
                        slot_mapping: torch.Tensor) -> None:
        """
        Write new K,V to cache using slot_mapping.
        
        Args:
            k: (batch_size, num_heads, seq_len, head_dim)
            v: (batch_size, num_heads, seq_len, head_dim)
            slot_mapping: (num_tokens,) - absolute positions in cache
        """
        if slot_mapping.numel() == 0:
            return

        # Flatten K,V to (num_tokens, num_heads, head_dim)
        k_flat = k.transpose(1, 2).reshape(-1, self.num_heads, self.head_dim)
        v_flat = v.transpose(1, 2).reshape(-1, self.num_heads, self.head_dim)

        # Compute block indices and offsets from slot_mapping
        block_indices = slot_mapping // self.block_size
        block_offsets = slot_mapping % self.block_size

        # Write to cache using advanced indexing
        self.kv_cache[block_indices, block_offsets, :, :, 0] = k_flat
        self.kv_cache[block_indices, block_offsets, :, :, 1] = v_flat

    def _gather_kv_from_cache(self, block_table: torch.Tensor,
                               context_lens: torch.Tensor,
                               max_context_len: int) -> tuple:
        """
        Gather K and V tensors from paged cache.
        
        Args:
            block_table: (batch_size, max_blocks_per_seq)
            context_lens: (batch_size,)
            max_context_len: Maximum context length to gather
            
        Returns:
            k_cache: (batch_size, num_heads, max_context_len, head_dim)
            v_cache: (batch_size, num_heads, max_context_len, head_dim)
        """
        batch_size = block_table.shape[0]
        device = self.kv_cache.device

        # Calculate how many blocks we need
        num_blocks_needed = (max_context_len + self.block_size - 1) // self.block_size

        # Gather only the blocks we need
        block_table_truncated = block_table[:, :num_blocks_needed]

        gathered_blocks = self.kv_cache[block_table_truncated.flatten()]
        gathered_blocks = gathered_blocks.view(
            batch_size, num_blocks_needed, self.block_size, self.num_heads, self.head_dim, 2
        )

        # Reshape to (batch, num_blocks * block_size, heads, head_dim, 2)
        gathered = gathered_blocks.view(
            batch_size, num_blocks_needed * self.block_size, self.num_heads, self.head_dim, 2
        )

        # Truncate to exactly max_context_len
        gathered = gathered[:, :max_context_len, :, :, :]

        # Split K and V, transpose to (batch, heads, context, head_dim)
        k_cache = gathered[..., 0].transpose(1, 2)
        v_cache = gathered[..., 1].transpose(1, 2)

        return k_cache, v_cache

    def _build_alibi_tensor(self, seq_len: int, batch_size: int,
                             device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Build ALiBi tensor matching HuggingFace's implementation.
        
        Returns:
            alibi: (batch_size * num_heads, 1, seq_len) - added to attention scores
        """
        # Positions: [0, 1, 2, ..., seq_len-1]
        arange_tensor = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(0)
        # Apply slopes: (num_heads, 1) * (1, seq_len) -> (num_heads, seq_len)
        alibi = self.alibi_slopes.to(device=device, dtype=dtype).unsqueeze(1) * arange_tensor
        # Expand for batch and reshape: (batch * num_heads, 1, seq_len)
        alibi = alibi.unsqueeze(0).expand(batch_size, -1, -1)
        alibi = alibi.reshape(batch_size * self.num_heads, 1, seq_len)
        return alibi

    def _prefill_attention(self, q: torch.Tensor, k: torch.Tensor,
                           v: torch.Tensor) -> torch.Tensor:
        """
        Compute attention during prefill phase (no cache read needed).
        Matches HuggingFace BLOOM's attention implementation.
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        device = q.device
        dtype = q.dtype

        # Reshape for batched matmul: (batch * heads, seq, head_dim)
        q_reshaped = q.reshape(batch_size * num_heads, seq_len, head_dim)
        k_reshaped = k.reshape(batch_size * num_heads, seq_len, head_dim).transpose(-1, -2)
        v_reshaped = v.reshape(batch_size * num_heads, seq_len, head_dim)

        # Build ALiBi tensor: (batch * heads, 1, seq_len)
        alibi = self._build_alibi_tensor(seq_len, batch_size, device, dtype)

        # Compute attention scores with ALiBi: alibi + scale * (q @ k.T)
        # Using baddbmm pattern: beta * alibi + alpha * (q @ k.T)
        scores = torch.baddbmm(
            alibi,
            q_reshaped,
            k_reshaped,
            beta=1.0,
            alpha=self.scale,
        )
        # scores shape: (batch * heads, seq, seq)

        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))

        # Softmax in float32 then cast back (matching HuggingFace)
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype)
        attn_output = torch.bmm(attn_weights, v_reshaped)

        # Reshape back: (batch, heads, seq, head_dim)
        attn_output = attn_output.view(batch_size, num_heads, seq_len, head_dim)
        return attn_output

    def _decode_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          block_table: torch.Tensor,
                          context_lens: torch.Tensor) -> torch.Tensor:
        """
        Compute attention during decode phase (with cache read).
        Matches HuggingFace BLOOM's attention implementation.
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        device = q.device
        dtype = q.dtype

        max_context_len = context_lens.max().item()

        if max_context_len > 0:
            # Gather K, V from cache
            k_cache, v_cache = self._gather_kv_from_cache(
                block_table, context_lens, max_context_len
            )
            # Concatenate cached and new K, V
            k_full = torch.cat([k_cache, k], dim=2)
            v_full = torch.cat([v_cache, v], dim=2)
        else:
            k_full = k
            v_full = v

        total_len = k_full.shape[2]

        # Reshape for batched matmul: (batch * heads, seq, head_dim)
        q_reshaped = q.reshape(batch_size * num_heads, seq_len, head_dim)
        k_reshaped = k_full.reshape(batch_size * num_heads, total_len, head_dim).transpose(-1, -2)
        v_reshaped = v_full.reshape(batch_size * num_heads, total_len, head_dim)

        # Build ALiBi tensor for full sequence: (batch * heads, 1, total_len)
        alibi = self._build_alibi_tensor(total_len, batch_size, device, dtype)

        # Compute attention scores with ALiBi: alibi + scale * (q @ k.T)
        scores = torch.baddbmm(
            alibi,
            q_reshaped,
            k_reshaped,
            beta=1.0,
            alpha=self.scale,
        )
        # scores shape: (batch * heads, seq_len, total_len)

        # Reshape for mask application: (batch, heads, seq_len, total_len)
        scores = scores.view(batch_size, num_heads, seq_len, total_len)

        # Causal mask: query at position i can only attend to keys at positions <= i
        # Query positions are context_lens, context_lens+1, ..., context_lens+seq_len-1
        query_positions = context_lens.view(batch_size, 1, 1) + torch.arange(seq_len, device=device).view(1, seq_len, 1)
        key_positions = torch.arange(total_len, device=device).view(1, 1, total_len)
        causal_mask = key_positions > query_positions  # (batch, seq_len, total_len)

        # Padding mask for cached K/V
        if max_context_len > 0:
            cache_positions = torch.arange(max_context_len, device=device).unsqueeze(0)
            padding_mask = cache_positions >= context_lens.unsqueeze(1)
            padding_mask = torch.cat([
                padding_mask,
                torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
            ], dim=1)
        else:
            padding_mask = torch.zeros(batch_size, total_len, dtype=torch.bool, device=device)

        # Combine masks: (batch, 1, seq_len, total_len)
        attn_mask = causal_mask | padding_mask.unsqueeze(1)
        attn_mask = attn_mask.unsqueeze(1)  # (batch, 1, seq_len, total_len)
        scores = scores.masked_fill(attn_mask, float('-inf'))

        # Reshape back and apply softmax in float32 (matching HuggingFace)
        scores = scores.view(batch_size * num_heads, seq_len, total_len)
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype)
        attn_output = torch.bmm(attn_weights, v_reshaped)

        # Reshape back: (batch, heads, seq, head_dim)
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
        # Ensure cache is on same device/dtype as inputs
        if self.kv_cache.device != q.device or self.kv_cache.dtype != q.dtype:
            self.kv_cache = self.kv_cache.to(device=q.device, dtype=q.dtype)
        
        # Write new K,V to cache
        self._write_to_cache(k, v, attn_metadata.slot_mapping)

        # Choose prefill or decode path
        if attn_metadata.is_prefill:
            return self._prefill_attention(q, k, v)
        else:
            return self._decode_attention(
                q, k, v,
                attn_metadata.block_table,
                attn_metadata.context_lens
            )
