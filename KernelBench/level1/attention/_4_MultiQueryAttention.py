import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


# ============================================================================
# Attention Metadata (vLLM-style) - shared with GQA
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


class Model(nn.Module):
    """
    Multi-Query Attention (MQA) with Paged KV Cache

    Used by: Falcon-7B, PaLM, StarCoder

    MQA uses a single key/value head shared across all query heads,
    dramatically reducing KV cache memory. Uses paged KV cache where
    cache entries are stored in non-contiguous blocks via a page table.
    
    This implementation owns its KV cache (per-layer cache pattern from vLLM).
    
    NOTE: Q/K/V projections are done separately using Linear operators.
    This operator takes pre-projected Q, K, V tensors.

    Shapes:
        q: (batch_size, num_heads, seq_len, head_dim) - projected query
        k: (batch_size, 1, seq_len, head_dim) - projected key (single head, new tokens)
        v: (batch_size, 1, seq_len, head_dim) - projected value (single head, new tokens)
        Output: (batch_size, num_heads, seq_len, head_dim)
    """

    def __init__(self, num_heads: int, head_dim: int, block_size: int = 16,
                 num_blocks: int = 1024, max_seq_len: int = 2048, dropout: float = 0.0):
        """
        Initialize MQA with paged KV cache.

        Args:
            num_heads: Number of query heads
            head_dim: Dimension of each attention head
            block_size: Number of tokens per cache block/page
            num_blocks: Total number of blocks in the cache pool
            max_seq_len: Maximum sequence length
            dropout: Attention dropout probability
        """
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.num_kv_heads = 1  # MQA always uses 1 KV head
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Initialize paged KV cache as buffer (not parameter - no gradients)
        # Shape: (num_blocks, block_size, 1, head_dim, 2) where 2 = [K, V]
        # Note: num_kv_heads=1 for MQA
        self.register_buffer('kv_cache', torch.zeros(
            num_blocks, block_size, 1, head_dim, 2
        ))

    def reset_cache(self):
        """Reset the KV cache to zeros."""
        self.kv_cache.zero_()

    def _write_to_cache(self, k: torch.Tensor, v: torch.Tensor,
                        slot_mapping: torch.Tensor) -> None:
        """
        Write new K,V to cache using slot_mapping.
        
        Args:
            k: (batch_size, 1, seq_len, head_dim)
            v: (batch_size, 1, seq_len, head_dim)
            slot_mapping: (num_tokens,) - absolute positions in cache
        """
        if slot_mapping.numel() == 0:
            return
            
        # Flatten K,V to (num_tokens, 1, head_dim)
        k_flat = k.transpose(1, 2).reshape(-1, 1, self.head_dim)
        v_flat = v.transpose(1, 2).reshape(-1, 1, self.head_dim)
        
        # Compute block indices and offsets from slot_mapping
        block_indices = slot_mapping // self.block_size
        block_offsets = slot_mapping % self.block_size
        
        # Write to cache using advanced indexing
        self.kv_cache[block_indices, block_offsets, :, :, 0] = k_flat
        self.kv_cache[block_indices, block_offsets, :, :, 1] = v_flat

    def _gather_kv_from_cache_truncated(self, block_table: torch.Tensor,
                                         context_lens: torch.Tensor,
                                         max_context_len: int) -> tuple:
        """
        Gather K and V tensors from paged cache, truncated to max_context_len.

        Args:
            block_table: (batch_size, max_blocks_per_seq)
            context_lens: (batch_size,)
            max_context_len: Maximum context length to gather

        Returns:
            k_cache: (batch_size, 1, max_context_len, head_dim)
            v_cache: (batch_size, 1, max_context_len, head_dim)
        """
        batch_size = block_table.shape[0]
        device = self.kv_cache.device
        
        # Calculate how many blocks we need
        num_blocks_needed = (max_context_len + self.block_size - 1) // self.block_size
        
        # Gather only the blocks we need
        block_table_truncated = block_table[:, :num_blocks_needed]
        
        gathered_blocks = self.kv_cache[block_table_truncated.flatten()]
        gathered_blocks = gathered_blocks.view(
            batch_size, num_blocks_needed, self.block_size, 1, self.head_dim, 2
        )
        
        # Reshape to (batch, num_blocks * block_size, 1, head_dim, 2)
        gathered = gathered_blocks.view(
            batch_size, num_blocks_needed * self.block_size, 1, self.head_dim, 2
        )
        
        # Truncate to exactly max_context_len
        gathered = gathered[:, :max_context_len, :, :, :]

        # Split K and V, transpose to (batch, 1, context, head_dim)
        k_cache = gathered[..., 0].transpose(1, 2)
        v_cache = gathered[..., 1].transpose(1, 2)

        return k_cache, v_cache

    def _prefill_attention(self, q: torch.Tensor, k: torch.Tensor, 
                           v: torch.Tensor) -> torch.Tensor:
        """
        Compute attention during prefill phase (no cache read needed).
        
        All tokens are new, so we only compute attention on the input K,V.
        K and V are broadcast across all query heads (MQA).
        
        Args:
            q: (batch_size, num_heads, seq_len, head_dim)
            k: (batch_size, 1, seq_len, head_dim)
            v: (batch_size, 1, seq_len, head_dim)
            
        Returns:
            Output tensor (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        device = q.device

        # Compute attention scores - K broadcasts across all heads
        # q: (batch, heads, seq, dim), k: (batch, 1, seq, dim)
        # scores: (batch, heads, seq, seq)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask for prefill: position i can only attend to positions <= i
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        if self.dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        # V broadcasts across all heads
        attn_output = torch.matmul(attn_weights, v)
        return attn_output

    def _decode_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          block_table: torch.Tensor, 
                          context_lens: torch.Tensor) -> torch.Tensor:
        """
        Compute attention during decode phase (with cache read).
        
        Args:
            q: (batch_size, num_heads, seq_len, head_dim) - typically seq_len=1
            k: (batch_size, 1, seq_len, head_dim) - new tokens
            v: (batch_size, 1, seq_len, head_dim) - new tokens
            block_table: (batch_size, max_blocks_per_seq)
            context_lens: (batch_size,) - tokens already in cache (before new tokens)
            
        Returns:
            Output tensor (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        device = q.device

        # For decode, we need to handle variable context lengths per sequence
        max_context_len = context_lens.max().item()
        
        if max_context_len > 0:
            # Gather K, V from cache (positions 0 to max_context_len-1)
            k_cache, v_cache = self._gather_kv_from_cache_truncated(
                block_table, context_lens, max_context_len
            )
            
            # Concatenate cached and new K, V
            # k_cache: (batch, 1, max_context_len, head_dim)
            # k: (batch, 1, seq_len, head_dim)
            k_full = torch.cat([k_cache, k], dim=2)
            v_full = torch.cat([v_cache, v], dim=2)
        else:
            # No cache, just use new K, V
            k_full = k
            v_full = v

        total_len = k_full.shape[2]

        # Compute attention scores - K broadcasts across all heads
        scores = torch.matmul(q, k_full.transpose(-2, -1)) * self.scale

        # Create attention mask that combines:
        # 1. Causal masking: query at position context_len + i can attend to 0..context_len+i
        # 2. Padding masking: mask out positions >= context_len (for padded cached positions)
        
        # Query positions (relative to start of sequence)
        query_positions = torch.arange(seq_len, device=device).unsqueeze(0) + context_lens.unsqueeze(1)
        # Key positions
        key_positions = torch.arange(total_len, device=device).unsqueeze(0)
        
        # Causal mask: can't attend to future positions
        causal_mask = key_positions > query_positions.unsqueeze(-1)
        
        # Padding mask for cached K/V: mask positions that are padding (>= context_len in cache portion)
        if max_context_len > 0:
            # For cache portion (first max_context_len positions), mask if position >= per-sequence context_len
            cache_positions = torch.arange(max_context_len, device=device).unsqueeze(0)
            padding_mask = cache_positions >= context_lens.unsqueeze(1)  # (batch, max_context_len)
            # Extend to cover new tokens (which are never padded)
            padding_mask = torch.cat([
                padding_mask,
                torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
            ], dim=1)  # (batch, total_len)
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, total_len)
        else:
            padding_mask = torch.zeros(batch_size, 1, 1, total_len, dtype=torch.bool, device=device)
        
        # Combine masks
        causal_mask = causal_mask.unsqueeze(1)  # (batch, 1, seq_len, total_len)
        attn_mask = causal_mask | padding_mask

        scores = scores.masked_fill(attn_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        if self.dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        # V broadcasts across all heads
        attn_output = torch.matmul(attn_weights, v_full)
        return attn_output

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                attn_metadata: AttentionMetadata) -> torch.Tensor:
        """
        Forward pass with paged KV cache and multi-query attention.

        Args:
            q: Projected query (batch_size, num_heads, seq_len, head_dim)
            k: Projected key for new tokens (batch_size, 1, seq_len, head_dim)
            v: Projected value for new tokens (batch_size, 1, seq_len, head_dim)
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


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    # Prefill-heavy: Falcon-7B initial prompt processing (2048 tokens)
    {"batch_size": 4, "seq_len": 2048, "context_len": 0, "num_heads": 71, "head_dim": 64, "block_size": 16, "max_blocks_per_seq": 129, "num_blocks": 520},
    # Prefill-heavy: Falcon-7B chunked prefill (1024 token chunks)
    {"batch_size": 8, "seq_len": 1024, "context_len": 1024, "num_heads": 71, "head_dim": 64, "block_size": 16, "max_blocks_per_seq": 129, "num_blocks": 1040},
    # Decode-heavy: Falcon-7B high-throughput decoding (1 token, 2k context)
    {"batch_size": 64, "seq_len": 1, "context_len": 2048, "num_heads": 71, "head_dim": 64, "block_size": 16, "max_blocks_per_seq": 129, "num_blocks": 8300},
    # Decode-heavy: StarCoder2-15B batched generation (1 token, 4k context)
    {"batch_size": 16, "seq_len": 1, "context_len": 4096, "num_heads": 48, "head_dim": 128, "block_size": 16, "max_blocks_per_seq": 257, "num_blocks": 4200},
    # Decode-heavy: StarCoder2-15B long context decoding (1 token, 8k context)
    {"batch_size": 8, "seq_len": 1, "context_len": 8192, "num_heads": 48, "head_dim": 128, "block_size": 16, "max_blocks_per_seq": 513, "num_blocks": 4200},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("attention", "4_MultiQueryAttention")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    # Pre-projected Q, K, V tensors (K, V have single head for MQA)
    q = DISTRIBUTIONS[dist_name]((p["batch_size"], p["num_heads"], p["seq_len"], p["head_dim"]), dtype=dtype, device=device)
    k = DISTRIBUTIONS[dist_name]((p["batch_size"], 1, p["seq_len"], p["head_dim"]), dtype=dtype, device=device)
    v = DISTRIBUTIONS[dist_name]((p["batch_size"], 1, p["seq_len"], p["head_dim"]), dtype=dtype, device=device)
    
    # Create attention metadata
    batch_size = p["batch_size"]
    seq_len = p["seq_len"]
    context_len = p["context_len"]
    block_size = p["block_size"]
    
    block_table = torch.randint(0, p["num_blocks"], (batch_size, p["max_blocks_per_seq"]), device=device)
    context_lens = torch.full((batch_size,), context_len, dtype=torch.long, device=device)
    seq_lens = context_lens + seq_len
    
    attn_metadata = create_attention_metadata(
        batch_size=batch_size,
        seq_lens=seq_lens,
        context_lens=context_lens,
        block_table=block_table,
        block_size=block_size,
        device=device,
    )
    
    return [q, k, v, attn_metadata]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["num_heads"], p["head_dim"], p["block_size"], p["num_blocks"]]
