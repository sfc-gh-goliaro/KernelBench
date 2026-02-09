import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Sliding Window Attention with Paged KV Cache

    Used by: Mistral, Gemma-2, Longformer

    Local attention where each token only attends to a fixed window of
    nearby tokens. This reduces memory and compute from O(n²) to O(n*w)
    where w is the window size. Uses paged KV cache for efficient memory.
    
    NOTE: Q/K/V projections are done separately using Linear operators.
    This operator takes pre-projected Q, K, V tensors.

    Shapes:
        q: (batch_size, num_heads, seq_len, head_dim) - projected query
        k: (batch_size, num_heads, seq_len, head_dim) - projected key (new tokens)
        v: (batch_size, num_heads, seq_len, head_dim) - projected value (new tokens)
        kv_cache_pool: (num_blocks, block_size, num_heads, head_dim, 2) - paged KV pool
        block_table: (batch_size, max_blocks_per_seq) - maps logical to physical blocks
        context_lens: (batch_size,) - number of cached tokens per sequence
        Output: (batch_size, num_heads, seq_len, head_dim)
    """

    def __init__(self, num_heads: int, head_dim: int, window_size: int = 4096,
                 block_size: int = 16, dropout: float = 0.0):
        """
        Initialize sliding window attention with paged KV cache.

        Args:
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            window_size: Size of attention window
            block_size: Number of tokens per cache block/page
            dropout: Attention dropout probability
        """
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.block_size = block_size
        self.dropout = dropout

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def _gather_kv_from_paged_cache(self, kv_cache_pool: torch.Tensor,
                                     block_table: torch.Tensor,
                                     context_lens: torch.Tensor) -> tuple:
        """
        Gather K and V tensors from paged cache using block table.

        Args:
            kv_cache_pool: (num_blocks, block_size, num_heads, head_dim, 2)
            block_table: (batch_size, max_blocks_per_seq)
            context_lens: (batch_size,)

        Returns:
            k_cache: (batch_size, num_heads, max_context_len, head_dim)
            v_cache: (batch_size, num_heads, max_context_len, head_dim)
        """
        batch_size = block_table.shape[0]
        max_blocks = block_table.shape[1]
        max_context_len = max_blocks * self.block_size
        device = kv_cache_pool.device

        gathered_blocks = kv_cache_pool[block_table.flatten()]
        gathered_blocks = gathered_blocks.view(
            batch_size, max_blocks, self.block_size, self.num_heads, self.head_dim, 2
        )
        gathered = gathered_blocks.view(
            batch_size, max_context_len, self.num_heads, self.head_dim, 2
        )

        k_cache = gathered[..., 0].transpose(1, 2)
        v_cache = gathered[..., 1].transpose(1, 2)

        positions = torch.arange(max_context_len, device=device).unsqueeze(0)
        mask = positions >= context_lens.unsqueeze(1)
        mask = mask.unsqueeze(1).unsqueeze(-1)
        k_cache = k_cache.masked_fill(mask, 0)
        v_cache = v_cache.masked_fill(mask, 0)

        return k_cache, v_cache

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                kv_cache_pool: torch.Tensor, block_table: torch.Tensor,
                context_lens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with paged KV cache and sliding window attention.

        Args:
            q: Projected query (batch_size, num_heads, seq_len, head_dim)
            k: Projected key for new tokens (batch_size, num_heads, seq_len, head_dim)
            v: Projected value for new tokens (batch_size, num_heads, seq_len, head_dim)
            kv_cache_pool: Paged KV cache (num_blocks, block_size, num_heads, head_dim, 2)
            block_table: Block table (batch_size, max_blocks_per_seq)
            context_lens: Context lengths (batch_size,)

        Returns:
            Output tensor (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        device = q.device

        # Gather K, V from paged cache
        k_cache, v_cache = self._gather_kv_from_paged_cache(
            kv_cache_pool, block_table, context_lens
        )

        # Concatenate cached and new K, V
        k_full = torch.cat([k_cache, k], dim=2)
        v_full = torch.cat([v_cache, v], dim=2)

        total_len = k_full.shape[2]

        # Compute attention scores
        scores = torch.matmul(q, k_full.transpose(-2, -1)) * self.scale

        # Create sliding window causal mask
        query_positions = torch.arange(seq_len, device=device).unsqueeze(0) + context_lens.unsqueeze(1)
        key_positions = torch.arange(total_len, device=device).unsqueeze(0)

        # Causal: can't attend to future
        causal_mask = key_positions > query_positions.unsqueeze(-1)

        # Sliding window: can't attend beyond window_size back
        window_start = query_positions - self.window_size + 1
        window_mask = key_positions < window_start.unsqueeze(-1)

        # Combined mask
        full_mask = causal_mask | window_mask
        full_mask = full_mask.unsqueeze(1)

        scores = scores.masked_fill(full_mask, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        if self.dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        attn_output = torch.matmul(attn_weights, v_full)

        return attn_output  # (batch, heads, seq_len, head_dim)


# ============================================================================
# Benchmark Configuration
# ============================================================================
