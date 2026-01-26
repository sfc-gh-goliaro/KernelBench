import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Chunked Local Attention with Paged KV Cache

    Used by: Llama 4, Longformer-style models

    Attention that operates on local chunks/windows for efficiency.
    Reduces memory from O(n^2) to O(n * chunk_size) for long sequences.
    Uses paged KV cache for efficient memory management.
    
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

    def __init__(self, num_heads: int, head_dim: int, chunk_size: int = 512,
                 block_size: int = 16, dropout: float = 0.0):
        """
        Initialize Chunked Local Attention with paged KV cache.

        Args:
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            chunk_size: Size of each local attention chunk
            block_size: Number of tokens per cache block/page
            dropout: Attention dropout probability
        """
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
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
        Compute chunked local attention with paged KV cache.

        Args:
            q: Projected query (batch_size, num_heads, seq_len, head_dim)
            k: Projected key for new tokens (batch_size, num_heads, seq_len, head_dim)
            v: Projected value for new tokens (batch_size, num_heads, seq_len, head_dim)
            kv_cache_pool: Paged KV cache (num_blocks, block_size, num_heads, head_dim, 2)
            block_table: Block table (batch_size, max_blocks_per_seq)
            context_lens: Context lengths (batch_size,)

        Returns:
            Attention output (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        device = q.device

        # Gather K, V from paged cache
        k_cache, v_cache = self._gather_kv_from_paged_cache(
            kv_cache_pool, block_table, context_lens
        )

        # Concatenate cached and new K, V: (batch, heads, total_len, head_dim)
        k_full = torch.cat([k_cache, k], dim=2)
        v_full = torch.cat([v_cache, v], dim=2)

        total_len = k_full.shape[2]

        # Pad sequence to be divisible by chunk_size
        pad_len = (self.chunk_size - total_len % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            k_full = F.pad(k_full, (0, 0, 0, pad_len))
            v_full = F.pad(v_full, (0, 0, 0, pad_len))

        padded_len = k_full.shape[2]
        num_chunks = padded_len // self.chunk_size

        # Reshape K, V to chunks: (batch, heads, num_chunks, chunk_size, head_dim)
        k_chunks = k_full.view(batch_size, num_heads, num_chunks, self.chunk_size, head_dim)
        v_chunks = v_full.view(batch_size, num_heads, num_chunks, self.chunk_size, head_dim)

        # For queries, determine which chunk each query token attends to
        query_positions = torch.arange(seq_len, device=device).unsqueeze(0) + context_lens.unsqueeze(1)
        query_chunk_idx = query_positions // self.chunk_size
        query_chunk_idx = query_chunk_idx.clamp(max=num_chunks - 1)

        # For simplicity, compute attention per position (can be optimized)
        output = torch.zeros(batch_size, num_heads, seq_len, head_dim, device=device, dtype=q.dtype)

        for i in range(seq_len):
            chunk_idx = query_chunk_idx[0, i].item()

            q_i = q[:, :, i:i+1, :]  # (batch, heads, 1, head_dim)
            k_chunk = k_chunks[:, :, chunk_idx, :, :]  # (batch, heads, chunk_size, head_dim)
            v_chunk = v_chunks[:, :, chunk_idx, :, :]

            scores = torch.matmul(q_i, k_chunk.transpose(-2, -1)) * self.scale

            # Causal mask within chunk
            q_pos = query_positions[:, i:i+1]
            chunk_start = chunk_idx * self.chunk_size
            k_positions = torch.arange(self.chunk_size, device=device) + chunk_start

            causal_mask = k_positions.unsqueeze(0) > q_pos
            causal_mask = causal_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(causal_mask, float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)
            if self.dropout > 0 and self.training:
                attn_weights = F.dropout(attn_weights, p=self.dropout)

            out_i = torch.matmul(attn_weights, v_chunk)
            output[:, :, i:i+1, :] = out_i

        return output  # (batch, heads, seq_len, head_dim)


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    # Prefill-heavy: Llama-4 initial prompt processing (4096 tokens, chunk_size=8192)
    {"batch_size": 2, "seq_len": 4096, "context_len": 0, "num_heads": 32, "head_dim": 128, "chunk_size": 8192, "block_size": 16, "max_blocks_per_seq": 257, "num_blocks": 520},
    # Prefill-heavy: Longformer-style chunked prefill (2048 tokens)
    {"batch_size": 4, "seq_len": 2048, "context_len": 2048, "num_heads": 32, "head_dim": 128, "chunk_size": 512, "block_size": 16, "max_blocks_per_seq": 257, "num_blocks": 1040},
    # Decode-heavy: Llama-4 high-throughput decoding (1 token, 8k context)
    {"batch_size": 64, "seq_len": 1, "context_len": 8192, "num_heads": 32, "head_dim": 128, "chunk_size": 8192, "block_size": 16, "max_blocks_per_seq": 513, "num_blocks": 33000},
    # Decode-heavy: Llama-4 long context generation (1 token, 32k context)
    {"batch_size": 8, "seq_len": 1, "context_len": 32768, "num_heads": 32, "head_dim": 128, "chunk_size": 8192, "block_size": 16, "max_blocks_per_seq": 2049, "num_blocks": 16500},
    # Decode-heavy: Longformer batched generation (1 token, 4k context)
    {"batch_size": 32, "seq_len": 1, "context_len": 4096, "num_heads": 12, "head_dim": 64, "chunk_size": 512, "block_size": 16, "max_blocks_per_seq": 257, "num_blocks": 8300},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("attention", "9_ChunkedLocalAttention")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    # Pre-projected Q, K, V tensors
    q = DISTRIBUTIONS[dist_name]((p["batch_size"], p["num_heads"], p["seq_len"], p["head_dim"]), dtype=dtype, device=device)
    k = DISTRIBUTIONS[dist_name]((p["batch_size"], p["num_heads"], p["seq_len"], p["head_dim"]), dtype=dtype, device=device)
    v = DISTRIBUTIONS[dist_name]((p["batch_size"], p["num_heads"], p["seq_len"], p["head_dim"]), dtype=dtype, device=device)
    kv_cache_pool = DISTRIBUTIONS[dist_name]((p["num_blocks"], p["block_size"], p["num_heads"], p["head_dim"], 2), dtype=dtype, device=device)
    block_table = torch.randint(0, p["num_blocks"], (p["batch_size"], p["max_blocks_per_seq"]), device=device)
    context_lens = torch.full((p["batch_size"],), p["context_len"], dtype=torch.long, device=device)
    return [q, k, v, kv_cache_pool, block_table, context_lens]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["num_heads"], p["head_dim"], p["chunk_size"], p["block_size"]]
