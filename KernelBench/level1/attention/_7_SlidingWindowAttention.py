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
    Sliding Window Attention with Paged KV Cache

    Used by: Mistral, Gemma-2, Longformer

    Local attention where each token only attends to a fixed window of
    nearby tokens. This reduces memory and compute from O(n²) to O(n*w)
    where w is the window size. Uses paged KV cache for efficient memory.

    Shapes:
        query: (batch_size, seq_len, hidden_size) - current query tokens
        kv_cache_pool: (num_blocks, block_size, num_heads, head_dim, 2) - paged KV pool
        block_table: (batch_size, max_blocks_per_seq) - maps logical to physical blocks
        context_lens: (batch_size,) - number of cached tokens per sequence
        Output: (batch_size, seq_len, hidden_size)
    """

    def __init__(self, hidden_size: int, num_heads: int, window_size: int = 4096,
                 block_size: int = 16, dropout: float = 0.0):
        """
        Initialize sliding window attention with paged KV cache.

        Args:
            hidden_size: Model hidden dimension
            num_heads: Number of attention heads
            window_size: Size of attention window
            block_size: Number of tokens per cache block/page
            dropout: Attention dropout probability
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
        self.block_size = block_size
        self.dropout = dropout

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

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
            k_cache: (batch_size, max_context_len, num_heads, head_dim)
            v_cache: (batch_size, max_context_len, num_heads, head_dim)
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

        k_cache = gathered[..., 0]
        v_cache = gathered[..., 1]

        positions = torch.arange(max_context_len, device=device).unsqueeze(0)
        mask = positions >= context_lens.unsqueeze(1)
        k_cache = k_cache.masked_fill(mask.unsqueeze(-1).unsqueeze(-1), 0)
        v_cache = v_cache.masked_fill(mask.unsqueeze(-1).unsqueeze(-1), 0)

        return k_cache, v_cache

    def forward(self, query: torch.Tensor, kv_cache_pool: torch.Tensor,
                block_table: torch.Tensor, context_lens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with paged KV cache and sliding window attention.

        Args:
            query: Query tensor (batch_size, seq_len, hidden_size)
            kv_cache_pool: Paged KV cache (num_blocks, block_size, num_heads, head_dim, 2)
            block_table: Block table (batch_size, max_blocks_per_seq)
            context_lens: Context lengths (batch_size,)

        Returns:
            Output tensor (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = query.shape
        device = query.device

        # Project query
        q = self.q_proj(query)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Gather K, V from paged cache
        k_cache, v_cache = self._gather_kv_from_paged_cache(
            kv_cache_pool, block_table, context_lens
        )

        # Project current tokens for K, V
        k_new = self.k_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v_new = self.v_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Concatenate cached and new K, V
        k = torch.cat([k_cache, k_new], dim=1).transpose(1, 2)
        v = torch.cat([v_cache, v_new], dim=1).transpose(1, 2)

        total_len = k.shape[2]

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Create sliding window causal mask
        # Query at absolute position q_pos can attend to keys at positions
        # max(0, q_pos - window_size + 1) to q_pos
        query_positions = torch.arange(seq_len, device=device).unsqueeze(0) + context_lens.unsqueeze(1)
        key_positions = torch.arange(total_len, device=device).unsqueeze(0)

        # Causal: can't attend to future
        causal_mask = key_positions > query_positions.unsqueeze(-1)

        # Sliding window: can't attend beyond window_size back
        window_start = query_positions - self.window_size + 1
        window_mask = key_positions < window_start.unsqueeze(-1)

        # Combined mask
        full_mask = causal_mask | window_mask
        full_mask = full_mask.unsqueeze(1)  # (batch, 1, seq_len, total_len)

        scores = scores.masked_fill(full_mask, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        if self.dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )

        return self.o_proj(attn_output)


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    # Prefill-heavy: Mistral-7B initial prompt processing (2048 tokens within window)
    {"batch_size": 4, "seq_len": 2048, "context_len": 0, "hidden_size": 4096, "num_heads": 32, "window_size": 4096, "block_size": 16, "max_blocks_per_seq": 129, "num_blocks": 520},
    # Prefill-heavy: Mistral-7B chunked prefill (1024 token chunks)
    {"batch_size": 8, "seq_len": 1024, "context_len": 2048, "hidden_size": 4096, "num_heads": 32, "window_size": 4096, "block_size": 16, "max_blocks_per_seq": 193, "num_blocks": 1560},
    # Decode-heavy: Mistral-7B high-throughput decoding (1 token, 4k context)
    {"batch_size": 64, "seq_len": 1, "context_len": 4096, "hidden_size": 4096, "num_heads": 32, "window_size": 4096, "block_size": 16, "max_blocks_per_seq": 257, "num_blocks": 16500},
    # Decode-heavy: Mixtral-8x7B batched generation (1 token, 8k context)
    {"batch_size": 16, "seq_len": 1, "context_len": 8192, "hidden_size": 4096, "num_heads": 32, "window_size": 4096, "block_size": 16, "max_blocks_per_seq": 513, "num_blocks": 8300},
    # Decode-heavy: Gemma-2-27B long context decoding (1 token, 8k context, window=4096)
    {"batch_size": 8, "seq_len": 1, "context_len": 8192, "hidden_size": 4608, "num_heads": 32, "window_size": 4096, "block_size": 16, "max_blocks_per_seq": 513, "num_blocks": 4200},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("attention", "7_SlidingWindowAttention")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    query = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_len"], p["hidden_size"]), dtype=dtype, device=device)
    kv_cache_pool = DISTRIBUTIONS[dist_name]((p["num_blocks"], p["block_size"], p["num_heads"], head_dim, 2), dtype=dtype, device=device)
    block_table = DISTRIBUTIONS[dist_name]((p["batch_size"], p["max_blocks_per_seq"]), dtype=dtype, device=device)
    return [query, kv_cache_pool, block_table, context_lens]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["hidden_size"], p["num_heads"], p["window_size"], p["block_size"]]
