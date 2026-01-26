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
    Grouped-Query Attention (GQA) with Paged KV Cache

    Used by: Llama-2 70B+, Llama-3, Mistral, Qwen-2

    GQA reduces memory and compute by having multiple query heads share
    the same key/value heads. Uses paged KV cache where cache entries
    are stored in non-contiguous blocks accessed via a page table.
    
    NOTE: Q/K/V projections are done separately using Linear operators.
    This operator takes pre-projected Q, K, V tensors.

    Shapes:
        q: (batch_size, num_heads, seq_len, head_dim) - projected query
        k: (batch_size, num_kv_heads, seq_len, head_dim) - projected key (new tokens)
        v: (batch_size, num_kv_heads, seq_len, head_dim) - projected value (new tokens)
        kv_cache_pool: (num_blocks, block_size, num_kv_heads, head_dim, 2) - paged KV pool
        block_table: (batch_size, max_blocks_per_seq) - maps logical to physical blocks
        context_lens: (batch_size,) - number of cached tokens per sequence
        Output: (batch_size, num_heads, seq_len, head_dim)
    """

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int,
                 block_size: int = 16, dropout: float = 0.0):
        """
        Initialize GQA with paged KV cache.

        Args:
            num_heads: Number of query heads
            num_kv_heads: Number of key/value heads (must divide num_heads)
            head_dim: Dimension of each attention head
            block_size: Number of tokens per cache block/page
            dropout: Attention dropout probability
        """
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_heads // num_kv_heads
        self.block_size = block_size
        self.dropout = dropout

        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def _gather_kv_from_paged_cache(self, kv_cache_pool: torch.Tensor,
                                     block_table: torch.Tensor,
                                     context_lens: torch.Tensor) -> tuple:
        """
        Gather K and V tensors from paged cache using block table.

        Args:
            kv_cache_pool: (num_blocks, block_size, num_kv_heads, head_dim, 2)
            block_table: (batch_size, max_blocks_per_seq)
            context_lens: (batch_size,)

        Returns:
            k_cache: (batch_size, num_kv_heads, max_context_len, head_dim)
            v_cache: (batch_size, num_kv_heads, max_context_len, head_dim)
        """
        batch_size = block_table.shape[0]
        max_blocks = block_table.shape[1]
        max_context_len = max_blocks * self.block_size
        device = kv_cache_pool.device

        gathered_blocks = kv_cache_pool[block_table.flatten()]
        gathered_blocks = gathered_blocks.view(
            batch_size, max_blocks, self.block_size, self.num_kv_heads, self.head_dim, 2
        )
        gathered = gathered_blocks.view(
            batch_size, max_context_len, self.num_kv_heads, self.head_dim, 2
        )

        k_cache = gathered[..., 0].transpose(1, 2)  # (batch, kv_heads, context, head_dim)
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
        Forward pass with paged KV cache and grouped-query attention.

        Args:
            q: Projected query (batch_size, num_heads, seq_len, head_dim)
            k: Projected key for new tokens (batch_size, num_kv_heads, seq_len, head_dim)
            v: Projected value for new tokens (batch_size, num_kv_heads, seq_len, head_dim)
            kv_cache_pool: Paged KV cache (num_blocks, block_size, num_kv_heads, head_dim, 2)
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
        k_full = torch.cat([k_cache, k], dim=2)  # (batch, kv_heads, total_len, head_dim)
        v_full = torch.cat([v_cache, v], dim=2)

        # Repeat K, V to match number of query heads
        k_full = k_full.repeat_interleave(self.num_key_value_groups, dim=1)
        v_full = v_full.repeat_interleave(self.num_key_value_groups, dim=1)

        total_len = k_full.shape[2]

        # Compute attention scores
        scores = torch.matmul(q, k_full.transpose(-2, -1)) * self.scale

        # Causal mask
        query_positions = torch.arange(seq_len, device=device).unsqueeze(0) + context_lens.unsqueeze(1)
        key_positions = torch.arange(total_len, device=device).unsqueeze(0)
        causal_mask = key_positions > query_positions.unsqueeze(-1)
        causal_mask = causal_mask.unsqueeze(1)

        scores = scores.masked_fill(causal_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        if self.dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        attn_output = torch.matmul(attn_weights, v_full)

        return attn_output  # (batch, heads, seq_len, head_dim)


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    # Prefill-heavy: Llama-3.1-8B initial prompt processing (2048 tokens)
    {"batch_size": 4, "seq_len": 2048, "context_len": 0, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128, "block_size": 16, "max_blocks_per_seq": 129, "num_blocks": 520},
    # Prefill-heavy: Llama-3.1-8B chunked prefill (1024 token chunks)
    {"batch_size": 8, "seq_len": 1024, "context_len": 2048, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128, "block_size": 16, "max_blocks_per_seq": 193, "num_blocks": 1560},
    # Decode-heavy: Llama-3.1-8B high-throughput decoding (1 token, 2k context)
    {"batch_size": 64, "seq_len": 1, "context_len": 2048, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128, "block_size": 16, "max_blocks_per_seq": 129, "num_blocks": 8300},
    # Decode-heavy: Llama-3.1-70B batched generation (1 token, 4k context)
    {"batch_size": 8, "seq_len": 1, "context_len": 4096, "num_heads": 64, "num_kv_heads": 8, "head_dim": 128, "block_size": 16, "max_blocks_per_seq": 257, "num_blocks": 2120},
    # Decode-heavy: Mistral-7B long context decoding (1 token, 8k context)
    {"batch_size": 16, "seq_len": 1, "context_len": 8192, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128, "block_size": 16, "max_blocks_per_seq": 513, "num_blocks": 8300},
    # Prefill-heavy: Qwen2-VL-7B multimodal prefill (image+text, 4k tokens)
    {"batch_size": 2, "seq_len": 4096, "context_len": 0, "num_heads": 28, "num_kv_heads": 4, "head_dim": 128, "block_size": 16, "max_blocks_per_seq": 257, "num_blocks": 520},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("attention", "3_GroupedQueryAttention")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    # Pre-projected Q, K, V tensors
    q = DISTRIBUTIONS[dist_name]((p["batch_size"], p["num_heads"], p["seq_len"], p["head_dim"]), dtype=dtype, device=device)
    k = DISTRIBUTIONS[dist_name]((p["batch_size"], p["num_kv_heads"], p["seq_len"], p["head_dim"]), dtype=dtype, device=device)
    v = DISTRIBUTIONS[dist_name]((p["batch_size"], p["num_kv_heads"], p["seq_len"], p["head_dim"]), dtype=dtype, device=device)
    kv_cache_pool = DISTRIBUTIONS[dist_name]((p["num_blocks"], p["block_size"], p["num_kv_heads"], p["head_dim"], 2), dtype=dtype, device=device)
    block_table = torch.randint(0, p["num_blocks"], (p["batch_size"], p["max_blocks_per_seq"]), device=device)
    context_lens = torch.full((p["batch_size"],), p["context_len"], dtype=torch.long, device=device)
    return [q, k, v, kv_cache_pool, block_table, context_lens]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["num_heads"], p["num_kv_heads"], p["head_dim"], p["block_size"]]
