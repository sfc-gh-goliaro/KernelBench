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
    Multi-Head Self-Attention with Causal Mask and Paged KV Cache

    Used by: All decoder-only LLMs (GPT, Llama, Mistral, etc.)

    Standard multi-head attention with lower-triangular causal mask for
    autoregressive generation. Uses paged KV cache where cache entries
    are stored in non-contiguous blocks accessed via a page table.

    Shapes:
        query: (batch_size, seq_len, hidden_size) - current query tokens
        kv_cache_pool: (num_blocks, block_size, num_kv_heads, head_dim, 2) - paged KV pool
        block_table: (batch_size, max_blocks_per_seq) - maps logical to physical blocks
        context_lens: (batch_size,) - number of cached tokens per sequence
        Output: (batch_size, seq_len, hidden_size)
    """

    def __init__(self, hidden_size: int, num_heads: int, block_size: int = 16,
                 dropout: float = 0.0):
        """
        Initialize causal multi-head attention with paged KV cache.

        Args:
            hidden_size: Model hidden dimension
            num_heads: Number of attention heads
            block_size: Number of tokens per cache block/page
            dropout: Attention dropout probability
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.block_size = block_size
        self.dropout = dropout

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

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

        # Gather blocks for each sequence
        # block_table: (batch, max_blocks) -> indices into kv_cache_pool
        gathered_blocks = kv_cache_pool[block_table.flatten()]
        # (batch * max_blocks, block_size, num_heads, head_dim, 2)

        gathered_blocks = gathered_blocks.view(
            batch_size, max_blocks, self.block_size, self.num_heads, self.head_dim, 2
        )

        # Reshape to (batch, max_context_len, num_heads, head_dim, 2)
        gathered = gathered_blocks.view(
            batch_size, max_context_len, self.num_heads, self.head_dim, 2
        )

        k_cache = gathered[..., 0]  # (batch, max_context, heads, head_dim)
        v_cache = gathered[..., 1]

        # Mask out positions beyond context_lens
        positions = torch.arange(max_context_len, device=device).unsqueeze(0)
        mask = positions >= context_lens.unsqueeze(1)
        k_cache = k_cache.masked_fill(mask.unsqueeze(-1).unsqueeze(-1), 0)
        v_cache = v_cache.masked_fill(mask.unsqueeze(-1).unsqueeze(-1), 0)

        return k_cache, v_cache

    def forward(self, query: torch.Tensor, kv_cache_pool: torch.Tensor,
                block_table: torch.Tensor, context_lens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with paged KV cache attention.

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
        # (batch, num_heads, seq_len, head_dim)

        # Gather K, V from paged cache
        k_cache, v_cache = self._gather_kv_from_paged_cache(
            kv_cache_pool, block_table, context_lens
        )

        # Also project current tokens for K, V and append
        k_new = self.k_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v_new = self.v_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Concatenate cached and new K, V
        k = torch.cat([k_cache, k_new], dim=1).transpose(1, 2)
        v = torch.cat([v_cache, v_new], dim=1).transpose(1, 2)
        # (batch, num_heads, total_len, head_dim)

        total_len = k.shape[2]

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Create causal mask: query positions can only attend to previous positions
        # Query at position i (relative to new tokens) attends to positions 0..context_len+i
        query_positions = torch.arange(seq_len, device=device).unsqueeze(0) + context_lens.unsqueeze(1)
        key_positions = torch.arange(total_len, device=device).unsqueeze(0)
        causal_mask = key_positions > query_positions.unsqueeze(-1)
        causal_mask = causal_mask.unsqueeze(1)  # (batch, 1, seq_len, total_len)

        scores = scores.masked_fill(causal_mask, float('-inf'))

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
    # Prefill-heavy: Llama-3.1-8B initial prompt processing (2048 tokens)
    {"batch_size": 4, "seq_len": 2048, "context_len": 0, "hidden_size": 4096, "num_heads": 32, "block_size": 16, "max_blocks_per_seq": 129, "num_blocks": 1096},
    # Prefill-heavy: Llama-3.1-8B chunked prefill (512 token chunks)
    {"batch_size": 8, "seq_len": 512, "context_len": 1024, "hidden_size": 4096, "num_heads": 32, "block_size": 16, "max_blocks_per_seq": 97, "num_blocks": 820},
    # Decode-heavy: Llama-3.1-8B autoregressive generation (1 token, 2k context)
    {"batch_size": 32, "seq_len": 1, "context_len": 2048, "hidden_size": 4096, "num_heads": 32, "block_size": 16, "max_blocks_per_seq": 129, "num_blocks": 4200},
    # Decode-heavy: Llama-3.1-8B long context decoding (1 token, 8k context)
    {"batch_size": 16, "seq_len": 1, "context_len": 8192, "hidden_size": 4096, "num_heads": 32, "block_size": 16, "max_blocks_per_seq": 513, "num_blocks": 8300},
    # Decode-heavy: Llama-3.1-70B batched generation (1 token, 4k context)
    {"batch_size": 8, "seq_len": 1, "context_len": 4096, "hidden_size": 8192, "num_heads": 64, "block_size": 16, "max_blocks_per_seq": 257, "num_blocks": 2120},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("attention", "2_MultiHeadAttention_Causal")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    query = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_len"], p["hidden_size"]), dtype=dtype, device=device)
    kv_cache_pool = DISTRIBUTIONS[dist_name]((p["num_blocks"], p["block_size"], p["num_heads"], p["hidden_size"] // p["num_heads"], 2), dtype=dtype, device=device)
    block_table = DISTRIBUTIONS[dist_name]((p["batch_size"], p["max_blocks_per_seq"]), dtype=dtype, device=device)
    return [query, kv_cache_pool, block_table, context_lens]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["hidden_size"], p["num_heads"], p["block_size"]]
