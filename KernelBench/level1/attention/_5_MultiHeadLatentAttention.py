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
    Multi-Head Latent Attention (MLA) with Paged KV Cache

    Used by: DeepSeek-V2, DeepSeek-V3

    MLA compresses KV into a low-rank latent space before caching,
    reducing KV cache memory while maintaining model quality.
    Uses paged cache where the compressed latent is stored in blocks.

    Shapes:
        query: (batch_size, seq_len, hidden_size) - current query tokens
        latent_cache_pool: (num_blocks, block_size, kv_lora_rank) - paged latent cache
        block_table: (batch_size, max_blocks_per_seq) - maps logical to physical blocks
        context_lens: (batch_size,) - number of cached tokens per sequence
        Output: (batch_size, seq_len, hidden_size)
    """

    def __init__(self, hidden_size: int, num_heads: int, kv_lora_rank: int = 512,
                 q_lora_rank: int = None, block_size: int = 16, dropout: float = 0.0):
        """
        Initialize MLA with paged latent cache.

        Args:
            hidden_size: Model hidden dimension
            num_heads: Number of attention heads
            kv_lora_rank: Rank for KV compression (stored in cache)
            q_lora_rank: Rank for Q compression (optional)
            block_size: Number of tokens per cache block/page
            dropout: Attention dropout probability
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.block_size = block_size
        self.dropout = dropout

        # Q projection (optionally compressed)
        if q_lora_rank is not None:
            self.q_down_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
            self.q_up_proj = nn.Linear(q_lora_rank, num_heads * self.head_dim, bias=False)
        else:
            self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)

        # KV compression: project to low-rank latent, then back up
        self.kv_down_proj = nn.Linear(hidden_size, kv_lora_rank, bias=False)
        self.k_up_proj = nn.Linear(kv_lora_rank, num_heads * self.head_dim, bias=False)
        self.v_up_proj = nn.Linear(kv_lora_rank, num_heads * self.head_dim, bias=False)

        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def _gather_latent_from_paged_cache(self, latent_cache_pool: torch.Tensor,
                                         block_table: torch.Tensor,
                                         context_lens: torch.Tensor) -> torch.Tensor:
        """
        Gather compressed latent from paged cache using block table.

        Args:
            latent_cache_pool: (num_blocks, block_size, kv_lora_rank)
            block_table: (batch_size, max_blocks_per_seq)
            context_lens: (batch_size,)

        Returns:
            latent_cache: (batch_size, max_context_len, kv_lora_rank)
        """
        batch_size = block_table.shape[0]
        max_blocks = block_table.shape[1]
        max_context_len = max_blocks * self.block_size
        device = latent_cache_pool.device

        gathered_blocks = latent_cache_pool[block_table.flatten()]
        gathered_blocks = gathered_blocks.view(
            batch_size, max_blocks, self.block_size, self.kv_lora_rank
        )
        latent_cache = gathered_blocks.view(
            batch_size, max_context_len, self.kv_lora_rank
        )

        positions = torch.arange(max_context_len, device=device).unsqueeze(0)
        mask = positions >= context_lens.unsqueeze(1)
        latent_cache = latent_cache.masked_fill(mask.unsqueeze(-1), 0)

        return latent_cache

    def forward(self, query: torch.Tensor, latent_cache_pool: torch.Tensor,
                block_table: torch.Tensor, context_lens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with paged latent cache and MLA.

        Args:
            query: Query tensor (batch_size, seq_len, hidden_size)
            latent_cache_pool: Paged latent cache (num_blocks, block_size, kv_lora_rank)
            block_table: Block table (batch_size, max_blocks_per_seq)
            context_lens: Context lengths (batch_size,)

        Returns:
            Output tensor (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = query.shape
        device = query.device

        # Compute Q
        if self.q_lora_rank is not None:
            q = self.q_up_proj(self.q_down_proj(query))
        else:
            q = self.q_proj(query)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Gather latent from paged cache
        latent_cache = self._gather_latent_from_paged_cache(
            latent_cache_pool, block_table, context_lens
        )

        # Compress current tokens to latent
        latent_new = self.kv_down_proj(query)

        # Concatenate cached and new latent
        latent = torch.cat([latent_cache, latent_new], dim=1)

        # Project latent back to K and V
        k = self.k_up_proj(latent)
        v = self.v_up_proj(latent)

        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        total_len = k.shape[2]

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask
        query_positions = torch.arange(seq_len, device=device).unsqueeze(0) + context_lens.unsqueeze(1)
        key_positions = torch.arange(total_len, device=device).unsqueeze(0)
        causal_mask = key_positions > query_positions.unsqueeze(-1)
        causal_mask = causal_mask.unsqueeze(1)

        scores = scores.masked_fill(causal_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        if self.dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )

        return self.o_proj(attn_output)


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    # Prefill-heavy: DeepSeek-V2 initial prompt processing (2048 tokens)
    {"batch_size": 4, "seq_len": 2048, "context_len": 0, "hidden_size": 4096, "num_heads": 32, "kv_lora_rank": 512, "block_size": 16, "max_blocks_per_seq": 129, "num_blocks": 520},
    # Prefill-heavy: DeepSeek-V2 chunked prefill (1024 token chunks)
    {"batch_size": 8, "seq_len": 1024, "context_len": 2048, "hidden_size": 4096, "num_heads": 32, "kv_lora_rank": 512, "block_size": 16, "max_blocks_per_seq": 193, "num_blocks": 1560},
    # Decode-heavy: DeepSeek-V2 high-throughput decoding (1 token, 4k context)
    {"batch_size": 64, "seq_len": 1, "context_len": 4096, "hidden_size": 4096, "num_heads": 32, "kv_lora_rank": 512, "block_size": 16, "max_blocks_per_seq": 257, "num_blocks": 16500},
    # Decode-heavy: DeepSeek-V2-Lite batched generation (1 token, 8k context)
    {"batch_size": 32, "seq_len": 1, "context_len": 8192, "hidden_size": 2048, "num_heads": 16, "kv_lora_rank": 512, "block_size": 16, "max_blocks_per_seq": 513, "num_blocks": 16500},
    # Decode-heavy: DeepSeek-V3 long context decoding (1 token, 16k context)
    {"batch_size": 8, "seq_len": 1, "context_len": 16384, "hidden_size": 4096, "num_heads": 32, "kv_lora_rank": 512, "block_size": 16, "max_blocks_per_seq": 1025, "num_blocks": 8300},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("attention", "5_MultiHeadLatentAttention")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    query = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_len"], p["hidden_size"]), dtype=dtype, device=device)
    latent_cache_pool = DISTRIBUTIONS[dist_name]((p["num_blocks"], p["block_size"], p["kv_lora_rank"]), dtype=dtype, device=device)
    block_table = DISTRIBUTIONS[dist_name]((p["batch_size"], p["max_blocks_per_seq"]), dtype=dtype, device=device)
    return [query, latent_cache_pool, block_table, context_lens]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["hidden_size"], p["num_heads"], p["kv_lora_rank"]]
