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
    
    NOTE: This operator takes pre-projected Q and pre-compressed KV latent.
    The latent-to-KV up-projections are done inside this operator as they
    are integral to the MLA attention mechanism.

    Shapes:
        q: (batch_size, num_heads, seq_len, head_dim) - projected query
        kv_latent: (batch_size, seq_len, kv_lora_rank) - compressed KV latent (new tokens)
        latent_cache_pool: (num_blocks, block_size, kv_lora_rank) - paged latent cache
        block_table: (batch_size, max_blocks_per_seq) - maps logical to physical blocks
        context_lens: (batch_size,) - number of cached tokens per sequence
        Output: (batch_size, num_heads, seq_len, head_dim)
    """

    def __init__(self, num_heads: int, head_dim: int, kv_lora_rank: int = 512,
                 block_size: int = 16, dropout: float = 0.0):
        """
        Initialize MLA with paged latent cache.

        Args:
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            kv_lora_rank: Rank for KV compression (stored in cache)
            block_size: Number of tokens per cache block/page
            dropout: Attention dropout probability
        """
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.kv_lora_rank = kv_lora_rank
        self.block_size = block_size
        self.dropout = dropout

        # KV up-projections: latent -> K and V (integral to MLA mechanism)
        self.k_up_proj = nn.Linear(kv_lora_rank, num_heads * head_dim, bias=False)
        self.v_up_proj = nn.Linear(kv_lora_rank, num_heads * head_dim, bias=False)

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

    def forward(self, q: torch.Tensor, kv_latent: torch.Tensor,
                latent_cache_pool: torch.Tensor, block_table: torch.Tensor,
                context_lens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with paged latent cache and MLA.

        Args:
            q: Projected query (batch_size, num_heads, seq_len, head_dim)
            kv_latent: Compressed KV latent for new tokens (batch_size, seq_len, kv_lora_rank)
            latent_cache_pool: Paged latent cache (num_blocks, block_size, kv_lora_rank)
            block_table: Block table (batch_size, max_blocks_per_seq)
            context_lens: Context lengths (batch_size,)

        Returns:
            Output tensor (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        device = q.device

        # Gather latent from paged cache
        latent_cache = self._gather_latent_from_paged_cache(
            latent_cache_pool, block_table, context_lens
        )

        # Concatenate cached and new latent
        latent = torch.cat([latent_cache, kv_latent], dim=1)

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

        return attn_output  # (batch, heads, seq_len, head_dim)


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    # Prefill-heavy: DeepSeek-V2 initial prompt processing (2048 tokens)
    {"batch_size": 4, "seq_len": 2048, "context_len": 0, "num_heads": 32, "head_dim": 128, "kv_lora_rank": 512, "block_size": 16, "max_blocks_per_seq": 129, "num_blocks": 520},
    # Prefill-heavy: DeepSeek-V2 chunked prefill (1024 token chunks)
    {"batch_size": 8, "seq_len": 1024, "context_len": 2048, "num_heads": 32, "head_dim": 128, "kv_lora_rank": 512, "block_size": 16, "max_blocks_per_seq": 193, "num_blocks": 1560},
    # Decode-heavy: DeepSeek-V2 high-throughput decoding (1 token, 4k context)
    {"batch_size": 64, "seq_len": 1, "context_len": 4096, "num_heads": 32, "head_dim": 128, "kv_lora_rank": 512, "block_size": 16, "max_blocks_per_seq": 257, "num_blocks": 16500},
    # Decode-heavy: DeepSeek-V2-Lite batched generation (1 token, 8k context)
    {"batch_size": 32, "seq_len": 1, "context_len": 8192, "num_heads": 16, "head_dim": 128, "kv_lora_rank": 512, "block_size": 16, "max_blocks_per_seq": 513, "num_blocks": 16500},
    # Decode-heavy: DeepSeek-V3 long context decoding (1 token, 16k context)
    {"batch_size": 8, "seq_len": 1, "context_len": 16384, "num_heads": 32, "head_dim": 128, "kv_lora_rank": 512, "block_size": 16, "max_blocks_per_seq": 1025, "num_blocks": 8300},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("attention", "5_MultiHeadLatentAttention")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    # Pre-projected Q and pre-compressed KV latent
    q = DISTRIBUTIONS[dist_name]((p["batch_size"], p["num_heads"], p["seq_len"], p["head_dim"]), dtype=dtype, device=device)
    kv_latent = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_len"], p["kv_lora_rank"]), dtype=dtype, device=device)
    latent_cache_pool = DISTRIBUTIONS[dist_name]((p["num_blocks"], p["block_size"], p["kv_lora_rank"]), dtype=dtype, device=device)
    block_table = torch.randint(0, p["num_blocks"], (p["batch_size"], p["max_blocks_per_seq"]), device=device)
    context_lens = torch.full((p["batch_size"],), p["context_len"], dtype=torch.long, device=device)
    return [q, kv_latent, latent_cache_pool, block_table, context_lens]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["num_heads"], p["head_dim"], p["kv_lora_rank"]]
