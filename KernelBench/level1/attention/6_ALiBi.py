import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Attention with Linear Biases (ALiBi) with Paged KV Cache

    Used by: BLOOM, MPT

    ALiBi adds a linear bias to attention scores based on the distance
    between query and key positions. This provides position information
    without explicit position embeddings and extrapolates well to longer
    sequences. Uses paged KV cache for efficient memory management.

    Shapes:
        query: (batch_size, seq_len, hidden_size) - current query tokens
        kv_cache_pool: (num_blocks, block_size, num_heads, head_dim, 2) - paged KV pool
        block_table: (batch_size, max_blocks_per_seq) - maps logical to physical blocks
        context_lens: (batch_size,) - number of cached tokens per sequence
        Output: (batch_size, seq_len, hidden_size)
    """

    def __init__(self, hidden_size: int, num_heads: int, block_size: int = 16,
                 dropout: float = 0.0):
        """
        Initialize ALiBi attention with paged KV cache.

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

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Compute ALiBi slopes for each head
        slopes = self._get_alibi_slopes(num_heads)
        self.register_buffer('slopes', slopes)

    def _get_alibi_slopes(self, num_heads: int) -> torch.Tensor:
        """Compute ALiBi slopes for each head."""
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

    def _get_alibi_bias(self, query_positions: torch.Tensor, key_positions: torch.Tensor,
                        device: torch.device) -> torch.Tensor:
        """
        Compute ALiBi bias for query attending to key positions.

        Args:
            query_positions: (batch_size, seq_len) absolute positions of queries
            key_positions: (1, total_len) positions of keys

        Returns:
            alibi: (batch_size, num_heads, seq_len, total_len)
        """
        # Distance: query_pos - key_pos (negative for keys before query)
        distance = query_positions.unsqueeze(-1) - key_positions.unsqueeze(1)
        # (batch, seq_len, total_len)

        # Apply slopes: (num_heads, 1, 1) * (batch, 1, seq_len, total_len)
        alibi = self.slopes.to(device).view(1, self.num_heads, 1, 1) * distance.unsqueeze(1)

        return alibi  # (batch, num_heads, seq_len, total_len)

    def forward(self, query: torch.Tensor, kv_cache_pool: torch.Tensor,
                block_table: torch.Tensor, context_lens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with paged KV cache and ALiBi attention.

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

        # Compute query and key positions for ALiBi
        query_positions = torch.arange(seq_len, device=device).unsqueeze(0) + context_lens.unsqueeze(1)
        key_positions = torch.arange(total_len, device=device).unsqueeze(0)

        # Add ALiBi bias
        alibi_bias = self._get_alibi_bias(query_positions, key_positions, device)
        scores = scores + alibi_bias

        # Causal mask
        causal_mask = key_positions > query_positions.unsqueeze(-1)
        causal_mask = causal_mask.unsqueeze(1)
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

batch_size = 8
seq_len = 1
context_len = 2048
hidden_size = 4096
num_heads = 32
block_size = 16
max_blocks_per_seq = (context_len + block_size - 1) // block_size
num_blocks = batch_size * max_blocks_per_seq + 64

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    query = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
    head_dim = hidden_size // num_heads

    kv_cache_pool = torch.randn(
        num_blocks, block_size, num_heads, head_dim, 2, device='cuda'
    )

    block_table = torch.zeros(batch_size, max_blocks_per_seq, dtype=torch.long, device='cuda')
    for b in range(batch_size):
        block_table[b] = torch.arange(max_blocks_per_seq, device='cuda') + b * max_blocks_per_seq

    context_lens = torch.full((batch_size,), context_len, dtype=torch.long, device='cuda')

    return [query, kv_cache_pool, block_table, context_lens]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [hidden_size, num_heads, block_size]
