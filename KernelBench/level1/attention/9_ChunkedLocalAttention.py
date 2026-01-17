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

    Shapes:
        query: (batch_size, seq_len, hidden_size) - current query tokens
        kv_cache_pool: (num_blocks, block_size, num_heads, head_dim, 2) - paged KV pool
        block_table: (batch_size, max_blocks_per_seq) - maps logical to physical blocks
        context_lens: (batch_size,) - number of cached tokens per sequence
        Output: (batch_size, seq_len, hidden_size)
    """

    def __init__(self, hidden_size: int, num_heads: int = 32, chunk_size: int = 512,
                 block_size: int = 16, dropout: float = 0.0):
        """
        Initialize Chunked Local Attention with paged KV cache.

        Args:
            hidden_size: Model hidden dimension
            num_heads: Number of attention heads
            chunk_size: Size of each local attention chunk
            block_size: Number of tokens per cache block/page
            dropout: Attention dropout probability
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.chunk_size = chunk_size
        self.block_size = block_size
        self.dropout = dropout
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

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
        Compute chunked local attention with paged KV cache.

        Args:
            query: Query tensor (batch_size, seq_len, hidden_size)
            kv_cache_pool: Paged KV cache (num_blocks, block_size, num_heads, head_dim, 2)
            block_table: Block table (batch_size, max_blocks_per_seq)
            context_lens: Context lengths (batch_size,)

        Returns:
            Attention output (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = query.shape
        device = query.device

        # Project query
        q = self.q_proj(query)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Gather K, V from paged cache
        k_cache, v_cache = self._gather_kv_from_paged_cache(
            kv_cache_pool, block_table, context_lens
        )

        # Project current tokens for K, V
        k_new = self.k_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v_new = self.v_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Concatenate cached and new K, V
        k = torch.cat([k_cache, k_new], dim=1)
        v = torch.cat([v_cache, v_new], dim=1)

        total_len = k.shape[1]

        # Pad sequence to be divisible by chunk_size
        pad_len = (self.chunk_size - total_len % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, 0, 0, pad_len))

        padded_len = k.shape[1]
        num_chunks = padded_len // self.chunk_size

        # Reshape K, V to chunks
        k = k.view(batch_size, num_chunks, self.chunk_size, self.num_heads, self.head_dim)
        v = v.view(batch_size, num_chunks, self.chunk_size, self.num_heads, self.head_dim)

        # Transpose for attention
        k = k.permute(0, 1, 3, 2, 4)  # (batch, num_chunks, heads, chunk_size, head_dim)
        v = v.permute(0, 1, 3, 2, 4)

        # For queries, we need to determine which chunk each query token attends to
        # based on its absolute position
        query_positions = torch.arange(seq_len, device=device).unsqueeze(0) + context_lens.unsqueeze(1)

        # Compute which chunk each query falls into
        query_chunks = query_positions // self.chunk_size
        query_offsets = query_positions % self.chunk_size

        # Reshape q to match chunk-based attention
        q = q.permute(0, 2, 1, 3)  # (batch, heads, seq_len, head_dim)

        # For each query, compute attention within its chunk and previous chunks
        # For simplicity, we compute attention within the relevant chunk window
        output = torch.zeros(batch_size, self.num_heads, seq_len, self.head_dim, device=device)

        for i in range(seq_len):
            # For each query position, find the relevant chunk
            chunk_idx = query_chunks[0, i].item()  # Assuming uniform context_lens for simplicity
            chunk_idx = min(chunk_idx, num_chunks - 1)

            # Query vector for this position
            q_i = q[:, :, i:i+1, :]  # (batch, heads, 1, head_dim)

            # Get K, V from the current chunk (local attention)
            k_chunk = k[:, chunk_idx, :, :, :]  # (batch, heads, chunk_size, head_dim)
            v_chunk = v[:, chunk_idx, :, :, :]

            # Compute attention scores
            scores = torch.matmul(q_i, k_chunk.transpose(-2, -1)) * self.scale
            # (batch, heads, 1, chunk_size)

            # Causal mask within chunk: can only attend to positions <= query position
            q_pos = query_positions[:, i:i+1]  # (batch, 1)
            chunk_start = chunk_idx * self.chunk_size
            k_positions = torch.arange(self.chunk_size, device=device) + chunk_start

            causal_mask = k_positions.unsqueeze(0) > q_pos
            causal_mask = causal_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, chunk_size)
            scores = scores.masked_fill(causal_mask, float('-inf'))

            # Softmax and apply
            attn_weights = F.softmax(scores, dim=-1)
            if self.dropout > 0 and self.training:
                attn_weights = F.dropout(attn_weights, p=self.dropout)

            out_i = torch.matmul(attn_weights, v_chunk)  # (batch, heads, 1, head_dim)
            output[:, :, i:i+1, :] = out_i

        # Reshape back
        output = output.permute(0, 2, 1, 3).contiguous()  # (batch, seq_len, heads, head_dim)
        output = output.view(batch_size, seq_len, self.hidden_size)

        return self.o_proj(output)


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 4
seq_len = 1
context_len = 4096
hidden_size = 4096
num_heads = 32
chunk_size = 512
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
    return [hidden_size, num_heads, chunk_size, block_size]
