import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

class Model(nn.Module):
    """
    Tree Attention with Paged KV Cache (Speculative Decoding)

    Used by: EAGLE, Medusa, SpecInfer, Sequoia

    Attention with tree-structured causal mask for parallel verification
    of draft token trees in speculative decoding. Uses paged KV cache
    for efficient memory management of the context.

    The tree structure is defined by parent pointers where each draft token
    points to its parent (either a context position or another draft token).
    A token can attend to:
    - All context tokens
    - Its ancestors in the draft tree (path from root to itself)

    Shapes:
        query: (batch, num_draft_tokens, hidden_size) - draft token queries
        kv_cache_pool: (num_blocks, block_size, num_heads, head_dim, 2) - paged context cache
        block_table: (batch_size, max_blocks_per_seq) - page table mapping
        context_lens: (batch_size,) - number of cached context tokens
        draft_kv: (batch, num_draft_tokens, hidden_size, 2) - draft token K/V
        parent_ids: (num_draft_tokens,) - parent index for each draft token
        Output: (batch, num_draft_tokens, hidden_size)
    """

    def __init__(self, hidden_size: int, num_heads: int, block_size: int = 16):
        """
        Initialize tree attention with paged KV cache.

        Args:
            hidden_size: Model hidden dimension
            num_heads: Number of attention heads
            block_size: Number of tokens per cache block/page
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.block_size = block_size

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def _gather_kv_from_paged_cache(self, kv_cache_pool: torch.Tensor,
                                     block_table: torch.Tensor,
                                     context_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        gathered_blocks = kv_cache_pool[block_table.flatten()]
        gathered_blocks = gathered_blocks.view(
            batch_size, max_blocks, self.block_size, self.num_heads, self.head_dim, 2
        )

        # Reshape to (batch, max_context_len, num_heads, head_dim, 2)
        gathered = gathered_blocks.view(
            batch_size, max_context_len, self.num_heads, self.head_dim, 2
        )

        k_cache = gathered[..., 0]
        v_cache = gathered[..., 1]

        # Mask out positions beyond context_lens
        positions = torch.arange(max_context_len, device=device).unsqueeze(0)
        mask = positions >= context_lens.unsqueeze(1)
        k_cache = k_cache.masked_fill(mask.unsqueeze(-1).unsqueeze(-1), 0)
        v_cache = v_cache.masked_fill(mask.unsqueeze(-1).unsqueeze(-1), 0)

        return k_cache, v_cache

    @staticmethod
    def build_tree_mask(parent_ids: torch.Tensor, context_len: int) -> torch.Tensor:
        """
        Build tree attention mask from parent pointers.

        The tree structure represents draft tokens where each token has a parent.
        Parent IDs encode the tree structure:
        - parent_ids[i] < context_len: token i's parent is a context token
        - parent_ids[i] >= context_len: token i's parent is draft token (parent_ids[i] - context_len)

        A draft token can attend to:
        1. All context tokens (positions 0 to context_len-1)
        2. Its ancestors in the tree (including itself)

        Args:
            parent_ids: (num_draft_tokens,) parent index for each draft token
                        Values in range [0, context_len + num_draft_tokens)
                        where values >= context_len refer to draft tokens
            context_len: Number of context tokens

        Returns:
            tree_mask: (num_draft_tokens, context_len + num_draft_tokens)
                       1 = can attend, 0 = blocked
        """
        num_draft = parent_ids.shape[0]
        total_len = context_len + num_draft
        device = parent_ids.device

        # Start with mask allowing all context tokens
        tree_mask = torch.zeros(num_draft, total_len, device=device)
        tree_mask[:, :context_len] = 1.0  # All draft tokens attend to full context

        # For each draft token, trace its path to root and mark ancestors
        for i in range(num_draft):
            # Each token can attend to itself
            tree_mask[i, context_len + i] = 1.0

            # Trace ancestors
            current = parent_ids[i].item()
            while current >= context_len:
                # This is a draft token ancestor
                ancestor_idx = current - context_len
                tree_mask[i, current] = 1.0
                # Move to the ancestor's parent
                current = parent_ids[ancestor_idx].item()

        return tree_mask

    @staticmethod
    def build_tree_mask_batched(parent_ids: torch.Tensor, context_len: int) -> torch.Tensor:
        """
        Build tree attention mask from parent pointers (vectorized version).

        Uses iterative ancestor traversal with matrix operations for efficiency.

        Args:
            parent_ids: (num_draft_tokens,) parent index for each draft token
            context_len: Number of context tokens

        Returns:
            tree_mask: (num_draft_tokens, context_len + num_draft_tokens)
        """
        num_draft = parent_ids.shape[0]
        total_len = context_len + num_draft
        device = parent_ids.device

        # Initialize mask: attend to all context + self
        tree_mask = torch.zeros(num_draft, total_len, device=device)
        tree_mask[:, :context_len] = 1.0  # Attend to all context

        # Self-attention for draft tokens
        draft_indices = torch.arange(num_draft, device=device)
        tree_mask[draft_indices, context_len + draft_indices] = 1.0

        # Iteratively mark ancestors (max depth = num_draft in worst case)
        # current_parents[i] = current ancestor being traced for draft token i
        current_parents = parent_ids.clone()
        active = current_parents >= context_len  # Still in draft token range

        for _ in range(num_draft):
            if not active.any():
                break

            # For active traces, mark the current ancestor
            active_indices = torch.where(active)[0]
            ancestor_positions = current_parents[active_indices]
            tree_mask[active_indices, ancestor_positions] = 1.0

            # Move to the next ancestor
            ancestor_draft_indices = ancestor_positions - context_len
            current_parents[active_indices] = parent_ids[ancestor_draft_indices]

            # Update active status
            active = current_parents >= context_len

        return tree_mask

    def forward(self, query: torch.Tensor, kv_cache_pool: torch.Tensor,
                block_table: torch.Tensor, context_lens: torch.Tensor,
                draft_hidden: torch.Tensor, parent_ids: torch.Tensor) -> torch.Tensor:
        """
        Tree attention forward pass with paged KV cache.

        Args:
            query: Draft token queries (batch, num_draft, hidden_size)
            kv_cache_pool: Paged KV cache (num_blocks, block_size, num_heads, head_dim, 2)
            block_table: Block table (batch_size, max_blocks_per_seq)
            context_lens: Context lengths (batch_size,)
            draft_hidden: Draft token hidden states (batch, num_draft, hidden_size)
            parent_ids: Parent indices for tree structure (num_draft,)

        Returns:
            Output tensor (batch, num_draft, hidden_size)
        """
        batch_size, num_draft, _ = query.shape
        device = query.device
        max_context_len = block_table.shape[1] * self.block_size

        # Project query
        q = self.q_proj(query)
        q = q.view(batch_size, num_draft, self.num_heads, self.head_dim).transpose(1, 2)

        # Gather K, V from paged cache for context
        k_cache, v_cache = self._gather_kv_from_paged_cache(
            kv_cache_pool, block_table, context_lens
        )
        # k_cache, v_cache: (batch, max_context_len, num_heads, head_dim)

        # Project draft tokens for K, V
        k_draft = self.k_proj(draft_hidden).view(batch_size, num_draft, self.num_heads, self.head_dim)
        v_draft = self.v_proj(draft_hidden).view(batch_size, num_draft, self.num_heads, self.head_dim)

        # Concatenate context and draft K, V
        k = torch.cat([k_cache, k_draft], dim=1).transpose(1, 2)
        v = torch.cat([v_cache, v_draft], dim=1).transpose(1, 2)
        # (batch, num_heads, max_context_len + num_draft, head_dim)

        total_len = k.shape[2]

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # (batch, num_heads, num_draft, total_len)

        # Build tree mask from parent pointers
        # Note: For batch processing, we assume same tree structure across batch
        # Use max context length for mask building
        tree_mask = self.build_tree_mask_batched(parent_ids, max_context_len)
        # tree_mask: (num_draft, total_len)

        # Apply tree mask (1 = attend, 0 = block)
        mask = tree_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, num_draft, total_len)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # Also mask out positions beyond actual context length per sequence
        # Create per-batch context mask
        key_positions = torch.arange(max_context_len, device=device).unsqueeze(0)
        context_mask = key_positions >= context_lens.unsqueeze(1)
        # Expand to (batch, 1, 1, max_context_len)
        context_mask = context_mask.unsqueeze(1).unsqueeze(2)
        # Zero pad for draft positions
        draft_mask = torch.zeros(batch_size, 1, 1, num_draft, device=device, dtype=torch.bool)
        full_context_mask = torch.cat([context_mask, draft_mask], dim=-1)
        scores = scores.masked_fill(full_context_mask, float('-inf'))

        # Softmax and apply to values
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, num_draft, self.hidden_size
        )

        return self.o_proj(attn_output)


def generate_tree_structure(num_draft: int, branching_factor: int = 2,
                            context_len: int = 2048, device: str = 'cuda') -> torch.Tensor:
    """
    Generate a realistic tree structure for speculative decoding.

    Creates a tree where:
    - Root draft tokens have parents in the context (last context position)
    - Other draft tokens have parents among earlier draft tokens
    - Tree has specified branching factor

    This mimics real speculative decoding trees (Medusa, EAGLE, Sequoia).

    Args:
        num_draft: Number of draft tokens in the tree
        branching_factor: Average number of children per node
        context_len: Number of context tokens
        device: Device for tensor

    Returns:
        parent_ids: (num_draft,) where parent_ids[i] is the parent of token i
    """
    parent_ids = torch.zeros(num_draft, dtype=torch.long, device=device)

    # First token's parent is the last context position
    parent_ids[0] = context_len - 1

    # Build tree level by level
    current_level_start = 0
    current_level_end = 1
    next_token_idx = 1

    while next_token_idx < num_draft:
        # For each token in current level, add children
        for parent_idx in range(current_level_start, current_level_end):
            for _ in range(branching_factor):
                if next_token_idx >= num_draft:
                    break
                # Parent is the draft token at parent_idx
                parent_ids[next_token_idx] = context_len + parent_idx
                next_token_idx += 1

        # Move to next level
        current_level_start = current_level_end
        current_level_end = next_token_idx

    return parent_ids


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 8, "context_len": 2048, "num_draft_tokens": 63, "hidden_size": 4096, "num_heads": 32, "block_size": 16, "branching_factor": 2, "max_blocks_per_seq": (context_len + block_size - 1) // block_size, "num_blocks": batch_size * max_blocks_per_seq + 64},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("speculative", "1_TreeAttention")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    query = DISTRIBUTIONS[dist_name]((p["batch_size"], p["num_draft_tokens"], p["hidden_size"]), dtype=dtype, device=device)
    draft_hidden = DISTRIBUTIONS[dist_name]((p["batch_size"], p["num_draft_tokens"], p["hidden_size"]), dtype=dtype, device=device)
    kv_cache_pool = DISTRIBUTIONS[dist_name]((p["num_blocks"], p["block_size"], p["num_heads"], p["hidden_size"] // p["num_heads"], 2), dtype=dtype, device=device)
    block_table = DISTRIBUTIONS[dist_name]((p["batch_size"], p["max_blocks_per_seq"]), dtype=dtype, device=device)
    return [query, kv_cache_pool, block_table, context_lens, draft_hidden, parent_ids]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["hidden_size"], p["num_heads"], p["block_size"]]
