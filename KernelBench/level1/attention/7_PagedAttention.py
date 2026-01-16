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
    Paged Attention
    
    Used by: vLLM, SGLang serving systems
    
    Memory-efficient attention using block tables to manage non-contiguous
    KV cache blocks. Enables efficient memory sharing across sequences
    and dynamic memory allocation.
    
    Shapes:
        query: (batch_size, num_heads, head_dim)  # Single query token
        key_cache: (num_blocks, block_size, num_heads, head_dim)
        value_cache: (num_blocks, block_size, num_heads, head_dim)
        block_tables: (batch_size, max_num_blocks_per_seq)
        context_lens: (batch_size,)
    """
    
    def __init__(self, num_heads: int, head_dim: int, block_size: int = 16, scale: float = None):
        """
        Initialize paged attention.
        
        Args:
            num_heads: Number of attention heads
            head_dim: Dimension of each head
            block_size: Number of tokens per block
            scale: Attention scale (default: 1/sqrt(head_dim))
        """
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)
    
    def forward(self, query: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor,
                block_tables: torch.Tensor, context_lens: torch.Tensor) -> torch.Tensor:
        """
        Paged attention forward pass.
        
        Args:
            query: Query tensor (batch, num_heads, head_dim)
            key_cache: Paged key cache (num_blocks, block_size, num_heads, head_dim)
            value_cache: Paged value cache (num_blocks, block_size, num_heads, head_dim)
            block_tables: Block indices per sequence (batch, max_blocks_per_seq)
            context_lens: Context length per sequence (batch,)
            
        Returns:
            Attention output (batch, num_heads, head_dim)
        """
        batch_size = query.shape[0]
        max_context_len = context_lens.max().item()
        
        outputs = []
        
        for b in range(batch_size):
            ctx_len = context_lens[b].item()
            num_blocks = (ctx_len + self.block_size - 1) // self.block_size
            
            # Gather K and V from paged cache
            k_list = []
            v_list = []
            
            for block_idx in range(num_blocks):
                physical_block = block_tables[b, block_idx].item()
                
                # Determine how many tokens in this block
                if block_idx == num_blocks - 1:
                    # Last block may be partial
                    tokens_in_block = ctx_len - block_idx * self.block_size
                else:
                    tokens_in_block = self.block_size
                
                k_list.append(key_cache[physical_block, :tokens_in_block])
                v_list.append(value_cache[physical_block, :tokens_in_block])
            
            # Concatenate to get full K, V for this sequence
            k = torch.cat(k_list, dim=0)  # (ctx_len, num_heads, head_dim)
            v = torch.cat(v_list, dim=0)  # (ctx_len, num_heads, head_dim)
            
            # Compute attention for this sequence
            q = query[b]  # (num_heads, head_dim)
            
            # Attention scores: (num_heads, 1, head_dim) @ (num_heads, head_dim, ctx_len)
            scores = torch.einsum('hd,shd->hs', q, k) * self.scale  # (num_heads, ctx_len)
            attn_weights = F.softmax(scores, dim=-1)  # (num_heads, ctx_len)
            
            # Apply attention: (num_heads, ctx_len) @ (ctx_len, num_heads, head_dim)
            out = torch.einsum('hs,shd->hd', attn_weights, v)  # (num_heads, head_dim)
            outputs.append(out)
        
        return torch.stack(outputs, dim=0)  # (batch, num_heads, head_dim)


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "num_heads": 32, "head_dim": 128, "block_size": 16, "num_blocks": 256, "max_context_len": 2048},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("attention", "7_PagedAttention")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    
    query = DISTRIBUTIONS[dist_name]((p["batch_size"], p["num_heads"], p["head_dim"]), dtype=dtype, device=device)
    key_cache = DISTRIBUTIONS[dist_name]((p["num_blocks"], p["block_size"], p["num_heads"], p["head_dim"]), dtype=dtype, device=device)
    value_cache = DISTRIBUTIONS[dist_name]((p["num_blocks"], p["block_size"], p["num_heads"], p["head_dim"]), dtype=dtype, device=device)
    
    # Create block tables (each sequence uses consecutive blocks for simplicity)
    max_blocks_per_seq = p["max_context_len"] // p["block_size"]
    block_tables = torch.zeros(p["batch_size"], max_blocks_per_seq, dtype=torch.long, device=device)
    for b in range(p["batch_size"]):
        start_block = b * max_blocks_per_seq
        block_tables[b] = torch.arange(start_block, start_block + max_blocks_per_seq, device=device)
    
    # Random context lengths
    context_lens = torch.randint(p["block_size"], p["max_context_len"] + 1, (p["batch_size"],), device=device)
    
    return [query, key_cache, value_cache, block_tables, context_lens]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["num_heads"], p["head_dim"], p["block_size"]]
