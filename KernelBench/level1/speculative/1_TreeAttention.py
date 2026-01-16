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
    Tree Attention (Speculative Decoding)
    
    Used by: EAGLE, Medusa, SpecInfer
    """
    
    def __init__(self, hidden_size: int, num_heads: int):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                tree_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_draft, _ = query.shape
        total_len = key.shape[1]
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        q = q.view(batch_size, num_draft, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, total_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, total_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        mask = tree_mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_draft, self.hidden_size)
        
        return self.o_proj(attn_output)


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "context_len": 2048, "num_draft_tokens": 64, "hidden_size": 4096, "num_heads": 32},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("speculative", "1_TreeAttention")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    
    query = DISTRIBUTIONS[dist_name]((p["batch_size"], p["num_draft_tokens"], p["hidden_size"]), dtype=dtype, device=device)
    key = DISTRIBUTIONS[dist_name]((p["batch_size"], p["context_len"] + p["num_draft_tokens"], p["hidden_size"]), dtype=dtype, device=device)
    value = DISTRIBUTIONS[dist_name]((p["batch_size"], p["context_len"] + p["num_draft_tokens"], p["hidden_size"]), dtype=dtype, device=device)
    
    tree_mask = torch.tril(torch.ones(p["num_draft_tokens"], p["context_len"] + p["num_draft_tokens"], device=device))
    tree_mask[:, :p["context_len"]] = 1
    
    return [query, key, value, tree_mask]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["hidden_size"], p["num_heads"]]
