import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Expert Dispatch
    
    Used by: All MoE architectures
    """
    
    def __init__(self, num_experts: int, top_k: int = 2):
        super(Model, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
    
    def forward(self, x: torch.Tensor, expert_indices: torch.Tensor) -> tuple:
        num_tokens = x.shape[0]
        
        flat_expert_indices = expert_indices.view(-1)
        token_indices = torch.arange(num_tokens, device=x.device).unsqueeze(1)
        token_indices = token_indices.expand(-1, self.top_k).reshape(-1)
        
        sorted_expert_indices, sort_order = flat_expert_indices.sort()
        dispatch_indices = token_indices[sort_order]
        dispatched_inputs = x[dispatch_indices]
        expert_counts = torch.bincount(sorted_expert_indices, minlength=self.num_experts)
        
        return dispatched_inputs, dispatch_indices, expert_counts


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "hidden_size": 4096, "num_experts": 8, "top_k": 2},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("moe", "2_Expert_Dispatch")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    num_tokens = p["batch_size"] * p["seq_length"]
    shape = (num_tokens, p["hidden_size"])
    x = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    expert_indices = torch.randint(0, p["num_experts"], (num_tokens, p["top_k"]), device=device)
    return [x, expert_indices]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["num_experts"], p["top_k"]]
