import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Expert Combine
    
    Used by: All MoE architectures
    """
    
    def __init__(self, top_k: int = 2):
        super(Model, self).__init__()
        self.top_k = top_k
    
    def forward(self, expert_outputs: torch.Tensor, dispatch_indices: torch.Tensor,
                routing_weights: torch.Tensor, num_tokens: int) -> torch.Tensor:
        hidden_size = expert_outputs.shape[-1]
        
        output = torch.zeros(num_tokens, hidden_size, device=expert_outputs.device, 
                            dtype=expert_outputs.dtype)
        
        flat_weights = routing_weights.view(-1)
        weighted_outputs = expert_outputs * flat_weights.unsqueeze(-1)
        output.index_add_(0, dispatch_indices, weighted_outputs)
        
        return output


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "hidden_size": 4096, "num_experts": 8, "top_k": 2},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("moe", "4_Expert_Combine")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    num_tokens = p["batch_size"] * p["seq_length"]
    total_dispatched = num_tokens * p["top_k"]
    
    expert_outputs = DISTRIBUTIONS[dist_name]((total_dispatched, p["hidden_size"]), dtype=dtype, device=device)
    dispatch_indices = torch.randint(0, num_tokens, (total_dispatched,), device=device)
    routing_weights = torch.softmax(torch.randn(num_tokens, p["top_k"], device=device), dim=-1)
    
    return [expert_outputs, dispatch_indices, routing_weights, num_tokens]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["top_k"]]
