import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Top-K Expert Router
    
    Used by: Mixtral, DeepSeek-MoE, Qwen-MoE, DBRX
    """
    
    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
    
    def forward(self, x: torch.Tensor) -> tuple:
        batch_size, seq_len, _ = x.shape
        
        x_flat = x.view(-1, self.hidden_size)
        router_logits = self.gate(x_flat)
        routing_weights, expert_indices = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        return expert_indices, routing_weights


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "hidden_size": 4096, "num_experts": 8, "top_k": 2},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("moe", "1_TopK_Router")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["seq_length"], p["hidden_size"])
    x = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["hidden_size"], p["num_experts"], p["top_k"]]
