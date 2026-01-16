import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Auxiliary Load Balancing Loss for MoE
    
    Used by: All MoE training
    """
    
    def __init__(self, num_experts: int, aux_loss_coef: float = 0.01):
        super(Model, self).__init__()
        self.num_experts = num_experts
        self.aux_loss_coef = aux_loss_coef
    
    def forward(self, router_logits: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        num_tokens = router_logits.shape[0]
        
        routing_probs = F.softmax(router_logits, dim=-1)
        
        expert_counts = torch.zeros(self.num_experts, device=router_logits.device)
        for expert_idx in range(self.num_experts):
            expert_counts[expert_idx] = (expert_indices == expert_idx).sum().float()
        
        total_selections = expert_indices.numel()
        f = expert_counts / total_selections
        P = routing_probs.mean(dim=0)
        aux_loss = self.num_experts * (f * P).sum()
        
        return self.aux_loss_coef * aux_loss


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "num_experts": 8, "top_k": 2},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("moe", "5_Aux_Loss_LoadBalance")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    num_tokens = p["batch_size"] * p["seq_length"]
    router_logits = DISTRIBUTIONS[dist_name]((num_tokens, p["num_experts"]), dtype=dtype, device=device)
    expert_indices = torch.randint(0, p["num_experts"], (num_tokens, p["top_k"]), device=device)
    return [router_logits, expert_indices]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["num_experts"]]
