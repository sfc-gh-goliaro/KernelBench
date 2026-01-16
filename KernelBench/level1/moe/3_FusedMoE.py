import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Fused MoE (Mixture of Experts)
    
    Used by: Mixtral, DeepSeek-V2/V3 (vLLM/SGLang)
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int, top_k: int = 2):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.w1 = nn.Parameter(torch.randn(num_experts, hidden_size, intermediate_size) * 0.02)
        self.w2 = nn.Parameter(torch.randn(num_experts, intermediate_size, hidden_size) * 0.02)
        self.w3 = nn.Parameter(torch.randn(num_experts, hidden_size, intermediate_size) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.hidden_size)
        num_tokens = x_flat.shape[0]
        
        router_logits = self.gate(x_flat)
        routing_weights, expert_indices = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        output = torch.zeros_like(x_flat)
        
        for expert_idx in range(self.num_experts):
            expert_mask = (expert_indices == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue
            
            token_indices = expert_mask.nonzero(as_tuple=True)[0]
            
            expert_weights = torch.zeros(num_tokens, device=x.device)
            for k in range(self.top_k):
                mask_k = expert_indices[:, k] == expert_idx
                expert_weights[mask_k] = routing_weights[mask_k, k]
            
            expert_weights = expert_weights[token_indices]
            expert_input = x_flat[token_indices]
            
            gate = F.silu(expert_input @ self.w1[expert_idx])
            up = expert_input @ self.w3[expert_idx]
            expert_output = (gate * up) @ self.w2[expert_idx]
            
            output[token_indices] += expert_weights.unsqueeze(-1) * expert_output
        
        return output.view(batch_size, seq_len, self.hidden_size)


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "hidden_size": 4096, "intermediate_size": 14336, "num_experts": 8, "top_k": 2},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("moe", "3_FusedMoE")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["seq_length"], p["hidden_size"])
    x = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["hidden_size"], p["intermediate_size"], p["num_experts"], p["top_k"]]
