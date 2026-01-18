import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Shared Expert Fused MoE
    
    Used by: DeepSeek-V2, DeepSeek-V3, DeepSeek-R1, Llama 4
    
    Mixture-of-Experts with a shared expert that is always activated.
    Combines sparse expert computation with a dense shared expert.
    Output = SharedExpert(x) + Σ(gate_i * Expert_i(x))
    
    Shapes:
        Input: (num_tokens, hidden_size)
        Output: (num_tokens, hidden_size)
    """
    
    def __init__(self, hidden_size: int = 7168, intermediate_size: int = 2048,
                 num_experts: int = 64, top_k: int = 6, shared_expert_intermediate: int = 2048):
        """
        Initialize Shared Fused MoE.
        
        Args:
            hidden_size: Input/output hidden dimension
            intermediate_size: Expert intermediate dimension
            num_experts: Number of routed experts
            top_k: Number of experts to select per token
            shared_expert_intermediate: Shared expert intermediate size
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router for sparse experts
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Sparse experts (simplified as stacked linear layers)
        self.experts_gate = nn.Parameter(
            torch.randn(num_experts, hidden_size, intermediate_size) * 0.02
        )
        self.experts_up = nn.Parameter(
            torch.randn(num_experts, hidden_size, intermediate_size) * 0.02
        )
        self.experts_down = nn.Parameter(
            torch.randn(num_experts, intermediate_size, hidden_size) * 0.02
        )
        
        # Shared expert (always activated)
        self.shared_gate = nn.Linear(hidden_size, shared_expert_intermediate, bias=False)
        self.shared_up = nn.Linear(hidden_size, shared_expert_intermediate, bias=False)
        self.shared_down = nn.Linear(shared_expert_intermediate, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Shared + Sparse MoE.
        
        Args:
            x: Input tensor (num_tokens, hidden_size)
            
        Returns:
            Output tensor (num_tokens, hidden_size)
        """
        num_tokens, hidden_size = x.shape
        
        # Compute shared expert output
        shared_gate_out = F.silu(self.shared_gate(x))
        shared_up_out = self.shared_up(x)
        shared_out = self.shared_down(shared_gate_out * shared_up_out)
        
        # Compute routing weights
        router_logits = self.router(x)  # (num_tokens, num_experts)
        routing_weights, selected_experts = torch.topk(
            F.softmax(router_logits, dim=-1), self.top_k, dim=-1
        )
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Compute sparse expert outputs (simplified loop implementation)
        sparse_out = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = selected_experts[:, i]  # (num_tokens,)
            weight = routing_weights[:, i:i+1]   # (num_tokens, 1)
            
            # Gather expert weights for selected experts
            gate_w = self.experts_gate[expert_idx]  # (num_tokens, hidden, inter)
            up_w = self.experts_up[expert_idx]
            down_w = self.experts_down[expert_idx]
            
            # Compute expert output
            gate_out = F.silu(torch.bmm(x.unsqueeze(1), gate_w).squeeze(1))
            up_out = torch.bmm(x.unsqueeze(1), up_w).squeeze(1)
            expert_out = torch.bmm((gate_out * up_out).unsqueeze(1), down_w).squeeze(1)
            
            sparse_out += weight * expert_out
        
        return shared_out + sparse_out


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"num_tokens": 4096, "hidden_size": 7168, "intermediate_size": 2048, "num_experts": 64, "top_k": 6, "shared_expert_intermediate": 2048},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("moe", "6_SharedFusedMoE")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["num_tokens"], p["hidden_size"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["hidden_size"], p["intermediate_size"], p["num_experts"], p["top_k"], p["shared_expert_intermediate"]]
