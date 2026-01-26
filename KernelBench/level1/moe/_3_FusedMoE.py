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
    
    Fused MoE kernel computing all expert FFNs in single kernel
    with efficient memory access. This is a reference implementation;
    production uses optimized Triton/CUDA kernels.
    
    Shapes:
        Input: (batch_size, seq_len, hidden_size)
        Output: (batch_size, seq_len, hidden_size)
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int, top_k: int = 2):
        """
        Initialize fused MoE.
        
        Args:
            hidden_size: Model hidden dimension
            intermediate_size: FFN intermediate dimension
            num_experts: Number of experts
            top_k: Number of experts per token
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Expert weights (all experts packed together)
        self.w1 = nn.Parameter(torch.randn(num_experts, hidden_size, intermediate_size) * 0.02)
        self.w2 = nn.Parameter(torch.randn(num_experts, intermediate_size, hidden_size) * 0.02)
        self.w3 = nn.Parameter(torch.randn(num_experts, hidden_size, intermediate_size) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fused MoE forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, hidden_size)
            
        Returns:
            Output tensor (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.hidden_size)
        num_tokens = x_flat.shape[0]
        
        # Router
        router_logits = self.gate(x_flat)
        routing_weights, expert_indices = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Process each expert (fused implementation would do this in parallel)
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (expert_indices == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue
            
            # Get token indices and their weights for this expert
            token_indices = expert_mask.nonzero(as_tuple=True)[0]
            
            # Get routing weights for this expert
            expert_weights = torch.zeros(num_tokens, device=x.device)
            for k in range(self.top_k):
                mask_k = expert_indices[:, k] == expert_idx
                expert_weights[mask_k] = routing_weights[mask_k, k]
            
            expert_weights = expert_weights[token_indices]
            
            # Get tokens for this expert
            expert_input = x_flat[token_indices]
            
            # SwiGLU: SiLU(x @ w1) * (x @ w3) @ w2
            gate = F.silu(expert_input @ self.w1[expert_idx])
            up = expert_input @ self.w3[expert_idx]
            expert_output = (gate * up) @ self.w2[expert_idx]
            
            # Weight by routing and accumulate
            output[token_indices] += expert_weights.unsqueeze(-1) * expert_output
        
        return output.view(batch_size, seq_len, self.hidden_size)


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "hidden_size": 4096, "intermediate_size": 14336, "num_experts": 8, "top_k": 2},
    # DeepSeek-V2-Lite: hidden_size=2048, intermediate_size=10944, num_experts=64, top_k=6
    {"batch_size": 8, "seq_length": 2048, "hidden_size": 2048, "intermediate_size": 10944, "num_experts": 64, "top_k": 6},
    # Mixtral-8x7B: hidden_size=4096, intermediate_size=14336, num_experts=8, top_k=2
    {"batch_size": 8, "seq_length": 4096, "hidden_size": 4096, "intermediate_size": 14336, "num_experts": 8, "top_k": 2},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("moe", "3_FusedMoE")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_length"], p["hidden_size"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["hidden_size"], p["intermediate_size"], p["num_experts"], p["top_k"]]
