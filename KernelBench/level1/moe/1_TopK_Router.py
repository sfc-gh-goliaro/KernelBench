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
    
    Expert routing via softmax over expert logits + top-k selection.
    Returns indices of selected experts and their routing weights.
    
    Shapes:
        Input: (batch_size, seq_len, hidden_size)
        Output: (indices: (batch*seq, k), weights: (batch*seq, k))
    """
    
    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2):
        """
        Initialize top-k router.
        
        Args:
            hidden_size: Input hidden dimension
            num_experts: Total number of experts
            top_k: Number of experts to route to per token
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Route tokens to top-k experts.
        
        Args:
            x: Input tensor (batch, seq_len, hidden_size)
            
        Returns:
            Tuple of (expert_indices, routing_weights):
                expert_indices: (batch*seq, top_k) selected expert IDs
                routing_weights: (batch*seq, top_k) normalized weights
        """
        batch_size, seq_len, _ = x.shape
        
        # Flatten batch and sequence
        x_flat = x.view(-1, self.hidden_size)
        
        # Compute routing scores
        router_logits = self.gate(x_flat)  # (batch*seq, num_experts)
        
        # Get top-k experts
        routing_weights, expert_indices = torch.topk(router_logits, self.top_k, dim=-1)
        
        # Normalize weights with softmax over selected experts
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        return expert_indices, routing_weights


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "hidden_size": 4096, "num_experts": 8, "top_k": 2},
    # DeepSeek-V2-Lite: hidden_size=2048, num_experts=64, num_experts_per_tok=6
    {"batch_size": 8, "seq_length": 2048, "hidden_size": 2048, "num_experts": 64, "top_k": 6},
    # Mixtral-8x7B: hidden_size=4096, num_experts=8, num_experts_per_tok=2
    {"batch_size": 8, "seq_length": 4096, "hidden_size": 4096, "num_experts": 8, "top_k": 2},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("moe", "1_TopK_Router")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_length"], p["hidden_size"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["hidden_size"], p["num_experts"], p["top_k"]]
