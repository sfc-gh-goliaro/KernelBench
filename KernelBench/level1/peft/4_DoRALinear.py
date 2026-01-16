import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    DoRA (Weight-Decomposed Low-Rank Adaptation) Linear Layer
    
    Used by: DoRA fine-tuning, parameter-efficient training
    """
    
    def __init__(self, in_features: int = 4096, out_features: int = 4096,
                 rank: int = 16, alpha: float = 32.0):
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank
        
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.02,
            requires_grad=False
        )
        
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        with torch.no_grad():
            base_norm = self.weight.norm(dim=1, keepdim=True)
        self.magnitude = nn.Parameter(base_norm.squeeze())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adapted_weight = self.weight + self.scaling * (self.lora_B @ self.lora_A)
        weight_norm = adapted_weight.norm(dim=1, keepdim=True)
        direction = adapted_weight / (weight_norm + 1e-8)
        final_weight = self.magnitude.unsqueeze(1) * direction
        return F.linear(x, final_weight)


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "in_features": 4096, "out_features": 4096, "rank": 16, "alpha": 32.0},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("peft", "4_DoRALinear")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["seq_length"], p["in_features"])
    x = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["in_features"], p["out_features"], p["rank"], p["alpha"]]
