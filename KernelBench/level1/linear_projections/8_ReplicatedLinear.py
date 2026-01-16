import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Replicated Linear (Router Linear)
    
    Used by: vLLM, TensorRT-LLM (MoE routing)
    """
    
    def __init__(self, hidden_size: int = 4096, num_experts: int = 8):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.weight = nn.Linear(hidden_size, num_experts, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight(x)


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"num_tokens": 16384, "hidden_size": 4096, "num_experts": 8},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("linear_projections", "8_ReplicatedLinear")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["num_tokens"], p["hidden_size"])
    x = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["hidden_size"], p["num_experts"]]
