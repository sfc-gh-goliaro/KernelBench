import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Gemma RMS Normalization
    
    Used by: Gemma, Gemma-2, Gemma-3
    
    Variant of RMSNorm with (1 + weight) scaling instead of just weight.
    """
    
    def __init__(self, hidden_size: int = 3584, eps: float = 1e-6):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return x_normed * (1 + self.weight)


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "hidden_size": 3584},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("normalization", "9_GemmaRMSNorm")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["seq_length"], p["hidden_size"])
    x = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["hidden_size"]]
