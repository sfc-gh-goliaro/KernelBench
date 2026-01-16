import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Task Arithmetic
    
    Used by: Multi-task merging
    """
    
    def __init__(self, scaling: float = 1.0):
        super(Model, self).__init__()
        self.scaling = scaling
    
    def forward(self, base: torch.Tensor, *deltas: torch.Tensor) -> torch.Tensor:
        result = base.clone()
        for delta in deltas:
            result = result + self.scaling * delta
        return result


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"param_shape": (4096, 4096), "num_deltas": 2, "scaling": 0.5},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("model_merging", "5_TaskArithmetic")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    base = DISTRIBUTIONS[dist_name](p["param_shape"], dtype=dtype, device=device)
    deltas = [DISTRIBUTIONS[dist_name](p["param_shape"], dtype=dtype, device=device) * 0.1 for _ in range(p["num_deltas"])]
    return [base] + deltas

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["scaling"]]
