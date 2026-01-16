import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Fisher-Weighted Averaging
    
    Used by: Fisher merging
    """
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, weights: list, fishers: list) -> torch.Tensor:
        numerator = sum(w * f for w, f in zip(weights, fishers))
        denominator = sum(fishers) + 1e-10
        return numerator / denominator


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"param_shape": (4096, 4096), "num_models": 3},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("model_merging", "6_FisherWeighted")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    weights = [DISTRIBUTIONS[dist_name](p["param_shape"], dtype=dtype, device=device) for _ in range(p["num_models"])]
    # Fisher information should be positive
    fishers = [DISTRIBUTIONS["uniform_pos"](p["param_shape"], dtype=dtype, device=device) for _ in range(p["num_models"])]
    return [weights, fishers]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
