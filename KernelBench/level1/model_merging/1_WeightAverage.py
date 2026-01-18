import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Weight Averaging
    
    Used by: Model soup, ensemble
    
    Simple weight averaging: (w1 + w2 + ... + wn) / n
    
    Shapes:
        weights: list of (param_shape) tensors
        Output: (param_shape) averaged
    """
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, *weights: torch.Tensor) -> torch.Tensor:
        return sum(weights) / len(weights)



PARAMETERS = [
    {"param_shape": (4096, 4096)},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("model_merging", "1_WeightAverage")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    w1 = DISTRIBUTIONS[dist_name]((*p["param_shape"]), dtype=dtype, device=device)
    w2 = DISTRIBUTIONS[dist_name]((*p["param_shape"]), dtype=dtype, device=device)
    w3 = DISTRIBUTIONS[dist_name]((*p["param_shape"]), dtype=dtype, device=device)
    return [w1, w2, w3]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
