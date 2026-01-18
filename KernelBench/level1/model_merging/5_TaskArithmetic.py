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
    
    Task vectors: add/subtract task-specific weight deltas.
    
    Shapes:
        base: (param_shape) base weights
        deltas: list of (param_shape) task vectors
        Output: (param_shape) merged
    """
    
    def __init__(self, scaling: float = 1.0):
        super(Model, self).__init__()
        self.scaling = scaling
    
    def forward(self, base: torch.Tensor, *deltas: torch.Tensor) -> torch.Tensor:
        result = base.clone()
        for delta in deltas:
            result = result + self.scaling * delta
        return result



PARAMETERS = [
    {"param_shape": (4096, 4096)},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("model_merging", "5_TaskArithmetic")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    base = DISTRIBUTIONS[dist_name]((*p["param_shape"]), dtype=dtype, device=device)
    d1 = DISTRIBUTIONS[dist_name]((*p["param_shape"]), dtype=dtype, device=device)
    d2 = DISTRIBUTIONS[dist_name]((*p["param_shape"]), dtype=dtype, device=device)
    return [base, d1, d2]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return [0.5]
