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
    
    Fisher-weighted averaging using Fisher information matrix.
    
    Shapes:
        weights: list of (param_shape) weight tensors
        fishers: list of (param_shape) Fisher information
        Output: (param_shape) merged
    """
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, weights: list, fishers: list) -> torch.Tensor:
        # Weighted average: sum(w_i * F_i) / sum(F_i)
        numerator = sum(w * f for w, f in zip(weights, fishers))
        denominator = sum(fishers) + 1e-10
        return numerator / denominator



PARAMETERS = [
    # Llama-3.1 8B: Fisher-weighted attention merging
    {"param_shape": (4096, 4096)},
    # Mistral 7B: Fisher-weighted MLP merging
    {"param_shape": (14336, 4096)},
    # Phi-3 Medium: Fisher importance averaging
    {"param_shape": (5120, 17920)},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("model_merging", "6_FisherWeighted")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["param_shape"])
    x = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
