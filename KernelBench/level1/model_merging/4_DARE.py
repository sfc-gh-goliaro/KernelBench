import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    DARE (Drop And REscale)
    
    Used by: DARE merging
    
    Randomly drop deltas, rescale remaining.
    
    Shapes:
        delta: (param_shape) task vector
        Output: (param_shape) processed
    """
    
    def __init__(self, drop_rate: float = 0.9):
        super(Model, self).__init__()
        self.drop_rate = drop_rate
    
    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        if self.training:
            mask = torch.bernoulli(torch.full_like(delta, 1 - self.drop_rate))
            # Rescale to maintain expected value
            return delta * mask / (1 - self.drop_rate)
        else:
            return delta



PARAMETERS = [
    {"param_shape": (4096, 4096)},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("model_merging", "4_DARE")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    delta = DISTRIBUTIONS[dist_name]((*p["param_shape"]), dtype=dtype, device=device)
    return [delta]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return [0.9]
