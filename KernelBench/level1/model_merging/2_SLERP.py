import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    SLERP (Spherical Linear Interpolation)
    
    Used by: Smooth model interpolation
    
    Spherical interpolation between model weights.
    
    Shapes:
        w1, w2: (param_shape) weight tensors
        Output: (param_shape) interpolated
    """
    
    def __init__(self, t: float = 0.5):
        super(Model, self).__init__()
        self.t = t
    
    def forward(self, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
        w1_flat = w1.flatten()
        w2_flat = w2.flatten()
        
        # Normalize
        w1_norm = w1_flat / (w1_flat.norm() + 1e-10)
        w2_norm = w2_flat / (w2_flat.norm() + 1e-10)
        
        # Angle between vectors
        dot = (w1_norm * w2_norm).sum().clamp(-1, 1)
        theta = torch.acos(dot)
        
        # SLERP
        if theta.abs() < 1e-6:
            result = (1 - self.t) * w1_flat + self.t * w2_flat
        else:
            sin_theta = torch.sin(theta)
            result = (torch.sin((1 - self.t) * theta) / sin_theta) * w1_flat + \
                     (torch.sin(self.t * theta) / sin_theta) * w2_flat
        
        return result.view(w1.shape)



PARAMETERS = [
    {"param_shape": (4096, 4096)},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("model_merging", "2_SLERP")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    w1 = DISTRIBUTIONS[dist_name]((*p["param_shape"]), dtype=dtype, device=device)
    w2 = DISTRIBUTIONS[dist_name]((*p["param_shape"]), dtype=dtype, device=device)
    return [w1, w2]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return [0.5]
