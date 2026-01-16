import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Hard-Swish Activation
    
    Used by: MobileNetV3, MnasNet
    
    Hard-Swish: x * ReLU6(x+3)/6, efficient approximation of Swish.
    More computationally efficient than regular Swish/SiLU.
    
    Shapes:
        Input: any shape
        Output: same shape as input
    """
    
    def __init__(self):
        """Initialize Hard-Swish."""
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Hard-Swish activation.
        
        Args:
            x: Input tensor of any shape
            
        Returns:
            Activated tensor of same shape
        """
        return x * F.relu6(x + 3) / 6


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 32, "channels": 960, "height": 7, "width": 7},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("activations", "17_HardSwish")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["channels"], p["height"], p["width"])
    x = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
