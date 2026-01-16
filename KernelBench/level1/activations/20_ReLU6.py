import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    ReLU6 Activation
    
    Used by: MobileNet, EfficientNet-Lite, HardSwish/HardSigmoid implementations
    
    Clipped ReLU: min(max(0, x), 6)
    Prevents activation explosion and is used in mobile-efficient networks.
    
    Shapes:
        Input: any shape
        Output: same shape as input
    """
    
    def __init__(self):
        """Initialize ReLU6."""
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply ReLU6 activation.
        
        Args:
            x: Input tensor of any shape
            
        Returns:
            Activated tensor with values clamped to [0, 6]
        """
        return F.relu6(x)


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 32, "channels": 576, "height": 14, "width": 14},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("activations", "20_ReLU6")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["channels"], p["height"], p["width"])
    x = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
