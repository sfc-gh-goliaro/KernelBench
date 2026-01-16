import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Adaptive Average Pooling 2D
    
    Used by: ResNet, EfficientNet, ViT (before classification head)
    """
    
    def __init__(self, output_size: tuple = (1, 1)):
        super(Model, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 32, "channels": 2048, "height": 7, "width": 7, "output_size": (1, 1)},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("pooling", "7_AdaptiveAvgPool2d")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["channels"], p["height"], p["width"])
    x = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["output_size"]]
