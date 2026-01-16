import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Squeeze-and-Excitation Block
    
    Used by: MobileNetV3, EfficientNet, MnasNet
    """
    
    def __init__(self, channels: int, reduction: int = 4):
        super(Model, self).__init__()
        self.channels = channels
        squeezed_channels = max(1, channels // reduction)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, squeezed_channels)
        self.fc2 = nn.Linear(squeezed_channels, channels)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        se = self.avg_pool(x).view(batch_size, -1)
        se = self.fc1(se)
        se = self.activation(se)
        se = self.fc2(se)
        se = torch.sigmoid(se)
        
        se = se.view(batch_size, self.channels, 1, 1)
        return x * se


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 32, "channels": 96, "height": 56, "width": 56},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("mobile", "2_SqueezeExcitation")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["channels"], p["height"], p["width"])
    x = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["channels"]]
