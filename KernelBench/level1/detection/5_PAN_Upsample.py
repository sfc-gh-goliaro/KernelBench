import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Path Aggregation Network (PAN) Upsample
    
    Used by: YOLOv5, YOLOv8
    
    PAN upsample + concat for bottom-up path aggregation.
    
    Shapes:
        high_level: (batch, channels, H, W)
        low_level: (batch, channels, 2H, 2W)
        Output: (batch, channels, 2H, 2W)
    """
    
    def __init__(self, channels: int):
        """
        Initialize PAN upsample.
        
        Args:
            channels: Number of channels
        """
        super(Model, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, high_level: torch.Tensor, low_level: torch.Tensor) -> torch.Tensor:
        high = self.upsample(high_level)
        return self.conv(torch.cat([high, low_level], dim=1))


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "channels": 256, "high_size": 20, "low_size": 40},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("detection", "5_PAN_Upsample")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    high_shape = (p["batch_size"], p["channels"], p["high_size"], p["high_size"])
    low_shape = (p["batch_size"], p["channels"], p["low_size"], p["low_size"])
    high = DISTRIBUTIONS[dist_name](high_shape, dtype=dtype, device=device)
    low = DISTRIBUTIONS[dist_name](low_shape, dtype=dtype, device=device)
    return [high, low]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["channels"]]
