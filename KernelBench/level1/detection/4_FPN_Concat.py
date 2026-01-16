import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Feature Pyramid Network (FPN) Concatenation
    
    Used by: YOLO, RetinaNet, Faster-RCNN
    
    FPN concatenation for multi-scale feature fusion.
    Upsamples high-level features and concatenates with low-level.
    
    Shapes:
        high_level: (batch, high_channels, H, W)
        low_level: (batch, low_channels, 2H, 2W)
        Output: (batch, out_channels, 2H, 2W)
    """
    
    def __init__(self, high_channels: int, low_channels: int, out_channels: int):
        """
        Initialize FPN concat.
        
        Args:
            high_channels: Channels in high-level (smaller) features
            low_channels: Channels in low-level (larger) features
            out_channels: Output channels
        """
        super(Model, self).__init__()
        
        # Reduce high-level channels
        self.reduce_high = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        # Reduce low-level channels
        self.reduce_low = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        # Fuse concatenated features
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, high_level: torch.Tensor, low_level: torch.Tensor) -> torch.Tensor:
        # Reduce channels
        high = self.reduce_high(high_level)
        low = self.reduce_low(low_level)
        
        # Upsample high-level to match low-level size
        high = F.interpolate(high, size=low.shape[2:], mode='nearest')
        
        # Concatenate and fuse
        return self.fuse(torch.cat([high, low], dim=1))


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "high_channels": 512, "low_channels": 256, "out_channels": 256, "high_size": 20, "low_size": 40},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("detection", "4_FPN_Concat")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    high_shape = (p["batch_size"], p["high_channels"], p["high_size"], p["high_size"])
    low_shape = (p["batch_size"], p["low_channels"], p["low_size"], p["low_size"])
    high = DISTRIBUTIONS[dist_name](high_shape, dtype=dtype, device=device)
    low = DISTRIBUTIONS[dist_name](low_shape, dtype=dtype, device=device)
    return [high, low]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["high_channels"], p["low_channels"], p["out_channels"]]
