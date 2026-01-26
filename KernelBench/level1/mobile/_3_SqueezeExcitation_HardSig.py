import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Squeeze-and-Excitation Block with Hard-Sigmoid
    
    Used by: MobileNetV3
    
    SE block using Hard-Sigmoid instead of Sigmoid for efficiency.
    
    Shapes:
        Input: (batch, channels, height, width)
        Output: (batch, channels, height, width)
    """
    
    def __init__(self, channels: int, reduction: int = 4):
        """
        Initialize SE block with Hard-Sigmoid.
        
        Args:
            channels: Number of input channels
            reduction: Reduction ratio for squeeze
        """
        super(Model, self).__init__()
        self.channels = channels
        squeezed_channels = max(1, channels // reduction)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, squeezed_channels)
        self.fc2 = nn.Linear(squeezed_channels, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SE with Hard-Sigmoid."""
        batch_size = x.shape[0]
        
        se = self.avg_pool(x).view(batch_size, -1)
        se = F.relu(self.fc1(se))
        se = self.fc2(se)
        # Hard-Sigmoid instead of Sigmoid
        se = F.relu6(se + 3) / 6
        se = se.view(batch_size, self.channels, 1, 1)
        
        return x * se


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 32, "channels": 96, "height": 56, "width": 56},
    # MobileNetV3-Large: SE in MBConv (72 channels, 56x56)
    {"batch_size": 32, "channels": 72, "height": 56, "width": 56},
    # MobileNetV3-Large: SE in MBConv (120 channels, 28x28)
    {"batch_size": 32, "channels": 120, "height": 28, "width": 28},
    # MobileNetV3-Small: SE in MBConv (96 channels, 14x14)
    {"batch_size": 32, "channels": 96, "height": 14, "width": 14},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("mobile", "3_SqueezeExcitation_HardSig")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["channels"], p["height"], p["width"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["channels"]]
