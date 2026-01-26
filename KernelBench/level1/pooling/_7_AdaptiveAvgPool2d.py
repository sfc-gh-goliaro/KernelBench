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
    
    Pools spatial dimensions to a fixed output size regardless of input size.
    Commonly used to reduce feature maps to 1x1 before the classification head.
    
    Shapes:
        Input: (batch_size, channels, height, width)
        Output: (batch_size, channels, output_height, output_width)
    """
    
    def __init__(self, output_size: tuple = (1, 1)):
        """
        Initialize Adaptive Average Pool 2D.
        
        Args:
            output_size: Target output size (height, width)
        """
        super(Model, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive average pooling.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            Pooled tensor (batch_size, channels, output_height, output_width)
        """
        return self.pool(x)


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 32, "channels": 2048, "height": 7, "width": 7, "output_size": (1, 1)},
    # ResNet-50: final adaptive pool before classifier (2048 channels, 7x7 -> 1x1)
    {"batch_size": 64, "channels": 2048, "height": 7, "width": 7, "output_size": (1, 1)},
    # EfficientNet-B4: final adaptive pool (1792 channels, 12x12 -> 1x1)
    {"batch_size": 16, "channels": 1792, "height": 12, "width": 12, "output_size": (1, 1)},
    # VGG-16: adaptive pool if modified for variable input (512 channels, 7x7 -> 1x1)
    {"batch_size": 32, "channels": 512, "height": 7, "width": 7, "output_size": (1, 1)},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("pooling", "7_AdaptiveAvgPool2d")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["channels"], p["height"], p["width"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["output_size"]]
