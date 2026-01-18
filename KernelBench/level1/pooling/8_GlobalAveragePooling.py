import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Global Average Pooling
    
    Used by: ResNet, EfficientNet, MobileNet, most CNN classifiers
    
    Computes the global average over spatial dimensions.
    Reduces (B, C, H, W) to (B, C) by averaging over H and W.
    
    Shapes:
        Input: (batch_size, channels, height, width)
        Output: (batch_size, channels)
    """
    
    def __init__(self):
        """Initialize Global Average Pooling."""
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply global average pooling.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            Pooled tensor (batch_size, channels)
        """
        return x.mean(dim=[-2, -1])


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 32, "channels": 2048, "height": 7, "width": 7},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("pooling", "8_GlobalAveragePooling")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["channels"], p["height"], p["width"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
