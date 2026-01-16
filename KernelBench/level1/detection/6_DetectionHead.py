import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Detection Head (Anchor-based)
    
    Used by: YOLOv5
    
    Detection head: conv layers → cls/box predictions per anchor.
    
    Shapes:
        Input: (batch, in_channels, height, width)
        Output: (batch, num_anchors, height, width, num_classes + 5)
    """
    
    def __init__(self, in_channels: int, num_classes: int, num_anchors: int = 3):
        """
        Initialize detection head.
        
        Args:
            in_channels: Input channels
            num_classes: Number of object classes
            num_anchors: Number of anchors per location
        """
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.num_outputs = num_classes + 5  # cls + (x, y, w, h, obj)
        
        self.conv = nn.Conv2d(in_channels, num_anchors * self.num_outputs, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        
        x = self.conv(x)
        x = x.view(batch_size, self.num_anchors, self.num_outputs, height, width)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        
        return x


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "in_channels": 256, "num_classes": 80, "height": 80, "width": 80},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("detection", "6_DetectionHead")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["in_channels"], p["height"], p["width"])
    x = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["in_channels"], p["num_classes"]]
