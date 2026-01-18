import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Anchor-Free Detection Head
    
    Used by: YOLOv8, FCOS
    
    Anchor-free head with center-based predictions.
    
    Shapes:
        Input: (batch, in_channels, height, width)
        Output: cls (batch, num_classes, H, W), box (batch, 4, H, W)
    """
    
    def __init__(self, in_channels: int, num_classes: int, reg_max: int = 16):
        """
        Initialize anchor-free head.
        
        Args:
            in_channels: Input channels
            num_classes: Number of object classes
            reg_max: Max value for DFL regression
        """
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        
        # Classification branch
        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, 1)
        )
        
        # Regression branch (DFL)
        self.reg_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, 4 * reg_max, 1)
        )
    
    def forward(self, x: torch.Tensor) -> tuple:
        cls_out = self.cls_conv(x)
        reg_out = self.reg_conv(x)
        
        return cls_out, reg_out


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 8, "in_channels": 256, "num_classes": 80, "height": 80, "width": 80},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("detection", "7_AnchorFreeHead")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["in_channels"], p["height"], p["width"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["in_channels"], p["num_classes"]]
