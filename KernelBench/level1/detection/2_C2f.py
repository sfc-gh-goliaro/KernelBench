import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    C2f Block (YOLOv8)
    
    Used by: YOLOv8
    
    CSP bottleneck with 2 convolutions, used in YOLOv8 backbone.
    Faster variant of CSP block.
    
    Shapes:
        Input: (batch, in_channels, height, width)
        Output: (batch, out_channels, height, width)
    """
    
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 1,
                 shortcut: bool = True):
        """
        Initialize C2f block.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            num_blocks: Number of bottleneck blocks
            shortcut: Whether to use residual connections in bottlenecks
        """
        super(Model, self).__init__()
        self.c = out_channels // 2
        self.shortcut = shortcut
        
        self.cv1 = nn.Sequential(
            nn.Conv2d(in_channels, 2 * self.c, 1, bias=False),
            nn.BatchNorm2d(2 * self.c),
            nn.SiLU(inplace=True)
        )
        
        self.cv2 = nn.Sequential(
            nn.Conv2d((2 + num_blocks) * self.c, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        # Bottleneck blocks
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.c, self.c, 3, padding=1, bias=False),
                nn.BatchNorm2d(self.c),
                nn.SiLU(inplace=True),
                nn.Conv2d(self.c, self.c, 3, padding=1, bias=False),
                nn.BatchNorm2d(self.c),
                nn.SiLU(inplace=True)
            ) for _ in range(num_blocks)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial conv and split
        y = list(self.cv1(x).chunk(2, dim=1))
        
        # Process through bottlenecks
        for block in self.blocks:
            if self.shortcut:
                y.append(y[-1] + block(y[-1]))
            else:
                y.append(block(y[-1]))
        
        return self.cv2(torch.cat(y, dim=1))


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 8, "in_channels": 256, "out_channels": 256, "height": 40, "width": 40},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("detection", "2_C2f")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["in_channels"], p["height"], p["width"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["in_channels"], p["out_channels"]]
