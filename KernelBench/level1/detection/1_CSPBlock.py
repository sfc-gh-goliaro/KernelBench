import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Cross Stage Partial (CSP) Block
    
    Used by: YOLOv5, YOLOv8
    
    CSP block: split → conv path → concat for efficient gradient flow.
    
    Shapes:
        Input: (batch, in_channels, height, width)
        Output: (batch, out_channels, height, width)
    """
    
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 1):
        """
        Initialize CSP block.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            num_blocks: Number of bottleneck blocks
        """
        super(Model, self).__init__()
        hidden_channels = out_channels // 2
        
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        
        self.conv2 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        
        # Bottleneck blocks
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, 1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.SiLU(inplace=True)
            ) for _ in range(num_blocks)
        ])
        
        self.conv3 = nn.Conv2d(hidden_channels * 2, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split path
        y1 = F.silu(self.bn1(self.conv1(x)))
        y2 = F.silu(self.bn2(self.conv2(x)))
        
        # Process one path
        y1 = self.blocks(y1)
        
        # Concatenate and fuse
        return F.silu(self.bn3(self.conv3(torch.cat([y1, y2], dim=1))))


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 8, "in_channels": 256, "out_channels": 256, "height": 40, "width": 40},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("detection", "1_CSPBlock")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["in_channels"], p["height"], p["width"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["in_channels"], p["out_channels"]]
