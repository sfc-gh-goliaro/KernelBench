import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a depthwise 2D convolution with asymmetric input and square kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), stride=stride, padding=padding, groups=in_channels, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d(x)

# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 64, "in_channels": 128, "out_channels": 128, "kernel_size": 3, "height_in": 256, "width_in": 512, "stride": 1, "padding": 0},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("convolutions", "32_DepthwiseConv2d_AsymInput")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["in_channels"], p["height_in"], p["width_in"])
    x = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["in_channels"], p["out_channels"], p["kernel_size"], p["stride"], p["padding"]]
