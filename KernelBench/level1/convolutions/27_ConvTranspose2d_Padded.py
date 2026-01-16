import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a 2D transposed convolution operation with asymmetric input and kernel, with optional padding.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_transpose2d(x)

# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "in_channels": 32, "out_channels": 32, "kernel_size": (3, 7), "height": 512, "width": 1024, "stride": (1, 1), "padding": (1, 3)},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("convolutions", "27_ConvTranspose2d_Padded")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["in_channels"], p["height"], p["width"])
    x = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["in_channels"], p["out_channels"], p["kernel_size"], p["stride"], p["padding"]]
