import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs 3D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the kernel to apply pooling.
            stride (int, optional): Stride of the pooling operation. Defaults to None, which uses the kernel size.
            padding (int, optional): Padding to apply before pooling. Defaults to 0.
        """
        super(Model, self).__init__()
        self.avg_pool = nn.AvgPool3d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Average Pooling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied, shape depends on kernel_size, stride and padding.
        """
        return self.avg_pool(x)


PARAMETERS = [
    {"batch_size": 16, "channels": 32, "depth": 128, "height": 128, "width": 256, "kernel_size": 3, "stride": 2, "padding": 1},
    # C3D: final avgpool before fc (512 channels, 1x4x4)
    {"batch_size": 8, "channels": 512, "depth": 1, "height": 4, "width": 4, "kernel_size": 1, "stride": 1, "padding": 0},
    # I3D: spatial-temporal pooling (832 channels, 8x7x7)
    {"batch_size": 4, "channels": 832, "depth": 8, "height": 7, "width": 7, "kernel_size": 2, "stride": 2, "padding": 0},
    # 3D-ResNet-50: avgpool before classifier (2048 channels, 4x7x7)
    {"batch_size": 8, "channels": 2048, "depth": 4, "height": 7, "width": 7, "kernel_size": 4, "stride": 1, "padding": 0},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("pooling", "6_AvgPool3d")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["channels"], p["depth"], p["height"], p["width"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["kernel_size"], p["stride"], p["padding"]]
