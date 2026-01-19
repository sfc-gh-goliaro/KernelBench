import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a transposed 3D convolution with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (kernel_depth, kernel_width, kernel_height), 
                             where kernel_width == kernel_height.
        stride (tuple, optional): Stride of the convolution. Defaults to (1, 1, 1).
        padding (tuple, optional): Padding applied to the input. Defaults to (0, 0, 0).
        output_padding (tuple, optional): Additional size added to one side of the output shape. Defaults to (0, 0, 0).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, width, height).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, width_out, height_out).
        """
        return self.conv_transpose3d(x)

# Test code

PARAMETERS = [
    {"batch_size": 16, "in_channels": 32, "out_channels": 64, "kernel_depth": 3, "kernel_width": 5, "kernel_height": 5, "depth": 64, "width": 64, "height": 64},
    # I3D: video action recognition decoder with temporal asymmetry
    {"batch_size": 4, "in_channels": 512, "out_channels": 256, "kernel_depth": 1, "kernel_width": 3, "kernel_height": 3, "depth": 8, "height": 28, "width": 28},
    # SlowFast: asymmetric temporal upsampling
    {"batch_size": 8, "in_channels": 256, "out_channels": 128, "kernel_depth": 5, "kernel_width": 3, "kernel_height": 3, "depth": 16, "height": 56, "width": 56},
    # MedicalNet: 3D CT/MRI segmentation decoder
    {"batch_size": 2, "in_channels": 128, "out_channels": 64, "kernel_depth": 3, "kernel_width": 5, "kernel_height": 5, "depth": 32, "height": 128, "width": 128},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("convolutions", "18_ConvTranspose3d_AsymKernel")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["in_channels"], p["depth"], p["width"], p["height"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["in_channels"], p["out_channels"], (kernel_depth, p["kernel_width"], kernel_height)]
