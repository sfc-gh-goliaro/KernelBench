import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a 2D transposed convolution operation with asymmetric input, asymmetric kernel, 
    grouped, padded, and dilated.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (tuple, optional): Stride of the convolution (height, width). Defaults to (1, 1).
        padding (tuple, optional): Padding applied to the input (height, width). Defaults to (0, 0).
        dilation (tuple, optional): Spacing between kernel elements (height, width). Defaults to (1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.conv_transpose2d(x)

# Test code

PARAMETERS = [
    {"batch_size": 16, "in_channels": 32, "out_channels": 64, "kernel_size": (3, 5), "height": 128, "width": 256, "stride": (2, 3), "padding": (1, 2), "dilation": (2, 1), "groups": 4},
    # SDXL-VAE: full-featured decoder transpose conv
    {"batch_size": 1, "in_channels": 512, "out_channels": 256, "kernel_size": (3, 3), "height": 64, "width": 64, "stride": (2, 2), "padding": (1, 1), "dilation": (1, 1), "groups": 1},
    # DeepLabV3+: ASPP with dilated grouped transpose
    {"batch_size": 4, "in_channels": 256, "out_channels": 256, "kernel_size": (3, 3), "height": 65, "width": 65, "stride": (1, 1), "padding": (6, 6), "dilation": (6, 6), "groups": 4},
    # StyleGAN2: adaptive generator upsampling
    {"batch_size": 8, "in_channels": 512, "out_channels": 256, "kernel_size": (4, 4), "height": 16, "width": 16, "stride": (2, 2), "padding": (1, 1), "dilation": (1, 1), "groups": 1},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("convolutions", "25_ConvTranspose2d_Full")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["in_channels"], p["height"], p["width"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["in_channels"], p["out_channels"], p["kernel_size"], p["stride"], p["padding"], p["dilation"], p["groups"]]
