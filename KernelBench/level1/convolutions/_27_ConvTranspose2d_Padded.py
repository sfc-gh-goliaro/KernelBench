import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a 2D transposed convolution operation with asymmetric input and kernel, with optional padding.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (tuple, optional): Stride of the convolution (height, width). Defaults to (1, 1).
        padding (tuple, optional): Padding applied to the input (height, width). Defaults to (0, 0).
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        
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
    {"batch_size": 8, "in_channels": 32, "out_channels": 32, "kernel_size": (3, 7), "height": 512, "width": 1024, "stride": (1, 1), "padding": (1, 3)},
    # SDXL: VAE decoder with same-size padded transpose
    {"batch_size": 1, "in_channels": 512, "out_channels": 512, "kernel_size": (3, 3), "height": 64, "width": 64, "stride": (1, 1), "padding": (1, 1)},
    # ResNet-Decoder: skip connection compatible upsampling
    {"batch_size": 8, "in_channels": 256, "out_channels": 128, "kernel_size": (3, 3), "height": 128, "width": 128, "stride": (1, 1), "padding": (1, 1)},
    # InceptionV3: asymmetric padded transpose conv
    {"batch_size": 16, "in_channels": 192, "out_channels": 192, "kernel_size": (1, 7), "height": 35, "width": 35, "stride": (1, 1), "padding": (0, 3)},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("convolutions", "27_ConvTranspose2d_Padded")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["in_channels"], p["height"], p["width"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["in_channels"], p["out_channels"], p["kernel_size"], p["stride"], p["padding"]]
