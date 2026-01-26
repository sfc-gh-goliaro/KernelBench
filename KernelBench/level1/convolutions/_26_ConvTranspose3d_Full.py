import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a 3D transposed convolution operation with square input and square kernel,
    and supports padding, dilation, and stride.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel (square kernel, so only one value needed).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D transposed convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        return self.conv_transpose3d(x)

# Test code

PARAMETERS = [
    {"batch_size": 16, "in_channels": 32, "out_channels": 64, "kernel_size": 3, "depth": 16, "height": 32, "width": 32, "stride": 2, "padding": 1, "dilation": 2},
    # 3D-UNet: volumetric segmentation with dilated decoder
    {"batch_size": 2, "in_channels": 256, "out_channels": 128, "kernel_size": 3, "depth": 8, "height": 16, "width": 16, "stride": 2, "padding": 1, "dilation": 1},
    # Video-VAE: latent video decoder with full options
    {"batch_size": 4, "in_channels": 512, "out_channels": 256, "kernel_size": 4, "depth": 4, "height": 8, "width": 8, "stride": 2, "padding": 1, "dilation": 1},
    # MedicalNet-3D: CT/MRI reconstruction decoder
    {"batch_size": 1, "in_channels": 128, "out_channels": 64, "kernel_size": 3, "depth": 32, "height": 64, "width": 64, "stride": 1, "padding": 2, "dilation": 2},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("convolutions", "26_ConvTranspose3d_Full")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["in_channels"], p["depth"], p["height"], p["width"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["in_channels"], p["out_channels"], p["kernel_size"], p["stride"], p["padding"], p["dilation"]]
