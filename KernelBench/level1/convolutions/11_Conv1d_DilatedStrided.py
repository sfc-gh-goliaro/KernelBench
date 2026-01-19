import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a standard 1D convolution operation with asymmetric input and a square kernel, potentially dilated and strided.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return self.conv1d(x)

# Test code

PARAMETERS = [
    {"batch_size": 64, "in_channels": 64, "out_channels": 128, "kernel_size": 3, "length": 524280, "stride": 3, "dilation": 4},
    # WaveNet: dilated causal convolution
    {"batch_size": 16, "in_channels": 256, "out_channels": 256, "kernel_size": 3, "length": 16000, "stride": 1, "dilation": 2},
    # TCN: temporal convolution
    {"batch_size": 32, "in_channels": 128, "out_channels": 128, "kernel_size": 5, "length": 4096, "stride": 1, "dilation": 4},
    # Audio downsampling
    {"batch_size": 16, "in_channels": 80, "out_channels": 512, "kernel_size": 3, "length": 3000, "stride": 2, "dilation": 1},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("convolutions", "11_Conv1d_DilatedStrided")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["in_channels"], p["length"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["in_channels"], p["out_channels"], p["kernel_size"], p["stride"], p["dilation"]]
