import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs 1D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        """
        Initializes the 1D Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to 1.
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(Model, self).__init__()
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 1D Average Pooling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, input_length).

        Returns:
            torch.Tensor: Output tensor with 1D Average Pooling applied, shape (batch_size, in_channels, output_length).
        """
        return self.avg_pool(x)


PARAMETERS = [
    {"batch_size": 64, "in_channels": 128, "input_length": 65536, "kernel_size": 8, "stride": 1, "padding": 4},
    # Wav2Vec 2.0: feature aggregation layer (768 channels, audio segments)
    {"batch_size": 32, "in_channels": 768, "input_length": 8000, "kernel_size": 4, "stride": 2, "padding": 1},
    # HuBERT: temporal averaging (1024 channels, speech frames)
    {"batch_size": 16, "in_channels": 1024, "input_length": 4096, "kernel_size": 3, "stride": 1, "padding": 1},
    # WaveGlow: audio synthesis pooling (256 channels, long sequences)
    {"batch_size": 8, "in_channels": 256, "input_length": 32000, "kernel_size": 8, "stride": 4, "padding": 2},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("pooling", "4_AvgPool1d")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["in_channels"], p["input_length"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["kernel_size"], p["stride"], p["padding"]]
