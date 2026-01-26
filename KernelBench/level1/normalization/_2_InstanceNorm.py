import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Instance Normalization.
    """
    def __init__(self, num_features: int):
        """
        Initializes the InstanceNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(Model, self).__init__()
        self.inorm = nn.InstanceNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Instance Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        return self.inorm(x)


PARAMETERS = [
    {"batch_size": 112, "features": 64, "dim1": 512, "dim2": 512},
    # SDXL UNet: GroupNorm replaced by InstanceNorm in some blocks (320 channels, 128x128)
    {"batch_size": 2, "features": 320, "dim1": 128, "dim2": 128},
    # SDXL UNet: mid-block (1280 channels, 16x16)
    {"batch_size": 2, "features": 1280, "dim1": 16, "dim2": 16},
    # Style transfer networks: instance norm on high-res features
    {"batch_size": 4, "features": 128, "dim1": 256, "dim2": 256},
    # SD3-Medium: InstanceNorm in decoder (640 channels, 32x32)
    {"batch_size": 2, "features": 640, "dim1": 32, "dim2": 32},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("normalization", "2_InstanceNorm")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["features"], p["dim1"], p["dim2"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["features"]]
