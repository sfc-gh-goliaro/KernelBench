import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Batch Normalization.
    """
    def __init__(self, num_features: int):
        """
        Initializes the BatchNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(Model, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Batch Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Batch Normalization applied, same shape as input.
        """
        return self.bn(x)


PARAMETERS = [
    {"batch_size": 64, "features": 64, "dim1": 512, "dim2": 512},
    # EfficientNet-B0: after stem (32 channels, 112x112)
    {"batch_size": 32, "features": 32, "dim1": 112, "dim2": 112},
    # EfficientNet-B7: after stem (64 channels, 300x300)
    {"batch_size": 8, "features": 64, "dim1": 300, "dim2": 300},
    # ResNet-50: after conv1 (64 channels, 112x112)
    {"batch_size": 32, "features": 64, "dim1": 112, "dim2": 112},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("normalization", "1_BatchNorm")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["features"], p["dim1"], p["dim2"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["features"]]
