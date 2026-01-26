import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a LeakyReLU activation.
    """
    def __init__(self, negative_slope: float = 0.01):
        """
        Initializes the LeakyReLU module.

        Args:
            negative_slope (float, optional): The negative slope of the activation function. Defaults to 0.01.
        """
        super(Model, self).__init__()
        self.negative_slope = negative_slope
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LeakyReLU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with LeakyReLU applied, same shape as input.
        """
        return torch.nn.functional.leaky_relu(x, negative_slope=self.negative_slope)


PARAMETERS = [
    {"batch_size": 4096, "dim": 393216},
    # ResNet-50: after conv blocks (batch=32, 256 channels, 56x56 spatial)
    {"batch_size": 32, "dim": 802816},
    # ResNet-101: after conv blocks (batch=16, 512 channels, 28x28)
    {"batch_size": 16, "dim": 401408},
    # VGG-16: after conv layers (batch=32, 512 channels, 7x7 spatial)
    {"batch_size": 32, "dim": 25088},
    # DenseNet-121: dense block output (batch=32, 1024 channels)
    {"batch_size": 32, "dim": 1024},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("activations", "2_LeakyReLU")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["dim"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
