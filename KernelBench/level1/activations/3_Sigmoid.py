import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a Sigmoid activation.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Sigmoid activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Sigmoid applied, same shape as input.
        """
        return torch.sigmoid(x)


PARAMETERS = [
    {"batch_size": 4096, "dim": 393216},
    # EfficientNet-B0: SE gate (batch=32, 8 channels for squeeze)
    {"batch_size": 32, "dim": 8},
    # EfficientNet-B7: SE gate (batch=8, 16 channels for squeeze)
    {"batch_size": 8, "dim": 16},
    # ResNet-50: gating mechanism
    {"batch_size": 32, "dim": 2048},
    # MobileNetV3-Large: SE gate (batch=32, 576 channels)
    {"batch_size": 32, "dim": 576},
    # BERT-Large: gating in attention (batch=16, hidden_size=1024)
    {"batch_size": 16, "dim": 1024},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("activations", "3_Sigmoid")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["dim"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
