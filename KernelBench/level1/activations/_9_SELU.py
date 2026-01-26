import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a SELU activation.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies SELU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with SELU applied, same shape as input.
        """
        return torch.selu(x)
    

PARAMETERS = [
    {"batch_size": 4096, "dim": 393216},
    # ResNet-50: self-normalizing layers
    {"batch_size": 32, "dim": 2048},
    # ResNet-101: self-normalizing layers
    {"batch_size": 16, "dim": 2048},
    # Self-normalizing network: MLP hidden layer (batch=64, dim=4096)
    {"batch_size": 64, "dim": 4096},
    # Transformer FFN alternative: intermediate layer (batch=32, dim=3072)
    {"batch_size": 32, "dim": 3072},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("activations", "9_SELU")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["dim"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
