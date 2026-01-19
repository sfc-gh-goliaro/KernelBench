import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a HardSigmoid activation.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies HardSigmoid activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with HardSigmoid applied, same shape as input.
        """
        return torch.nn.functional.hardsigmoid(x)


PARAMETERS = [
    {"batch_size": 4096, "dim": 393216},
    # EfficientNet-B0: SE block squeeze (stem_channels=32, reduced by 4)
    {"batch_size": 32, "dim": 8},
    # EfficientNet-B4: SE block squeeze (stem_channels=48)
    {"batch_size": 16, "dim": 12},
    # EfficientNet-B7: SE block squeeze (stem_channels=64)
    {"batch_size": 8, "dim": 16},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("activations", "10_HardSigmoid")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["dim"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
