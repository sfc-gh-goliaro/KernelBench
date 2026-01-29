import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a GELU activation.
    
    Args:
        approximate: Approximation method for GELU. Options:
            - 'none': Exact GELU using erf (default)
            - 'tanh': Tanh approximation using PyTorch's implementation
    """
    def __init__(self, approximate: str = 'none'):
        super(Model, self).__init__()
        self.approximate = approximate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies GELU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with GELU applied, same shape as input.
        """
        return torch.nn.functional.gelu(x, approximate=self.approximate)


PARAMETERS = [
    {"batch_size": 4096, "dim": 393216},
    # Qwen2-VL-7B vision encoder: vision_hidden=1280
    {"batch_size": 8, "dim": 1280},
    # Qwen2-VL-7B text: intermediate_size=18944
    {"batch_size": 8, "dim": 18944},
    # Qwen2-VL-2B: intermediate_size=8960
    {"batch_size": 16, "dim": 8960},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("activations", "8_GELU")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["dim"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
