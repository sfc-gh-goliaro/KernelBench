import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs Min or Max reduction over a specified dimension.
    The only difference between min and max is the comparison operator,
    so they share the same parallel reduction kernel structure.
    """
    def __init__(self, dim: int, mode: str = "max"):
        """
        Initializes the model with the dimension to reduce over and the reduction mode.

        Args:
            dim (int): The dimension to reduce over.
            mode (str): Either "min" or "max" to select the reduction operation.
        """
        super(Model, self).__init__()
        self.dim = dim
        self.mode = mode
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies min or max reduction over the specified dimension.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after reduction over the specified dimension.
        """
        if self.mode == "max":
            return torch.max(x, dim=self.dim)[0]
        else:
            return torch.min(x, dim=self.dim)[0]


PARAMETERS = [
    {"batch_size": 128, "dim1": 4096, "dim2": 4095},
    # Llama-3.1-8B: max pooling over hidden states
    {"batch_size": 32, "dim1": 4096, "dim2": 8192},
    # BERT-Large: min/max normalization bounds
    {"batch_size": 64, "dim1": 1024, "dim2": 512},
    # Falcon-40B: activation clamping reduction
    {"batch_size": 8, "dim1": 8192, "dim2": 16384},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("reductions", "2_MinMax")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["dim1"], p["dim2"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return [1, "max"]
