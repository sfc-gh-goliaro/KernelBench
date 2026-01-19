import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs sum reduction over a specified dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies sum reduction over the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        return torch.sum(x, dim=self.dim, keepdim=True)


PARAMETERS = [
    {"batch_size": 128, "dim1": 4096, "dim2": 4095, "reduce_dim": 1},
    # Llama-3.1-8B: hidden state reduction across sequence
    {"batch_size": 32, "dim1": 4096, "dim2": 8192, "reduce_dim": 1},
    # GPT-2-XL: attention score aggregation
    {"batch_size": 64, "dim1": 1600, "dim2": 2048, "reduce_dim": 2},
    # Mistral-7B: MLP intermediate sum
    {"batch_size": 16, "dim1": 14336, "dim2": 4096, "reduce_dim": 1},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("reductions", "1_Sum")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["dim1"], p["dim2"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["reduce_dim"]]
