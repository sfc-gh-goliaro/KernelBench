import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs Argmin or Argmax over a specified dimension.
    Both operations use identical reduction kernels that track value and index,
    differing only in the comparison operator.
    """
    def __init__(self, dim: int, mode: str = "max"):
        """
        Initializes the model with the dimension and mode for arg reduction.

        Args:
            dim (int): The dimension to perform arg reduction over.
            mode (str): Either "min" or "max" to select argmin or argmax.
        """
        super(Model, self).__init__()
        self.dim = dim
        self.mode = mode
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies argmin or argmax over the specified dimension.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with indices of min/max values.
        """
        if self.mode == "max":
            return torch.argmax(x, dim=self.dim)
        else:
            return torch.argmin(x, dim=self.dim)


PARAMETERS = [
    {"batch_size": 128, "dim1": 4096, "dim2": 4095},
    # Llama-3.1-8B: top-k token selection indices
    {"batch_size": 32, "dim1": 4096, "dim2": 128256},
    # GPT-NeoX-20B: vocabulary argmax for generation
    {"batch_size": 16, "dim1": 6144, "dim2": 50432},
    # Gemma-7B: attention head selection
    {"batch_size": 64, "dim1": 3072, "dim2": 8192},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("reductions", "3_ArgMinMax")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["dim1"], p["dim2"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return [1, "max"]
