import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Layer Normalization.
    """
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True):
        """
        Initializes the LayerNorm layer.

        Args:
            normalized_shape: Shape of the input tensor to be normalized.
                Can be an int, tuple, or list.
            eps: A small value added for numerical stability.
            elementwise_affine: Whether to include learnable affine parameters.
        """
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine
        )

    @property
    def weight(self):
        """Access the weight parameter."""
        return self.ln.weight
    
    @property
    def bias(self):
        """Access the bias parameter."""
        return self.ln.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        return self.ln(x)


PARAMETERS = [
    {"batch_size": 16, "features": 64, "dim1": 256, "dim2": 256},
    # T5-Base: hidden_size=768
    {"batch_size": 16, "features": 768, "dim1": 1, "dim2": 512},
    # T5-Large: hidden_size=1024
    {"batch_size": 8, "features": 1024, "dim1": 1, "dim2": 512},
    # T5-3B: hidden_size=1024, intermediate_size=16384
    {"batch_size": 4, "features": 1024, "dim1": 1, "dim2": 512},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("normalization", "6_LayerNorm")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["features"], p["dim1"], p["dim2"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [(features, p["dim1"], dim2)]
