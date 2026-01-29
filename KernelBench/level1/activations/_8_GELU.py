import os
import sys
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
