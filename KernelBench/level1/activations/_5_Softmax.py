import os
import sys
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a Softmax activation.
    
    Args:
        dim: Dimension along which Softmax will be computed. Default: -1 (last dim).
    """
    def __init__(self, dim: int = -1):
        super(Model, self).__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        return torch.softmax(x, dim=self.dim)
