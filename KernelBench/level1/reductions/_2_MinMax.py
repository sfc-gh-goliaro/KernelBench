import os
import sys
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
