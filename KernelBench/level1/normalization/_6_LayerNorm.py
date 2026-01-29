import os
import sys
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
