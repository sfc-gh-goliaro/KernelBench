import os
import sys
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Vector Norm Normalization

    Computes p-norm of tensors and normalizes by it. This consolidates:
    - L1Norm: p=1, computes sum of absolute values
    - L2Norm: p=2, computes Euclidean norm
    - FrobeniusNorm: p='fro', equivalent to L2 norm on flattened matrix

    All use the same reduction kernel structure with different power operations.

    Shapes:
        Input: Arbitrary shape tensor
        Output: Same shape as input, normalized
    """

    def __init__(self, p: float = 2.0, dim: int = None, keepdim: bool = True):
        """
        Initialize VectorNorm layer.

        Args:
            p: The order of the norm (1, 2, or 'fro' for Frobenius)
            dim: Dimension along which to compute norm. None for global norm.
            keepdim: Whether to keep the reduced dimension.
        """
        super(Model, self).__init__()
        self.p = p
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply p-norm normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Normalized output tensor, same shape as input.
        """
        if self.dim is None:
            # Global norm (Frobenius-style)
            norm = torch.norm(x, p=self.p)
        else:
            norm = torch.norm(x, p=self.p, dim=self.dim, keepdim=self.keepdim)

        return x / (norm + 1e-12)
