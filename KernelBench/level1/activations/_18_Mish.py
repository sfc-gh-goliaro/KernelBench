"""
Mish Activation Function.

Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))

Used by: YOLOv4/v5, some diffusion model variants, TimestepEmbedding (diffusers).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Mish activation function (parameter-free).

    Uses PyTorch's optimized F.mish() implementation.

    Shapes:
        Input: (*) any shape
        Output: (*) same shape as input
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Mish activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Mish applied, same shape as input.
        """
        return F.mish(x)
