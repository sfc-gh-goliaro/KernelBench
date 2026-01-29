"""
Swish / SiLU Activation Function.

Swish(x) = x * sigmoid(x) = SiLU(x)

Used by: Llama, GPT-NeoX, and many modern transformers.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Swish (SiLU) activation function.
    
    Uses PyTorch's optimized F.silu() implementation for best performance
    and numerical precision (especially important for bf16).
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Swish/SiLU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Swish applied, same shape as input.
        """
        return F.silu(x)
