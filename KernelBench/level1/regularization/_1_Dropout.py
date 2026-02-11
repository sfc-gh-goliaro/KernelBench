"""
Dropout Regularization.

Dropout(x, p) randomly zeroes elements with probability p during training,
scaling remaining elements by 1/(1-p).

Used by: Transformers (attention dropout, FFN dropout), diffusion models,
         ResNet blocks, most neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Dropout regularization (parameter-free).

    During training, randomly zeroes elements with probability p.
    During evaluation, passes input through unchanged.

    Shapes:
        Input: (*) any shape
        Output: (*) same shape as input
    """

    def __init__(self, p: float = 0.5, inplace: bool = False):
        """
        Initialize Dropout.

        Args:
            p: Probability of an element being zeroed. Default: 0.5
            inplace: If True, do operation in-place. Default: False
        """
        super(Model, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply dropout to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with dropout applied (training only),
                          same shape as input.
        """
        return F.dropout(x, p=self.p, training=self.training, inplace=self.inplace)
