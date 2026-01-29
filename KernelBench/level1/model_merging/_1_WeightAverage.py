import os
import sys
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Weight Averaging
    
    Used by: Model soup, ensemble
    
    Simple weight averaging: (w1 + w2 + ... + wn) / n
    
    Shapes:
        weights: list of (param_shape) tensors
        Output: (param_shape) averaged
    """
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, *weights: torch.Tensor) -> torch.Tensor:
        return sum(weights) / len(weights)
