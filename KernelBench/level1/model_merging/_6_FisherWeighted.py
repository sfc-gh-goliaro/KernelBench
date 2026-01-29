import os
import sys
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Fisher-Weighted Averaging
    
    Used by: Fisher merging
    
    Fisher-weighted averaging using Fisher information matrix.
    
    Shapes:
        weights: list of (param_shape) weight tensors
        fishers: list of (param_shape) Fisher information
        Output: (param_shape) merged
    """
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, weights: list, fishers: list) -> torch.Tensor:
        # Weighted average: sum(w_i * F_i) / sum(F_i)
        numerator = sum(w * f for w, f in zip(weights, fishers))
        denominator = sum(fishers) + 1e-10
        return numerator / denominator
