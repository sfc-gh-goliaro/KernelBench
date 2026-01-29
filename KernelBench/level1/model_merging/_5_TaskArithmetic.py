import os
import sys
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Task Arithmetic
    
    Used by: Multi-task merging
    
    Task vectors: add/subtract task-specific weight deltas.
    
    Shapes:
        base: (param_shape) base weights
        deltas: list of (param_shape) task vectors
        Output: (param_shape) merged
    """
    
    def __init__(self, scaling: float = 1.0):
        super(Model, self).__init__()
        self.scaling = scaling
    
    def forward(self, base: torch.Tensor, *deltas: torch.Tensor) -> torch.Tensor:
        result = base.clone()
        for delta in deltas:
            result = result + self.scaling * delta
        return result
