import os
import sys
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Wide Component (Wide & Deep)
    
    Used by: WDL (Wide & Deep Learning)
    
    Wide linear component with cross-product feature transformations.
    
    Shapes:
        Input: (batch, input_dim)
        Output: (batch, 1)
    """
    
    def __init__(self, input_dim: int):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
