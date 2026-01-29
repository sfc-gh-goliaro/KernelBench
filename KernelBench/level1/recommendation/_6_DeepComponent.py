import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Deep Component (Wide & Deep / DeepFM)
    
    Used by: WDL, DeepFM
    
    Deep MLP for learning non-linear feature interactions.
    
    Shapes:
        Input: (batch, input_dim)
        Output: (batch, output_dim)
    """
    
    def __init__(self, input_dim: int, hidden_dims: list = [256, 128], output_dim: int = 1):
        super(Model, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU(), nn.Dropout(0.1)])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
