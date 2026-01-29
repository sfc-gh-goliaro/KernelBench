import os
import sys
import torch
import torch.nn as nn
import math

class Model(nn.Module):
    """
    Linear Layer (Y = X @ W.T + b)
    
    Used by: All transformer projection layers (QKV, output, FFN)
    
    This is distinct from MatMul because:
    - Weight is a learnable parameter (not a runtime input)
    - Weight is stored transposed (out_features, in_features)
    - Optional bias addition
    
    Shapes:
        X: (*, in_features)
        W: (out_features, in_features) - learnable parameter
        b: (out_features,) - optional learnable parameter
        Output: (*, out_features)
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        """
        Initialize Linear layer.
        
        Args:
            in_features: Size of each input sample
            out_features: Size of each output sample
            bias: If True, adds a learnable bias. Default: False (most LLMs don't use bias)
        """
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply linear transformation.
        
        Args:
            x: Input tensor of shape (*, in_features)
            
        Returns:
            Output tensor of shape (*, out_features)
        """
        # Use F.linear for numerical consistency with PyTorch's nn.Linear
        return torch.nn.functional.linear(x, self.weight, self.bias)
