import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
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
        output = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            output = output + self.bias
        return output


PARAMETERS = [
    # General case
    {"in_features": 4096, "out_features": 4096, "batch_size": 8, "seq_len": 2048},
    # Llama-3.1-8B: QKV projection (hidden=4096, num_heads=32, head_dim=128)
    {"in_features": 4096, "out_features": 4096, "batch_size": 8, "seq_len": 2048},
    # Llama-3.1-8B: FFN up projection (hidden=4096, intermediate=14336)
    {"in_features": 4096, "out_features": 14336, "batch_size": 8, "seq_len": 2048},
    # Llama-3.1-8B: FFN down projection (intermediate=14336, hidden=4096)
    {"in_features": 14336, "out_features": 4096, "batch_size": 8, "seq_len": 2048},
    # Llama-3.1-70B: QKV projection (hidden=8192)
    {"in_features": 8192, "out_features": 8192, "batch_size": 4, "seq_len": 2048},
    # Llama-3.1-70B: FFN up projection (hidden=8192, intermediate=28672)
    {"in_features": 8192, "out_features": 28672, "batch_size": 4, "seq_len": 2048},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("matmul", "10_Linear")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_len"], p["in_features"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["in_features"], p["out_features"]]
