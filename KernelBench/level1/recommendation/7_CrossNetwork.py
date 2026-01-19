import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Cross Network Layer (DCN)
    
    Used by: DCN (Deep & Cross Network)
    
    Cross layer: x_0 * x_l^T * w + b + x_l for explicit crossing.
    
    Shapes:
        Input: (batch, input_dim)
        Output: (batch, input_dim)
    """
    
    def __init__(self, input_dim: int, num_layers: int = 3):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(input_dim, 1) * 0.01) for _ in range(num_layers)])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        xl = x
        for w, b in zip(self.weights, self.biases):
            # x_l+1 = x_0 * (x_l^T * w) + b + x_l
            xl = x0 * (xl @ w) + b + xl
        return xl



PARAMETERS = [
    {"batch_size": 4096, "input_dim": 416},
    # DCN: cross network for Criteo dataset
    {"batch_size": 2048, "input_dim": 512},
    # DCN-v2: mixture-of-experts cross layer
    {"batch_size": 1024, "input_dim": 1024},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("recommendation", "7_CrossNetwork")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["input_dim"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["input_dim"]]
