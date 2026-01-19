import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
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



PARAMETERS = [
    {"batch_size": 4096, "input_dim": 416},
    # DLRM: bottom MLP for dense feature processing
    {"batch_size": 2048, "input_dim": 512},
    # DeepFM: deep component for Criteo CTR
    {"batch_size": 1024, "input_dim": 1024},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("recommendation", "6_DeepComponent")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["input_dim"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["input_dim"]]
