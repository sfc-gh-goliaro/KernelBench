import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Factorization Machine
    
    Used by: DeepFM, FFM
    
    FM: linear + sum of pairwise feature interactions.
    Efficient O(n*k) computation via sum-of-squares trick.
    
    Shapes:
        Input: (batch, num_features, embed_dim)
        Output: (batch, 1)
    """
    
    def __init__(self, num_features: int, embed_dim: int):
        super(Model, self).__init__()
        self.linear = nn.Linear(num_features * embed_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Sum of squares trick for pairwise interactions
        sum_squared = x.sum(dim=1).pow(2).sum(dim=1, keepdim=True)
        squared_sum = x.pow(2).sum(dim=1).sum(dim=1, keepdim=True)
        interactions = 0.5 * (sum_squared - squared_sum)
        
        # Linear term
        linear = self.linear(x.view(batch_size, -1))
        
        return linear + interactions



PARAMETERS = [
    {"batch_size": 4096, "num_features": 26, "embed_dim": 16},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("recommendation", "3_FactorizationMachine")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["num_features"], p["embed_dim"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["num_features"], p["embed_dim"]]
