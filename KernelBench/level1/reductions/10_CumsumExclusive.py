import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs an exclusive cumulative sum (does not include the current element).
    """

    def __init__(self, dim):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x):
        cumsum = torch.cumsum(x.narrow(dim=self.dim, start=0, length=x.size(self.dim)-1), dim=self.dim)
        return torch.cat((torch.zeros_like(x.select(self.dim, 0).unsqueeze(self.dim)), cumsum), dim=self.dim)

# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 32768, "input_dim": 32768, "reduce_dim": 1},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("reductions", "10_CumsumExclusive")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["input_dim"])
    x = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["reduce_dim"]]
