import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a masked cumulative sum, only summing elements that satisfy a condition.
    """

    def __init__(self, dim):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x, mask):
        return torch.cumsum(x * mask, dim=self.dim)

# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 32768, "input_dim": 32768, "reduce_dim": 1},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("reductions", "11_MaskedCumsum")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["input_dim"])
    x = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    mask = torch.randint(0, 2, shape, device=device).bool()
    return [x, mask]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["reduce_dim"]]
