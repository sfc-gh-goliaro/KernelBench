import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) where one of the matrices is tall and skinny
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A, B):
        return torch.matmul(A, B)

# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"M": 32768, "N": 32},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("matmul", "9_TallSkinny")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    A = DISTRIBUTIONS[dist_name]((p["M"], p["N"]), dtype=dtype, device=device)
    B = DISTRIBUTIONS[dist_name]((p["N"], p["M"]), dtype=dtype, device=device)
    return [A, B]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
