import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs 4D tensor-matrix multiplication: 
        C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A, B):
        return torch.einsum("bijl,lk->bijk", A, B)

# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"b": 8, "i": 256, "j": 512, "l": 256, "k": 768},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("matmul", "11_4DTensor")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    A = DISTRIBUTIONS[dist_name]((p["b"], p["i"], p["j"], p["l"]), dtype=dtype, device=device)
    B = DISTRIBUTIONS[dist_name]((p["l"], p["k"]), dtype=dtype, device=device)
    return [A, B]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
