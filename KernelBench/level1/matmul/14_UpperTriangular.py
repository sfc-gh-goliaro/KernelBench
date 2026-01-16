import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs matrix multiplication (C = A * B) for upper triangular matrices.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A, B):
        return torch.triu(torch.matmul(A, B))

# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"N": 4096},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("matmul", "14_UpperTriangular")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    A = torch.triu(DISTRIBUTIONS[dist_name]((p["N"], p["N"]), dtype=dtype, device=device))
    B = torch.triu(DISTRIBUTIONS[dist_name]((p["N"], p["N"]), dtype=dtype, device=device))
    return [A, B]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
