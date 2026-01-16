import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        return out

# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 32, "num_heads": 32, "sequence_length": 512, "embedding_dimension": 1024},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("attention", "1_ScaledDotProductAttention")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["num_heads"], p["sequence_length"], p["embedding_dimension"])
    Q = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    K = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    V = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [Q, K, V]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
