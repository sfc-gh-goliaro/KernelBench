import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs RMS Normalization.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(Model, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)
        return x / rms

# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 112, "features": 64, "dim1": 512, "dim2": 512},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("normalization", "4_RMSNorm")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["features"], p["dim1"], p["dim2"])
    x = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["features"]]
