import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes Hinge Loss for binary classification tasks.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        return torch.mean(torch.clamp(1 - predictions * targets, min=0))

# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 32768, "dim": 32768},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("loss", "6_HingeLoss")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["dim"])
    predictions = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    # Targets for hinge loss should be -1 or 1
    targets = (DISTRIBUTIONS["indices"]((p["batch_size"],), 2, dtype=torch.int64, device=device).float() * 2 - 1).unsqueeze(1).expand(-1, p["dim"])
    return [predictions, targets]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
