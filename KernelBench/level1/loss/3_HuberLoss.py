import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes Smooth L1 (Huber) Loss for regression tasks.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        return torch.nn.functional.smooth_l1_loss(predictions, targets)

# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 32768, "dim": 32768},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("loss", "3_HuberLoss")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["dim"])
    predictions = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    targets = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [predictions, targets]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
