import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes Kullback-Leibler Divergence for comparing two distributions.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        return torch.nn.functional.kl_div(torch.log(predictions), targets, reduction='batchmean')

# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 16384, "dim": 16384},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("loss", "4_KLDivLoss")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["dim"])
    # KL divergence requires probability distributions (softmax outputs)
    predictions = DISTRIBUTIONS["uniform_pos"](shape, dtype=dtype, device=device).softmax(dim=-1)
    targets = DISTRIBUTIONS["uniform_pos"](shape, dtype=dtype, device=device).softmax(dim=-1)
    return [predictions, targets]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
