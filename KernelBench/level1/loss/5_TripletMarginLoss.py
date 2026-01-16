import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes Triplet Margin Loss for metric learning tasks.
    """
    def __init__(self, margin=1.0):
        super(Model, self).__init__()
        self.loss_fn = torch.nn.TripletMarginLoss(margin=margin)

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)

# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 32768, "dim": 8192, "margin": 1.0},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("loss", "5_TripletMarginLoss")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["dim"])
    anchor = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    positive = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    negative = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [anchor, positive, negative]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["margin"]]
