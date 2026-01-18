import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes Triplet Margin Loss for metric learning tasks.

    Parameters:
        margin (float): The margin between the positive and negative samples.
    """
    def __init__(self, margin=1.0):
        super(Model, self).__init__()
        self.loss_fn = torch.nn.TripletMarginLoss(margin=margin)

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)


PARAMETERS = [
    {"batch_size": 32768, "input_shape": (8192,), "dim": 1},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("loss", "5_TripletMarginLoss")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    scale = DISTRIBUTIONS[dist_name](((), dtype=dtype, device=device)
    return [torch.rand(batch_size, *input_shape)*scale, torch.rand(batch_size, *input_shape), torch.rand(batch_size, *input_shape)]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return [1.0]
