import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes Hinge Loss for binary classification tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        return torch.mean(torch.clamp(1 - predictions * targets, min=0))


PARAMETERS = [
    {"batch_size": 32768, "input_shape": (32768,), "dim": 1},
    # ResNet-50: binary classification head (2048 features)
    {"batch_size": 4096, "input_shape": (2048,), "dim": 1},
    # BERT-base: binary text classification (768 dim CLS token)
    {"batch_size": 2048, "input_shape": (768,), "dim": 1},
    # EfficientNet-B0: binary classifier (1280 features)
    {"batch_size": 8192, "input_shape": (1280,), "dim": 1},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("loss", "6_HingeLoss")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["input_shape"])
    x = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
