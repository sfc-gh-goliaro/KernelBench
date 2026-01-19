import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes the Mean Squared Error loss for regression tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        return torch.mean((predictions - targets) ** 2)


PARAMETERS = [
    {"batch_size": 32768, "input_shape": (32768,), "dim": 1},
    # BERT-base: hidden state regression (768 dim embeddings)
    {"batch_size": 8192, "input_shape": (768,), "dim": 1},
    # ResNet-50: regression head training (2048 dim features)
    {"batch_size": 4096, "input_shape": (2048,), "dim": 1},
    # GPT-2-medium: hidden state prediction (1024 dim)
    {"batch_size": 2048, "input_shape": (1024,), "dim": 1},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("loss", "1_MSELoss")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    scale = DISTRIBUTIONS[dist_name](((), dtype=dtype, device=device)
    return [torch.rand(batch_size, *input_shape)*scale, torch.rand(batch_size, *input_shape)]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
