import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes Kullback-Leibler Divergence for comparing two distributions.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        return torch.nn.functional.kl_div(torch.log(predictions), targets, reduction='batchmean')


PARAMETERS = [
    {"batch_size": 8192 * 2, "input_shape": (8192 * 2,), "dim": 1},
    # BERT-base distillation: vocab_size=30522
    {"batch_size": 4096, "input_shape": (30522,), "dim": 1},
    # GPT-2: knowledge distillation (vocab_size=50257)
    {"batch_size": 2048, "input_shape": (50257,), "dim": 1},
    # DistilBERT: teacher-student training (vocab_size=30522)
    {"batch_size": 8192, "input_shape": (30522,), "dim": 1},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("loss", "4_KLDivLoss")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    scale = DISTRIBUTIONS[dist_name](((), dtype=dtype, device=device)
    return [(torch.rand(batch_size, *input_shape)*scale).softmax(dim=-1), torch.rand(batch_size, *input_shape).softmax(dim=-1)]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
