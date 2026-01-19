import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a LogSoftmax activation.
    """
    def __init__(self, dim: int = 1):
        super(Model, self).__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LogSoftmax activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with LogSoftmax applied, same shape as input.
        """
        return torch.log_softmax(x, dim=self.dim)


PARAMETERS = [
    {"batch_size": 4096, "dim": 393216},
    # Llama-3.1-8B: log probabilities over vocab_size=128256
    {"batch_size": 8, "dim": 128256},
    # Llama-3.1-70B: log probabilities over vocab_size=128256
    {"batch_size": 4, "dim": 128256},
    # T5-Base: log probabilities over vocab_size=32128
    {"batch_size": 16, "dim": 32128},
    # GPT-2: log probabilities over vocab_size=50257
    {"batch_size": 16, "dim": 50257},
    # BERT-Base: log probabilities over vocab_size=30522
    {"batch_size": 32, "dim": 30522},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("activations", "6_LogSoftmax")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["dim"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
