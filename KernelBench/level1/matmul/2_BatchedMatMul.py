import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs batched matrix multiplication (C = A * B) where A, B, and C have the same batch dimension.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication.

        Args:
            A: Input tensor of shape (batch_size, m, k).
            B: Input tensor of shape (batch_size, k, n).

        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        return torch.bmm(A, B)


PARAMETERS = [
    {"batch_size": 128, "m": 128 * 4, "k": 256 * 4, "n": 512 * 4},
    # Llama-3.1-8B attention: batch=8*32=256, seq=4096, head_dim=128
    {"batch_size": 256, "m": 4096, "k": 128, "n": 4096},
    # Llama-3.1-70B attention: batch=4*64=256, seq=4096, head_dim=128
    {"batch_size": 256, "m": 4096, "k": 128, "n": 4096},
    # Mistral-7B attention: batch=8*32=256, seq=4096, head_dim=128
    {"batch_size": 256, "m": 4096, "k": 128, "n": 4096},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("matmul", "2_BatchedMatMul")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    A = DISTRIBUTIONS[dist_name]((p["batch_size"], p["m"], p["k"]), dtype=dtype, device=device)
    B = DISTRIBUTIONS[dist_name]((p["batch_size"], p["k"], p["n"]), dtype=dtype, device=device)
    return [A, B]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
