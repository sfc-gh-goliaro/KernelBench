import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B)
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return torch.matmul(A.T, B)


PARAMETERS = [
    {"M": 1024 * 2, "K": 4096 * 2, "N": 2048 * 2},
    # Llama-3.1-8B: attention output projection (hidden=4096, heads*head_dim=4096)
    {"M": 4096, "K": 4096, "N": 4096},
    # Llama-3.1-70B: FFN down projection (hidden=8192, intermediate=28672)
    {"M": 8192, "K": 28672, "N": 8192},
    # Mistral-7B: FFN gate projection (hidden=4096, intermediate=14336)
    {"M": 14336, "K": 4096, "N": 4096},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("matmul", "7_TransposedA")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    A = DISTRIBUTIONS[dist_name]((p["K"], p["M"]), dtype=dtype, device=device)
    B = DISTRIBUTIONS[dist_name]((p["K"], p["N"]), dtype=dtype, device=device)
    return [A, B]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
