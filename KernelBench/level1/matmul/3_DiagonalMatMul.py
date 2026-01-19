import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): A 1D tensor representing the diagonal of the diagonal matrix. Shape: (N,).
            B (torch.Tensor): A 2D tensor representing the second matrix. Shape: (N, M).

        Returns:
            torch.Tensor: The result of the matrix multiplication. Shape: (N, M).
        """
        return torch.diag(A) @ B


PARAMETERS = [
    {"M": 4096, "N": 4096},
    # Llama-3.1-8B: diagonal scaling in RMSNorm (hidden_size=4096)
    {"M": 4096, "N": 4096},
    # Llama-3.1-70B: diagonal scaling in RMSNorm (hidden_size=8192)
    {"M": 8192, "N": 8192},
    # GPT-3 175B: diagonal scaling (hidden_size=12288)
    {"M": 12288, "N": 12288},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("matmul", "3_DiagonalMatMul")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    A = DISTRIBUTIONS[dist_name]((p["N"]), dtype=dtype, device=device)
    B = DISTRIBUTIONS[dist_name]((p["N"], p["M"]), dtype=dtype, device=device)
    return [A, B]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
