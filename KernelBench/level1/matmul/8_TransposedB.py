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
        return torch.matmul(A, B.T)


PARAMETERS = [
    {"M": 1024 * 2, "K": 4096 * 2, "N": 2048 * 2},
    # Llama-3.1-8B: QKV projection (seq_len=4096, hidden=4096, qkv_dim=4096*3)
    {"M": 4096, "K": 4096, "N": 12288},
    # GPT-3 175B: FFN up projection (hidden=12288, intermediate=49152)
    {"M": 4096, "K": 12288, "N": 49152},
    # Mistral-7B: attention key projection (seq_len=8192, hidden=4096)
    {"M": 8192, "K": 4096, "N": 4096},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("matmul", "8_TransposedB")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    A = DISTRIBUTIONS[dist_name]((p["M"], p["K"]), dtype=dtype, device=device)
    B = DISTRIBUTIONS[dist_name]((p["N"], p["K"]), dtype=dtype, device=device)
    return [A, B]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
