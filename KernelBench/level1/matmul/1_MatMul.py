import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Matrix Multiplication (C = A @ B)

    This is the canonical matmul that handles all shape variants:
    - SquareMatMul: M=K=N (absorbed)
    - StandardMatMul: General M, K, N (this file)
    - MatrixVector: N=1 (absorbed)
    - MatrixScalar: M=N=1 (absorbed)
    - LargeK, SmallK: Different K dimensions (absorbed)
    - IrregularShapes: Non-power-of-2 dimensions (absorbed)
    - TallSkinny: M >> N or N >> M (absorbed)
    - 3DTensorMatMul, 4DTensorMatMul: Higher-rank tensors (use BatchedMatMul)

    A general matmul kernel handles all these cases; shape differences
    affect tiling strategy but not the fundamental algorithm.

    Shapes:
        A: (M, K)
        B: (K, N)
        Output: (M, N)
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
        return torch.matmul(A, B)



PARAMETERS = [
    {"M": 2048, "K": 4096, "N": 2048},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("matmul", "1_MatMul")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    A = DISTRIBUTIONS[dist_name]((p["M"], p["K"]), dtype=dtype, device=device)
    B = DISTRIBUTIONS[dist_name]((p["K"], p["N"]), dtype=dtype, device=device)
    return [A, B]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
