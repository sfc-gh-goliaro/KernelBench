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


M = 2048
K = 4096
N = 2048

def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, N)
    return [A, B]

def get_init_inputs():
    return []
