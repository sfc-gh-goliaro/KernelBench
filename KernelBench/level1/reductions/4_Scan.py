import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs parallel prefix scan operations (cumulative sum or product).
    All scan variants use the same Blelloch-style parallel algorithm structure,
    differing only in the binary operation (+ for cumsum, * for cumprod).

    This consolidates: Cumsum, Cumprod, CumsumReverse, CumsumExclusive, MaskedCumsum
    """

    def __init__(self, dim: int, op: str = "sum", reverse: bool = False, exclusive: bool = False):
        """
        Initialize the Scan model.

        Args:
            dim (int): The dimension along which to perform the scan.
            op (str): The binary operation - "sum" for cumsum, "prod" for cumprod.
            reverse (bool): If True, perform scan in reverse order.
            exclusive (bool): If True, perform exclusive scan (exclude current element).
        """
        super(Model, self).__init__()
        self.dim = dim
        self.op = op
        self.reverse = reverse
        self.exclusive = exclusive
        if op not in ("sum", "prod"):
            raise ValueError(f"op must be 'sum' or 'prod', got {op}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass, computing the scan along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor of the same shape as x after applying the scan.
        """
        if self.reverse:
            x = torch.flip(x, dims=[self.dim])

        if self.op == "sum":
            result = torch.cumsum(x, dim=self.dim)
        else:
            result = torch.cumprod(x, dim=self.dim)

        if self.exclusive:
            # Shift result and pad with identity element (0 for sum, 1 for prod)
            identity = 0.0 if self.op == "sum" else 1.0
            # Create padding
            pad_shape = list(result.shape)
            pad_shape[self.dim] = 1
            pad = torch.full(pad_shape, identity, dtype=result.dtype, device=result.device)
            # Slice off last element and prepend identity
            slices = [slice(None)] * result.dim()
            slices[self.dim] = slice(None, -1)
            result = torch.cat([pad, result[tuple(slices)]], dim=self.dim)

        if self.reverse:
            result = torch.flip(result, dims=[self.dim])

        return result

# Define input dimensions and parameters

PARAMETERS = [
    {"batch_size": 32768, "input_shape": (32768,), "dim": 1},
    # Mamba-2.8B: selective state space cumulative sum
    {"batch_size": 2048, "input_shape": (4096,), "dim": 1},
    # RWKV-7B: linear attention cumulative product
    {"batch_size": 1024, "input_shape": (8192,), "dim": 1},
    # S4-Large: structured state space scan
    {"batch_size": 4096, "input_shape": (2048,), "dim": 1},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("reductions", "4_Scan")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["input_shape"])
    x = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["dim"], "sum", False, False]
