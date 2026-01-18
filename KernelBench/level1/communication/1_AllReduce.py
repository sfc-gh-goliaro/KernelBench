import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn
import torch.distributed as dist

class Model(nn.Module):
    """
    AllReduce (Distributed)

    Used by: Tensor parallel (post-GEMM reduction), data parallel gradient sync

    Sum tensor values across all distributed ranks, result replicated
    on all ranks. Uses NCCL backend for GPU tensors.

    Shapes:
        Input: (batch, seq_len, hidden_size)
        Output: (batch, seq_len, hidden_size) - sum across all ranks
    """

    def __init__(self, process_group=None, async_op: bool = False):
        """
        Initialize AllReduce.

        Args:
            process_group: The process group to work on. If None, uses default group
            async_op: Whether to perform asynchronous operation
        """
        super(Model, self).__init__()
        self.process_group = process_group
        self.async_op = async_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform AllReduce operation.

        Args:
            x: Input tensor (batch_size, seq_len, hidden_size)

        Returns:
            Reduced tensor (sum across all ranks)
        """
        # Clone to avoid modifying input in-place
        output = x.clone()

        # Perform all-reduce (sum by default)
        handle = dist.all_reduce(output, op=dist.ReduceOp.SUM,
                                  group=self.process_group, async_op=self.async_op)

        if self.async_op and handle is not None:
            handle.wait()

        return output


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "hidden_size": 4096},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("communication", "1_AllReduce")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_length"], p["hidden_size"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
