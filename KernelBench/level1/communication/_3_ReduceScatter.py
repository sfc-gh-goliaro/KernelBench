import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn
import torch.distributed as dist

class Model(nn.Module):
    """
    ReduceScatter (Distributed)

    Used by: Sequence parallel (gradient sync), tensor parallel

    Reduce (sum) then scatter result so each rank gets different shard.
    Uses NCCL backend for GPU tensors.

    Shapes:
        Input: (batch, seq_len, hidden_size) per rank
        Output: (batch, seq_len, hidden_size // world_size) - this rank's shard
    """

    def __init__(self, scatter_dim: int = -1, process_group=None):
        """
        Initialize ReduceScatter.

        Args:
            scatter_dim: Dimension to scatter along
            process_group: The process group to work on. If None, uses default group
        """
        super(Model, self).__init__()
        self.scatter_dim = scatter_dim
        self.process_group = process_group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform ReduceScatter operation.

        Args:
            x: Input tensor (same on all ranks, will be reduced and scattered)

        Returns:
            This rank's shard of the reduced result
        """
        world_size = dist.get_world_size(self.process_group)

        # Compute output shard size
        input_shape = list(x.shape)
        scatter_dim = self.scatter_dim if self.scatter_dim >= 0 else len(input_shape) + self.scatter_dim
        shard_size = input_shape[scatter_dim] // world_size

        # Create output tensor for this rank's shard
        output_shape = input_shape.copy()
        output_shape[scatter_dim] = shard_size
        output = torch.empty(output_shape, dtype=x.dtype, device=x.device)

        # Split input into chunks for each rank
        input_chunks = list(x.chunk(world_size, dim=scatter_dim))

        # Perform reduce-scatter
        dist.reduce_scatter(output, input_chunks, op=dist.ReduceOp.SUM,
                           group=self.process_group)

        return output


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "hidden_size": 4096},
    # Llama-3.1-8B: sequence parallel gradient sync
    {"batch_size": 8, "seq_length": 4096, "hidden_size": 4096},
    # Llama-3.1-70B: sequence parallel gradient sync
    {"batch_size": 4, "seq_length": 4096, "hidden_size": 8192},
    # DeepSeek-V2: tensor parallel reduce-scatter
    {"batch_size": 8, "seq_length": 2048, "hidden_size": 5120},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("communication", "3_ReduceScatter")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_length"], p["hidden_size"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return [-1]
