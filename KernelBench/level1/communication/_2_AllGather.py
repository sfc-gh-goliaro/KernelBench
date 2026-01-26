import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn
import torch.distributed as dist

class Model(nn.Module):
    """
    AllGather (Distributed)

    Used by: Tensor parallel (weight gathering), sequence parallel

    Gather tensor shards from all ranks into full tensor on each rank.
    Uses NCCL backend for GPU tensors.

    Shapes:
        Input: (batch, seq_len, hidden_size // world_size) per rank
        Output: (batch, seq_len, hidden_size) gathered tensor
    """

    def __init__(self, gather_dim: int = -1, process_group=None):
        """
        Initialize AllGather.

        Args:
            gather_dim: Dimension to gather along
            process_group: The process group to work on. If None, uses default group
        """
        super(Model, self).__init__()
        self.gather_dim = gather_dim
        self.process_group = process_group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform AllGather operation.

        Args:
            x: Local tensor shard

        Returns:
            Gathered full tensor (concatenated along gather_dim)
        """
        world_size = dist.get_world_size(self.process_group)

        # Create output tensor list for gathering
        gathered_tensors = [torch.empty_like(x) for _ in range(world_size)]

        # Perform all-gather
        dist.all_gather(gathered_tensors, x, group=self.process_group)

        # Concatenate along gather dimension
        output = torch.cat(gathered_tensors, dim=self.gather_dim)

        return output


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "hidden_size": 4096},
    # Llama-3.1-8B: tensor parallel weight gathering
    {"batch_size": 8, "seq_length": 4096, "hidden_size": 4096},
    # Llama-3.1-70B: tensor parallel weight gathering
    {"batch_size": 4, "seq_length": 4096, "hidden_size": 8192},
    # Mistral-7B: sequence parallel
    {"batch_size": 16, "seq_length": 2048, "hidden_size": 4096},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("communication", "2_AllGather")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_length"], shard_size), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return [-1]
