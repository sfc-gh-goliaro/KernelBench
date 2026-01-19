import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn
import torch.distributed as dist

class Model(nn.Module):
    """
    AllToAll (Distributed)

    Used by: Expert parallel MoE

    Transpose/exchange data between ranks. Each rank sends different
    data to each other rank. Used for MoE expert parallelism.

    Shapes:
        Input: (world_size, tokens_per_rank, hidden_size)
        Output: (world_size, tokens_per_rank, hidden_size) transposed across ranks
    """

    def __init__(self, process_group=None):
        """
        Initialize AllToAll.

        Args:
            process_group: The process group to work on. If None, uses default group
        """
        super(Model, self).__init__()
        self.process_group = process_group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform AllToAll operation.

        In MoE, this exchanges tokens between expert-parallel ranks
        so each rank processes all tokens for its assigned experts.

        Args:
            x: Input tensor (world_size, tokens_per_rank, hidden_size)
               where first dim represents data destined for each rank

        Returns:
            Transposed tensor where each rank now has data from all other ranks
        """
        world_size = dist.get_world_size(self.process_group)

        # Verify input shape matches world size
        assert x.shape[0] == world_size, \
            f"First dimension ({x.shape[0]}) must match world_size ({world_size})"

        # Split input into chunks for each destination rank
        input_chunks = list(x.chunk(world_size, dim=0))
        input_chunks = [chunk.contiguous() for chunk in input_chunks]

        # Create output tensors for data from each source rank
        output_chunks = [torch.empty_like(chunk) for chunk in input_chunks]

        # Perform all-to-all
        dist.all_to_all(output_chunks, input_chunks, group=self.process_group)

        # Stack output chunks
        output = torch.cat(output_chunks, dim=0)

        return output


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"tokens_per_rank": 2048, "hidden_size": 4096},
    # Mixtral-8x7B: MoE expert parallelism
    {"tokens_per_rank": 4096, "hidden_size": 4096},
    # DeepSeek-V2: MoE expert parallelism
    {"tokens_per_rank": 2048, "hidden_size": 5120},
    # DeepSeek-V3: MoE expert parallelism
    {"tokens_per_rank": 4096, "hidden_size": 7168},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("communication", "4_AllToAll")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((world_size, p["tokens_per_rank"], p["hidden_size"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
