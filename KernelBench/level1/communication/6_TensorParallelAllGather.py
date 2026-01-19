import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn
import torch.distributed as dist

class Model(nn.Module):
    """
    Tensor Parallel AllGather (Distributed)

    Used by: vLLM, TensorRT-LLM, Megatron-LM (tensor parallelism)

    AllGather operation specifically for tensor parallelism.
    Gathers partial results from column-parallel linear layers.
    Each rank contributes output_size // world_size columns.

    Shapes:
        Input: (batch_size, seq_length, output_size_per_rank)
        Output: (batch_size, seq_length, output_size)
    """

    def __init__(self, gather_dim: int = -1, process_group=None, async_op: bool = False):
        """
        Initialize Tensor Parallel AllGather.

        Args:
            gather_dim: Dimension to gather along (default: last dim for TP)
            process_group: The tensor parallel process group
            async_op: Whether to perform asynchronous operation
        """
        super(Model, self).__init__()
        self.gather_dim = gather_dim
        self.process_group = process_group
        self.async_op = async_op

    def forward(self, local_output: torch.Tensor) -> torch.Tensor:
        """
        Perform tensor parallel all-gather.

        Each rank has partial output (output_size // world_size columns).
        AllGather combines all partial outputs into full output.

        Args:
            local_output: Local partial output (batch, seq, output_size_per_rank)

        Returns:
            Full gathered output (batch, seq, total_output_size)
        """
        world_size = dist.get_world_size(self.process_group)

        if world_size == 1:
            return local_output

        # Create list for gathered tensors
        gathered_tensors = [torch.empty_like(local_output) for _ in range(world_size)]

        # Perform all-gather
        handle = dist.all_gather(gathered_tensors, local_output,
                                  group=self.process_group, async_op=self.async_op)

        if self.async_op and handle is not None:
            handle.wait()

        # Concatenate along gather dimension
        output = torch.cat(gathered_tensors, dim=self.gather_dim)

        return output


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "hidden_size": 4096},
    # Llama-3.1-8B: vLLM tensor parallelism
    {"batch_size": 8, "seq_length": 4096, "hidden_size": 4096},
    # Llama-3.1-70B: Megatron-LM tensor parallelism
    {"batch_size": 4, "seq_length": 4096, "hidden_size": 8192},
    # Mistral-7B: TensorRT-LLM tensor parallelism
    {"batch_size": 16, "seq_length": 2048, "hidden_size": 4096},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("communication", "6_TensorParallelAllGather")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    local_output = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_length"], output_size_per_rank), dtype=dtype, device=device)
    return [local_output]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return [-1]
