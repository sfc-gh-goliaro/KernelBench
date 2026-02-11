import os
import sys
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
