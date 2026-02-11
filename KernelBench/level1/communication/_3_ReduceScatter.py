import os
import sys
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
