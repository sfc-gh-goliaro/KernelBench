import os
import sys
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
