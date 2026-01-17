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

batch_size = 8
seq_length = 2048
hidden_size = 4096

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    world_size = dist.get_world_size() if dist.is_initialized() else 8
    shard_size = hidden_size // world_size
    x = torch.randn(batch_size, seq_length, shard_size, device='cuda')
    return [x]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [-1]
