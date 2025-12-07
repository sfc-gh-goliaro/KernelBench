import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    All-Gather Collective Communication.
    
    Gathers tensors from all ranks and concatenates them.
    Used in tensor parallelism to gather partial outputs,
    and in sequence parallelism to gather sequence chunks.
    
    Simulates the communication pattern for benchmarking purposes.
    """
    def __init__(self, world_size, gather_dim=0):
        """
        :param world_size: Number of parallel ranks
        :param gather_dim: Dimension along which to gather
        """
        super(Model, self).__init__()
        self.world_size = world_size
        self.gather_dim = gather_dim
    
    def forward(self, tensors):
        """
        Simulate all-gather across ranks.
        
        :param tensors: List of tensors from each rank
        :return: Gathered tensor available on all ranks
        """
        assert len(tensors) == self.world_size, \
            f"Expected {self.world_size} tensors, got {len(tensors)}"
        
        # Concatenate along gather dimension
        gathered = torch.cat(tensors, dim=self.gather_dim)
        
        # Each rank gets the full gathered tensor
        return [gathered.clone() for _ in range(self.world_size)]


# All-Gather with variable sizes (for dynamic batching)
class AllGatherV(nn.Module):
    """
    All-Gather-V: Variable-size all-gather.
    
    Supports gathering tensors of different sizes from each rank.
    Used for dynamic batching scenarios.
    """
    def __init__(self, world_size, gather_dim=0):
        super(AllGatherV, self).__init__()
        self.world_size = world_size
        self.gather_dim = gather_dim
    
    def forward(self, tensors, sizes=None):
        """
        Variable-size all-gather.
        
        :param tensors: List of tensors (potentially different sizes along gather_dim)
        :param sizes: Optional list of sizes for each tensor
        :return: Gathered tensor and size information
        """
        # Concatenate (sizes can vary along gather_dim)
        gathered = torch.cat(tensors, dim=self.gather_dim)
        
        # Track sizes for later splitting if needed
        if sizes is None:
            sizes = [t.shape[self.gather_dim] for t in tensors]
        
        return [gathered.clone() for _ in range(self.world_size)], sizes


# Test parameters
world_size = 8
batch_per_rank = 32
seq_len = 512
hidden_dim = 1024

def get_inputs():
    # Each rank has a portion of the batch
    tensors = [torch.randn(batch_per_rank, seq_len, hidden_dim) for _ in range(world_size)]
    return [tensors]

def get_init_inputs():
    return [world_size, 0]  # gather along batch dimension

