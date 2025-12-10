import torch
import torch.nn as nn
import torch.nn.functional as F


TASK_CONFIG = {
    'comparison_mode': 'default',
    'atol': 1e-4,
    'rtol': 1e-4,
}

class Model(nn.Module):
    """
    All-Reduce Collective Communication.
    
    Reduces tensors across all ranks and distributes the result back
    to all ranks. Essential for gradient synchronization in data parallelism
    and activation aggregation in tensor parallelism.
    
    Simulates the communication pattern for benchmarking purposes.
    """
    def __init__(self, world_size, reduce_op='sum'):
        """
        :param world_size: Number of parallel ranks
        :param reduce_op: Reduction operation ('sum', 'mean', 'max', 'min')
        """
        super(Model, self).__init__()
        self.world_size = world_size
        self.reduce_op = reduce_op
    
    def forward(self, tensors):
        """
        Simulate all-reduce across ranks.
        
        :param tensors: List of tensors from each rank, each of shape (...)
        :return: Reduced tensor broadcast to all ranks
        """
        assert len(tensors) == self.world_size, \
            f"Expected {self.world_size} tensors, got {len(tensors)}"
        
        # Stack tensors
        stacked = torch.stack(tensors, dim=0)  # (world_size, ...)
        
        # Apply reduction
        if self.reduce_op == 'sum':
            reduced = stacked.sum(dim=0)
        elif self.reduce_op == 'mean':
            reduced = stacked.mean(dim=0)
        elif self.reduce_op == 'max':
            reduced = stacked.max(dim=0).values
        elif self.reduce_op == 'min':
            reduced = stacked.min(dim=0).values
        else:
            raise ValueError(f"Unknown reduce_op: {self.reduce_op}")
        
        # Broadcast result to all ranks (simulate by returning list)
        return [reduced.clone() for _ in range(self.world_size)]


# Ring All-Reduce implementation (bandwidth-optimal)
class RingAllReduce(nn.Module):
    """
    Ring All-Reduce: Bandwidth-optimal all-reduce implementation.
    
    Splits tensor into chunks and performs reduce-scatter followed
    by all-gather in a ring topology.
    """
    def __init__(self, world_size):
        super(RingAllReduce, self).__init__()
        self.world_size = world_size
    
    def forward(self, tensors):
        """
        Ring all-reduce simulation.
        
        :param tensors: List of tensors from each rank
        :return: List of reduced tensors for each rank
        """
        tensor_size = tensors[0].numel()
        chunk_size = tensor_size // self.world_size
        
        # Phase 1: Reduce-Scatter
        # Each rank ends up with a fully reduced chunk
        chunks = [t.view(self.world_size, -1) for t in tensors]
        
        reduced_chunks = []
        for chunk_idx in range(self.world_size):
            chunk_sum = sum(chunks[rank][chunk_idx] for rank in range(self.world_size))
            reduced_chunks.append(chunk_sum)
        
        # Phase 2: All-Gather
        # Gather all reduced chunks to all ranks
        full_reduced = torch.cat(reduced_chunks, dim=0)
        
        return [full_reduced.clone().view_as(tensors[0]) for _ in range(self.world_size)]


# Test parameters
world_size = 8
tensor_shape = (1024, 1024)

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    # Simulate tensors from different ranks
    tensors = [torch.randn(*tensor_shape) for _ in range(world_size)]
    return [tensors]

def get_init_inputs():
    return [world_size, 'sum']

