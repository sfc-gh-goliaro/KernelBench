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
    Reduce-Scatter Collective Communication.
    
    Reduces tensors across ranks and scatters the result so each rank
    gets a portion. Used in tensor parallelism for efficient gradient
    reduction and in ZeRO optimizer.
    
    Simulates the communication pattern for benchmarking purposes.
    """
    def __init__(self, world_size, scatter_dim=0, reduce_op='sum'):
        """
        :param world_size: Number of parallel ranks
        :param scatter_dim: Dimension along which to scatter
        :param reduce_op: Reduction operation ('sum', 'mean')
        """
        super(Model, self).__init__()
        self.world_size = world_size
        self.scatter_dim = scatter_dim
        self.reduce_op = reduce_op
    
    def forward(self, tensors):
        """
        Simulate reduce-scatter across ranks.
        
        :param tensors: List of tensors from each rank, same shape
        :return: List of scattered chunks, one per rank
        """
        assert len(tensors) == self.world_size, \
            f"Expected {self.world_size} tensors, got {len(tensors)}"
        
        # Stack and reduce
        stacked = torch.stack(tensors, dim=0)  # (world_size, ...)
        
        if self.reduce_op == 'sum':
            reduced = stacked.sum(dim=0)
        elif self.reduce_op == 'mean':
            reduced = stacked.mean(dim=0)
        else:
            raise ValueError(f"Unknown reduce_op: {self.reduce_op}")
        
        # Scatter along specified dimension
        scatter_size = reduced.shape[self.scatter_dim] // self.world_size
        chunks = torch.split(reduced, scatter_size, dim=self.scatter_dim)
        
        return list(chunks)


# Reduce-Scatter with overlap (for pipeline efficiency)
class ReduceScatterOverlap(nn.Module):
    """
    Reduce-Scatter with computation overlap.
    
    Splits the reduce-scatter into chunks that can overlap
    with computation for better efficiency.
    """
    def __init__(self, world_size, num_chunks=4):
        super(ReduceScatterOverlap, self).__init__()
        self.world_size = world_size
        self.num_chunks = num_chunks
    
    def forward(self, tensors):
        """
        Chunked reduce-scatter for overlap with computation.
        
        :param tensors: List of tensors from each rank
        :return: List of scattered results
        """
        tensor_size = tensors[0].numel()
        chunk_size = tensor_size // self.num_chunks
        
        results = [[] for _ in range(self.world_size)]
        
        for chunk_idx in range(self.num_chunks):
            start = chunk_idx * chunk_size
            end = start + chunk_size if chunk_idx < self.num_chunks - 1 else tensor_size
            
            # Extract chunks
            chunks = [t.view(-1)[start:end] for t in tensors]
            
            # Reduce
            reduced = sum(chunks)
            
            # Scatter to appropriate rank
            scatter_size = reduced.numel() // self.world_size
            for rank in range(self.world_size):
                rank_start = rank * scatter_size
                rank_end = rank_start + scatter_size
                results[rank].append(reduced[rank_start:rank_end])
        
        # Concatenate chunks for each rank
        return [torch.cat(r, dim=0) for r in results]


# Test parameters
world_size = 8
batch_size = 256
hidden_dim = 4096

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    # Each rank has the same shape tensor (e.g., gradients)
    tensors = [torch.randn(batch_size, hidden_dim) for _ in range(world_size)]
    return [tensors]

def get_init_inputs():
    return [world_size, 0, 'sum']

