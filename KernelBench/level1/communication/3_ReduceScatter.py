import torch
import torch.nn as nn

class Model(nn.Module):
    """
    ReduceScatter (Simulated)
    
    Used by: Sequence parallel (gradient sync)
    
    Reduce (sum) then scatter result so each rank gets different shard.
    This is a simulated single-GPU version.
    
    Shapes:
        Input: (batch, seq_len, hidden_size) per rank
        Output: (batch, seq_len, hidden_size // num_ranks) per rank
    """
    
    def __init__(self, num_ranks: int = 8, scatter_dim: int = -1):
        """
        Initialize simulated ReduceScatter.
        
        Args:
            num_ranks: Number of simulated ranks
            scatter_dim: Dimension to scatter along
        """
        super(Model, self).__init__()
        self.num_ranks = num_ranks
        self.scatter_dim = scatter_dim
    
    def forward(self, *tensors: torch.Tensor, rank: int = 0) -> torch.Tensor:
        """
        Simulate ReduceScatter.
        
        Args:
            tensors: Tensors from each rank to reduce
            rank: Which rank's shard to return
            
        Returns:
            This rank's shard of the reduced result
        """
        # First reduce (sum)
        reduced = tensors[0]
        for t in tensors[1:]:
            reduced = reduced + t
        
        # Then scatter (each rank gets a shard)
        shard_size = reduced.shape[self.scatter_dim] // self.num_ranks
        start = rank * shard_size
        end = start + shard_size
        
        if self.scatter_dim == -1 or self.scatter_dim == len(reduced.shape) - 1:
            return reduced[..., start:end]
        else:
            # General case: use narrow
            return reduced.narrow(self.scatter_dim, start, shard_size)


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 2048
hidden_size = 4096
num_ranks = 8

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    tensors = [torch.randn(batch_size, seq_length, hidden_size, device='cuda') 
               for _ in range(num_ranks)]
    return tensors

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [num_ranks, -1]

