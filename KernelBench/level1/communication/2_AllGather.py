import torch
import torch.nn as nn

class Model(nn.Module):
    """
    AllGather (Simulated)
    
    Used by: Tensor parallel (weight gathering)
    
    Gather tensor shards from all ranks into full tensor on each rank.
    This is a simulated single-GPU version.
    
    Shapes:
        Input: (batch, seq_len, hidden_size // num_ranks) per rank
        Output: (batch, seq_len, hidden_size) gathered tensor
    """
    
    def __init__(self, num_ranks: int = 8, gather_dim: int = -1):
        """
        Initialize simulated AllGather.
        
        Args:
            num_ranks: Number of simulated ranks
            gather_dim: Dimension to gather along
        """
        super(Model, self).__init__()
        self.num_ranks = num_ranks
        self.gather_dim = gather_dim
    
    def forward(self, *shards: torch.Tensor) -> torch.Tensor:
        """
        Simulate AllGather by concatenating shards.
        
        Args:
            shards: Tensor shards from each rank
            
        Returns:
            Gathered full tensor
        """
        return torch.cat(shards, dim=self.gather_dim)


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 2048
hidden_size = 4096
num_ranks = 8

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    shard_size = hidden_size // num_ranks
    shards = [torch.randn(batch_size, seq_length, shard_size, device='cuda') 
              for _ in range(num_ranks)]
    return shards

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [num_ranks, -1]

