import torch
import torch.nn as nn

class Model(nn.Module):
    """
    AllReduce (Simulated)
    
    Used by: Tensor parallel (post-GEMM reduction)
    
    Sum tensor values across all distributed ranks, result replicated
    on all ranks. This is a simulated single-GPU version.
    
    Shapes:
        Input: (batch, seq_len, hidden_size)
        Output: (batch, seq_len, hidden_size) - sum across "ranks"
    """
    
    def __init__(self, num_ranks: int = 8):
        """
        Initialize simulated AllReduce.
        
        Args:
            num_ranks: Number of simulated ranks
        """
        super(Model, self).__init__()
        self.num_ranks = num_ranks
    
    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        """
        Simulate AllReduce by summing input tensors.
        
        Args:
            tensors: Variable number of tensors to reduce (simulating different ranks)
            
        Returns:
            Sum of all input tensors
        """
        if len(tensors) == 1:
            # Single tensor: simulate by splitting and reducing
            x = tensors[0]
            # In real TP, this would sum partial results from different ranks
            return x
        else:
            # Multiple tensors: sum them (simulating reduction)
            result = tensors[0]
            for t in tensors[1:]:
                result = result + t
            return result


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 2048
hidden_size = 4096

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    # Simulate 2 rank partial results
    x1 = torch.randn(batch_size, seq_length, hidden_size, device='cuda')
    x2 = torch.randn(batch_size, seq_length, hidden_size, device='cuda')
    return [x1, x2]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [8]

