import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Column Parallel Linear Layer
    
    Used by: Megatron-LM, vLLM, TensorRT-LLM (tensor parallelism)
    
    Linear layer that partitions weights along the output dimension.
    Each GPU holds a slice of the output features.
    Used for MLP's first projection (gate/up) in tensor parallel training.
    
    Shapes:
        Input: (batch_size, seq_length, input_size)
        Output: (batch_size, seq_length, output_size_per_partition)
    """
    
    def __init__(self, input_size: int = 4096, output_size: int = 14336,
                 num_partitions: int = 1, partition_idx: int = 0):
        """
        Initialize Column Parallel Linear.
        
        Args:
            input_size: Input hidden dimension
            output_size: Total output dimension (before partitioning)
            num_partitions: Number of tensor parallel partitions
            partition_idx: This partition's index
        """
        super(Model, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_partitions = num_partitions
        self.partition_idx = partition_idx
        
        # Each partition gets output_size // num_partitions columns
        assert output_size % num_partitions == 0
        self.output_size_per_partition = output_size // num_partitions
        
        # Local weight matrix (full input, partial output)
        self.weight = nn.Parameter(
            torch.randn(self.output_size_per_partition, input_size) * 0.02
        )
        self.bias = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Column parallel linear transformation.
        
        Args:
            x: Input tensor (batch_size, seq_length, input_size)
            
        Returns:
            Output tensor (batch_size, seq_length, output_size_per_partition)
        """
        # F.linear: output = input @ weight.T + bias
        return torch.nn.functional.linear(x, self.weight, self.bias)


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 2048
input_size = 4096
output_size = 14336  # Llama-3 FFN intermediate size
num_partitions = 1
partition_idx = 0

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    x = torch.randn(batch_size, seq_length, input_size, device='cuda')
    return [x]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [input_size, output_size, num_partitions, partition_idx]

