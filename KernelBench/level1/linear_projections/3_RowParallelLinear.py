import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Row Parallel Linear Layer
    
    Used by: Megatron-LM, vLLM, TensorRT-LLM (tensor parallelism)
    
    Linear layer that partitions weights along the input dimension.
    Each GPU holds a slice of the input features. Requires reduce-scatter
    or all-reduce after computation to combine partial results.
    Used for MLP's second projection (down) in tensor parallel training.
    
    Shapes:
        Input: (batch_size, seq_length, input_size_per_partition)
        Output: (batch_size, seq_length, output_size)
    """
    
    def __init__(self, input_size: int = 14336, output_size: int = 4096,
                 num_partitions: int = 1, partition_idx: int = 0):
        """
        Initialize Row Parallel Linear.
        
        Args:
            input_size: Total input dimension (before partitioning)
            output_size: Output hidden dimension
            num_partitions: Number of tensor parallel partitions
            partition_idx: This partition's index
        """
        super(Model, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_partitions = num_partitions
        self.partition_idx = partition_idx
        
        # Each partition gets input_size // num_partitions rows
        assert input_size % num_partitions == 0
        self.input_size_per_partition = input_size // num_partitions
        
        # Local weight matrix (partial input, full output)
        self.weight = nn.Parameter(
            torch.randn(output_size, self.input_size_per_partition) * 0.02
        )
        self.bias = nn.Parameter(torch.zeros(output_size)) if num_partitions == 1 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Row parallel linear transformation.
        
        Args:
            x: Input tensor (batch_size, seq_length, input_size_per_partition)
            
        Returns:
            Output tensor (batch_size, seq_length, output_size)
            Note: In distributed setting, this is a partial result that needs reduction
        """
        return torch.nn.functional.linear(x, self.weight, self.bias)


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 2048
input_size = 14336  # Llama-3 FFN intermediate size
output_size = 4096
num_partitions = 1
partition_idx = 0

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    input_size_per_partition = input_size // num_partitions
    x = torch.randn(batch_size, seq_length, input_size_per_partition, device='cuda')
    return [x]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [input_size, output_size, num_partitions, partition_idx]

