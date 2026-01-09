import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Tensor Parallel AllGather
    
    Used by: vLLM, TensorRT-LLM, Megatron-LM (tensor parallelism)
    
    AllGather operation specifically for tensor parallelism.
    Gathers partial results from column-parallel linear layers.
    Each rank contributes output_size // world_size columns.
    
    Shapes:
        Input: (batch_size, seq_length, output_size_per_rank)
        Output: (batch_size, seq_length, output_size)
    """
    
    def __init__(self, output_size_per_rank: int = 4096, world_size: int = 8):
        """
        Initialize Tensor Parallel AllGather.
        
        Args:
            output_size_per_rank: Output dimension per tensor parallel rank
            world_size: Number of tensor parallel ranks
        """
        super(Model, self).__init__()
        self.output_size_per_rank = output_size_per_rank
        self.world_size = world_size
        self.total_output_size = output_size_per_rank * world_size
    
    def forward(self, local_output: torch.Tensor) -> torch.Tensor:
        """
        Simulate tensor parallel all-gather.
        
        In actual distributed setting:
        - Each rank has partial output (output_size // world_size columns)
        - AllGather combines all partial outputs
        
        Args:
            local_output: Local partial output (batch, seq, output_size_per_rank)
            
        Returns:
            Full gathered output (batch, seq, total_output_size)
        """
        batch_size, seq_length, _ = local_output.shape
        
        # Simulate gathering from all ranks
        # In practice: dist.all_gather(tensor_list, local_output)
        # Here we simulate by replicating (represents memory bandwidth pattern)
        gathered = local_output.repeat(1, 1, self.world_size)
        
        return gathered


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 2048
output_size_per_rank = 512  # 4096 / 8 ranks
world_size = 8

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    local_output = torch.randn(batch_size, seq_length, output_size_per_rank, device='cuda')
    return [local_output]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [output_size_per_rank, world_size]

