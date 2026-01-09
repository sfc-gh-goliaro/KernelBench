import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Merged Column Parallel Linear (Fused Gate/Up Projection)
    
    Used by: vLLM, TensorRT-LLM, Megatron-LM (SwiGLU MLP)
    
    Fused projection for gate and up projections in a single GEMM.
    More efficient than separate projections due to better memory access.
    Used in FFN: [gate, up] = MergedColumnParallel(x), then SiLU(gate) * up
    
    Shapes:
        Input: (batch_size, seq_length, hidden_size)
        Output: (batch_size, seq_length, 2 * intermediate_size)
    """
    
    def __init__(self, hidden_size: int = 4096, intermediate_size: int = 14336):
        """
        Initialize Merged Column Parallel Linear.
        
        Args:
            hidden_size: Input hidden dimension
            intermediate_size: FFN intermediate dimension (per output)
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Merged gate and up projections (2x intermediate_size output)
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute merged gate and up projections.
        
        Args:
            x: Input tensor (batch_size, seq_length, hidden_size)
            
        Returns:
            Merged output (batch_size, seq_length, 2 * intermediate_size)
            First half is gate projection, second half is up projection
        """
        return self.gate_up_proj(x)


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 2048
hidden_size = 4096
intermediate_size = 14336  # Llama-3 8B

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    x = torch.randn(batch_size, seq_length, hidden_size, device='cuda')
    return [x]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [hidden_size, intermediate_size]

