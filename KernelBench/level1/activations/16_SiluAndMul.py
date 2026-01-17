import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    SiLU and Multiply (SwiGLU Activation)
    
    Used by: Llama, Mistral, Qwen, most modern LLMs
    
    Gated activation: SiLU(x1) * x2 where x1 and x2 are split from input.
    This is the core activation pattern in SwiGLU FFN blocks.
    SiLU(x) = x * sigmoid(x)
    
    Shapes:
        Input: (batch_size, seq_length, 2 * intermediate_size)
        Output: (batch_size, seq_length, intermediate_size)
    """
    
    def __init__(self):
        """Initialize SiLU and Multiply."""
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU activation.
        
        Args:
            x: Input tensor (batch_size, seq_length, 2 * intermediate_size)
               Contains concatenated gate and up projections
            
        Returns:
            Activated tensor (batch_size, seq_length, intermediate_size)
        """
        # Split into gate and value
        gate, value = x.chunk(2, dim=-1)
        
        # SiLU(gate) * value
        return F.silu(gate) * value


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 2048
intermediate_size = 14336  # Llama-3 8B intermediate size

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    # Input is concatenated gate and up projections
    x = torch.randn(batch_size, seq_length, 2 * intermediate_size, device='cuda')
    return [x]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return []

