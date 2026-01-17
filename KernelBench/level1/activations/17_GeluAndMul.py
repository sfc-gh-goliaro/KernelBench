import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    GELU and Multiply (GeGLU Activation)
    
    Used by: Gemma, some T5 variants, PaLM
    
    Gated activation: GELU(x1) * x2 where x1 and x2 are split from input.
    This is the core activation pattern in GeGLU FFN blocks.
    
    Shapes:
        Input: (batch_size, seq_length, 2 * intermediate_size)
        Output: (batch_size, seq_length, intermediate_size)
    """
    
    def __init__(self, approximate: str = 'none'):
        """
        Initialize GELU and Multiply.
        
        Args:
            approximate: GELU approximation ('none' for exact, 'tanh' for faster)
        """
        super(Model, self).__init__()
        self.approximate = approximate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GeGLU activation.
        
        Args:
            x: Input tensor (batch_size, seq_length, 2 * intermediate_size)
               Contains concatenated gate and up projections
            
        Returns:
            Activated tensor (batch_size, seq_length, intermediate_size)
        """
        # Split into gate and value
        gate, value = x.chunk(2, dim=-1)
        
        # GELU(gate) * value
        return F.gelu(gate, approximate=self.approximate) * value


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 2048
intermediate_size = 14336  # Gemma-style intermediate size

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    # Input is concatenated gate and up projections
    x = torch.randn(batch_size, seq_length, 2 * intermediate_size, device='cuda')
    return [x]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return ['none']

