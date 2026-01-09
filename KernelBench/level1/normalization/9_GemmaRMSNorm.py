import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Gemma RMS Normalization
    
    Used by: Gemma, Gemma-2, Gemma-3
    
    Variant of RMSNorm with (1 + weight) scaling instead of just weight.
    This allows weights to be initialized to zero while still providing
    identity-like behavior initially.
    
    Shapes:
        Input: (batch_size, seq_length, hidden_size)
        Output: (batch_size, seq_length, hidden_size)
    """
    
    def __init__(self, hidden_size: int = 3584, eps: float = 1e-6):
        """
        Initialize Gemma RMS Norm.
        
        Args:
            hidden_size: Hidden dimension to normalize
            eps: Small constant for numerical stability
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Gemma RMS normalization.
        
        Args:
            x: Input tensor (batch_size, seq_length, hidden_size)
            
        Returns:
            Normalized tensor (batch_size, seq_length, hidden_size)
        """
        # Compute RMS
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        
        # Apply (1 + weight) scaling
        return x_normed * (1 + self.weight)


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 2048
hidden_size = 3584  # Gemma-2-9B hidden size

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    x = torch.randn(batch_size, seq_length, hidden_size, device='cuda')
    return [x]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [hidden_size]

