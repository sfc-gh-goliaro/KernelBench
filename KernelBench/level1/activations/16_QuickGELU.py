import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Quick GELU
    
    Used by: CLIP, OpenCLIP
    
    Approximate GELU using sigmoid: x * sigmoid(1.702 * x).
    Faster than exact GELU while maintaining similar properties.
    
    Shapes:
        Input: any shape
        Output: same shape as input
    """
    
    def __init__(self):
        """Initialize Quick GELU."""
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Quick GELU activation.
        
        Args:
            x: Input tensor of any shape
            
        Returns:
            Activated tensor of same shape
        """
        return x * torch.sigmoid(1.702 * x)


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 32
seq_length = 197  # ViT: 196 patches + 1 CLS
hidden_size = 768

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    x = torch.randn(batch_size, seq_length, hidden_size, device='cuda')
    return [x]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return []

