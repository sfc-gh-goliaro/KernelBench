import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Hard-Swish Activation
    
    Used by: MobileNetV3, MnasNet
    
    Hard-Swish: x * ReLU6(x+3)/6, efficient approximation of Swish.
    More computationally efficient than regular Swish/SiLU.
    
    Shapes:
        Input: any shape
        Output: same shape as input
    """
    
    def __init__(self):
        """Initialize Hard-Swish."""
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Hard-Swish activation.
        
        Args:
            x: Input tensor of any shape
            
        Returns:
            Activated tensor of same shape
        """
        return x * F.relu6(x + 3) / 6


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 32
channels = 960
height = 7
width = 7

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    x = torch.randn(batch_size, channels, height, width, device='cuda')
    return [x]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return []

