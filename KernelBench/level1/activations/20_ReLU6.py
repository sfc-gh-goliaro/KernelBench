import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    ReLU6 Activation
    
    Used by: MobileNet, EfficientNet-Lite, HardSwish/HardSigmoid implementations
    
    Clipped ReLU: min(max(0, x), 6)
    Prevents activation explosion and is used in mobile-efficient networks.
    
    Shapes:
        Input: any shape
        Output: same shape as input
    """
    
    def __init__(self):
        """Initialize ReLU6."""
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply ReLU6 activation.
        
        Args:
            x: Input tensor of any shape
            
        Returns:
            Activated tensor with values clamped to [0, 6]
        """
        return F.relu6(x)


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 32
channels = 576
height = 14
width = 14

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    x = torch.randn(batch_size, channels, height, width, device='cuda')
    return [x]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return []

