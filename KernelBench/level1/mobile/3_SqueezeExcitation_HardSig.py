import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Squeeze-and-Excitation Block with Hard-Sigmoid
    
    Used by: MobileNetV3
    
    SE block using Hard-Sigmoid instead of Sigmoid for efficiency.
    
    Shapes:
        Input: (batch, channels, height, width)
        Output: (batch, channels, height, width)
    """
    
    def __init__(self, channels: int, reduction: int = 4):
        """
        Initialize SE block with Hard-Sigmoid.
        
        Args:
            channels: Number of input channels
            reduction: Reduction ratio for squeeze
        """
        super(Model, self).__init__()
        self.channels = channels
        squeezed_channels = max(1, channels // reduction)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, squeezed_channels)
        self.fc2 = nn.Linear(squeezed_channels, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SE with Hard-Sigmoid."""
        batch_size = x.shape[0]
        
        se = self.avg_pool(x).view(batch_size, -1)
        se = F.relu(self.fc1(se))
        se = self.fc2(se)
        # Hard-Sigmoid instead of Sigmoid
        se = F.relu6(se + 3) / 6
        se = se.view(batch_size, self.channels, 1, 1)
        
        return x * se


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 32
channels = 96
height = 56
width = 56

def get_inputs():
    x = torch.randn(batch_size, channels, height, width, device='cuda')
    return [x]

def get_init_inputs():
    return [channels]

