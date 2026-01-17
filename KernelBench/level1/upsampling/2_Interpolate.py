import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Interpolation Upsampling
    
    Used by: UNet, FPN, diffusion models, segmentation networks
    
    Upsamples spatial dimensions using various interpolation methods.
    Supports bilinear, bicubic, nearest, and trilinear modes.
    
    Shapes:
        Input: (batch_size, channels, height, width)
        Output: (batch_size, channels, new_height, new_width)
    """
    
    def __init__(self, scale_factor: float = 2.0, mode: str = 'bilinear',
                 align_corners: bool = False):
        """
        Initialize Interpolation.
        
        Args:
            scale_factor: Upsampling factor
            mode: Interpolation mode ('nearest', 'bilinear', 'bicubic', 'trilinear')
            align_corners: If True, align corners of input and output
        """
        super(Model, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners if mode != 'nearest' else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply interpolation upsampling.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            Upsampled tensor (batch_size, channels, new_height, new_width)
        """
        return F.interpolate(
            x, 
            scale_factor=self.scale_factor, 
            mode=self.mode,
            align_corners=self.align_corners
        )


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 16
channels = 512
height = 32
width = 32
scale_factor = 2.0
mode = 'bilinear'
align_corners = False

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    x = torch.randn(batch_size, channels, height, width, device='cuda')
    return [x]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [scale_factor, mode, align_corners]

