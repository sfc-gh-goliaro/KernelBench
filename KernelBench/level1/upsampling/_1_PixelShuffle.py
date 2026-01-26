import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Pixel Shuffle (Sub-pixel Convolution)
    
    Used by: ESPCN, super-resolution models, some diffusion decoders
    
    Rearranges elements from channels to spatial dimensions.
    (B, C*r^2, H, W) -> (B, C, H*r, W*r)
    Efficient upsampling without transposed convolutions.
    
    Shapes:
        Input: (batch_size, channels * upscale_factor^2, height, width)
        Output: (batch_size, channels, height * upscale_factor, width * upscale_factor)
    """
    
    def __init__(self, upscale_factor: int = 2):
        """
        Initialize Pixel Shuffle.
        
        Args:
            upscale_factor: Upsampling factor (r in the paper)
        """
        super(Model, self).__init__()
        self.upscale_factor = upscale_factor
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply pixel shuffle upsampling.
        
        Args:
            x: Input tensor (batch_size, channels * r^2, height, width)
            
        Returns:
            Upsampled tensor (batch_size, channels, height * r, width * r)
        """
        return self.pixel_shuffle(x)


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 16, "channels": 64, "upscale_factor": 2, "height": 64, "width": 64},
    # ESPCN: super-resolution 4x upscaling
    {"batch_size": 8, "channels": 48, "upscale_factor": 4, "height": 128, "width": 128},
    # Real-ESRGAN: high-quality image restoration
    {"batch_size": 4, "channels": 64, "upscale_factor": 2, "height": 256, "width": 256},
    # SDXL VAE decoder: latent to pixel space
    {"batch_size": 2, "channels": 128, "upscale_factor": 2, "height": 64, "width": 64},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("upsampling", "1_PixelShuffle")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], in_channels, p["height"], p["width"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["upscale_factor"]]
