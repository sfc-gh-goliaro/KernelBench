import os
import sys
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
