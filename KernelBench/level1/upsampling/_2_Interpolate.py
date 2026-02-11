import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Interpolation Upsampling
    
    Used by: UNet, FPN, diffusion models, segmentation networks
    
    Upsamples spatial dimensions using various interpolation methods.
    Supports bilinear, bicubic, nearest, and trilinear modes.
    
    Can be configured with either a fixed scale_factor (set at init) or
    a dynamic target size (passed at forward time). When both are None
    at forward time, the init-time scale_factor is used.
    
    Shapes:
        Input: (batch_size, channels, height, width)
        Output: (batch_size, channels, new_height, new_width)
    """
    
    def __init__(self, scale_factor: float = 2.0, mode: str = 'bilinear',
                 align_corners: bool = False):
        """
        Initialize Interpolation.
        
        Args:
            scale_factor: Default upsampling factor (used when no size is given at forward time)
            mode: Interpolation mode ('nearest', 'bilinear', 'bicubic', 'trilinear')
            align_corners: If True, align corners of input and output
        """
        super(Model, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners if mode != 'nearest' else None
    
    def forward(self, x: torch.Tensor, size=None) -> torch.Tensor:
        """
        Apply interpolation upsampling.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            size: Optional target output size (int or tuple). When provided,
                  scale_factor is ignored and the output is resized to this size.
            
        Returns:
            Upsampled tensor (batch_size, channels, new_height, new_width)
        """
        if size is not None:
            return F.interpolate(
                x,
                size=size,
                mode=self.mode,
                align_corners=self.align_corners,
            )
        return F.interpolate(
            x, 
            scale_factor=self.scale_factor, 
            mode=self.mode,
            align_corners=self.align_corners
        )
