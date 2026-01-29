import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Feature Pyramid Network (FPN) Concatenation
    
    Used by: YOLO, RetinaNet, Faster-RCNN
    
    FPN concatenation for multi-scale feature fusion.
    Upsamples high-level features and concatenates with low-level.
    
    Shapes:
        high_level: (batch, high_channels, H, W)
        low_level: (batch, low_channels, 2H, 2W)
        Output: (batch, out_channels, 2H, 2W)
    """
    
    def __init__(self, high_channels: int, low_channels: int, out_channels: int):
        """
        Initialize FPN concat.
        
        Args:
            high_channels: Channels in high-level (smaller) features
            low_channels: Channels in low-level (larger) features
            out_channels: Output channels
        """
        super(Model, self).__init__()
        
        # Reduce high-level channels
        self.reduce_high = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        # Reduce low-level channels
        self.reduce_low = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        # Fuse concatenated features
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, high_level: torch.Tensor, low_level: torch.Tensor) -> torch.Tensor:
        # Reduce channels
        high = self.reduce_high(high_level)
        low = self.reduce_low(low_level)
        
        # Upsample high-level to match low-level size
        high = F.interpolate(high, size=low.shape[2:], mode='nearest')
        
        # Concatenate and fuse
        return self.fuse(torch.cat([high, low], dim=1))


# ============================================================================
# Benchmark Configuration
# ============================================================================
