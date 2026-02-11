import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Path Aggregation Network (PAN) Upsample
    
    Used by: YOLOv5, YOLOv8
    
    PAN upsample + concat for bottom-up path aggregation.
    
    Shapes:
        high_level: (batch, channels, H, W)
        low_level: (batch, channels, 2H, 2W)
        Output: (batch, channels, 2H, 2W)
    """
    
    def __init__(self, channels: int):
        """
        Initialize PAN upsample.
        
        Args:
            channels: Number of channels
        """
        super(Model, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, high_level: torch.Tensor, low_level: torch.Tensor) -> torch.Tensor:
        high = self.upsample(high_level)
        return self.conv(torch.cat([high, low_level], dim=1))
