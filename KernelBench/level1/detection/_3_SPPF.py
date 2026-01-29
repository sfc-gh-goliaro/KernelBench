import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Spatial Pyramid Pooling Fast (SPPF)
    
    Used by: YOLOv5, YOLOv8
    
    Multi-scale pooling with shared MaxPool for efficiency.
    
    Shapes:
        Input: (batch, in_channels, height, width)
        Output: (batch, out_channels, height, width)
    """
    
    def __init__(self, in_channels: int, out_channels: int, pool_size: int = 5):
        """
        Initialize SPPF.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            pool_size: MaxPool kernel size
        """
        super(Model, self).__init__()
        hidden_channels = in_channels // 2
        
        self.cv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True)
        )
        
        self.cv2 = nn.Sequential(
            nn.Conv2d(hidden_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=1, padding=pool_size // 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        
        # Apply pooling multiple times (equivalent to different scales)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


# ============================================================================
# Benchmark Configuration
# ============================================================================
