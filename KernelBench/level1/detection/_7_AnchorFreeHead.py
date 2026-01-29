import os
import sys
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Anchor-Free Detection Head
    
    Used by: YOLOv8, FCOS
    
    Anchor-free head with center-based predictions.
    
    Shapes:
        Input: (batch, in_channels, height, width)
        Output: cls (batch, num_classes, H, W), box (batch, 4, H, W)
    """
    
    def __init__(self, in_channels: int, num_classes: int, reg_max: int = 16):
        """
        Initialize anchor-free head.
        
        Args:
            in_channels: Input channels
            num_classes: Number of object classes
            reg_max: Max value for DFL regression
        """
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        
        # Classification branch
        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, 1)
        )
        
        # Regression branch (DFL)
        self.reg_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, 4 * reg_max, 1)
        )
    
    def forward(self, x: torch.Tensor) -> tuple:
        cls_out = self.cls_conv(x)
        reg_out = self.reg_conv(x)
        
        return cls_out, reg_out


# ============================================================================
# Benchmark Configuration
# ============================================================================
