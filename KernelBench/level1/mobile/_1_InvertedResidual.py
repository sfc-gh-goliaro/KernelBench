import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Inverted Residual Block (MBConv)
    
    Used by: MobileNetV2/V3, EfficientNet, MnasNet
    
    Inverted residual: expand-depthwise-project pattern.
    Expands channels, applies depthwise conv, then projects back.
    
    Shapes:
        Input: (batch, in_channels, height, width)
        Output: (batch, out_channels, height, width)
    """
    
    def __init__(self, in_channels: int, out_channels: int, expand_ratio: int = 6,
                 stride: int = 1, kernel_size: int = 3):
        """
        Initialize inverted residual block.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            expand_ratio: Expansion ratio for hidden channels
            stride: Stride for depthwise conv
            kernel_size: Kernel size for depthwise conv
        """
        super(Model, self).__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        hidden_dim = in_channels * expand_ratio
        
        layers = []
        
        # Expand (if expand_ratio > 1)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise
        padding = (kernel_size - 1) // 2
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding, 
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])
        
        # Project
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, in_channels, height, width)
            
        Returns:
            Output tensor (batch, out_channels, height, width)
        """
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


# ============================================================================
# Benchmark Configuration
# ============================================================================
