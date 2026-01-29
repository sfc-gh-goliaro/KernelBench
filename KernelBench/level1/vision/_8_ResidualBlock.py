import os
import sys
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Residual Block (ResNet-style)
    
    Used by: ResNet, ResNeXt, WideResNet, many vision backbones
    
    Basic residual block with skip connection:
    output = F(x) + x (or projection(x) if dimensions change)
    
    Shapes:
        Input: (batch_size, in_channels, height, width)
        Output: (batch_size, out_channels, height', width')
    """
    
    def __init__(self, in_channels: int = 256, out_channels: int = 256,
                 stride: int = 1, downsample: bool = False):
        """
        Initialize Residual Block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Convolution stride (for downsampling)
            downsample: Whether to use projection shortcut
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut (identity or projection)
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply residual block.
        
        Args:
            x: Input tensor (batch_size, in_channels, height, width)
            
        Returns:
            Output tensor (batch_size, out_channels, height', width')
        """
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


# ============================================================================
# Benchmark Configuration
# ============================================================================
