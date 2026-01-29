import os
import sys
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Adaptive Average Pooling 2D
    
    Used by: ResNet, EfficientNet, ViT (before classification head)
    
    Pools spatial dimensions to a fixed output size regardless of input size.
    Commonly used to reduce feature maps to 1x1 before the classification head.
    
    Shapes:
        Input: (batch_size, channels, height, width)
        Output: (batch_size, channels, output_height, output_width)
    """
    
    def __init__(self, output_size: tuple = (1, 1)):
        """
        Initialize Adaptive Average Pool 2D.
        
        Args:
            output_size: Target output size (height, width)
        """
        super(Model, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive average pooling.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            Pooled tensor (batch_size, channels, output_height, output_width)
        """
        return self.pool(x)


# ============================================================================
# Benchmark Configuration
# ============================================================================
