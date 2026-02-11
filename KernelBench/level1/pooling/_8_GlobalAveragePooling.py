import os
import sys
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Global Average Pooling
    
    Used by: ResNet, EfficientNet, MobileNet, most CNN classifiers
    
    Computes the global average over spatial dimensions.
    Reduces (B, C, H, W) to (B, C) by averaging over H and W.
    
    Shapes:
        Input: (batch_size, channels, height, width)
        Output: (batch_size, channels)
    """
    
    def __init__(self):
        """Initialize Global Average Pooling."""
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply global average pooling.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            Pooled tensor (batch_size, channels)
        """
        return x.mean(dim=[-2, -1])
