import os
import sys
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Squeeze-and-Excitation Block
    
    Used by: MobileNetV3, EfficientNet, MnasNet
    
    SE block: global pool → FC → activation → FC → sigmoid → scale.
    Channel attention mechanism.
    
    Shapes:
        Input: (batch, channels, height, width)
        Output: (batch, channels, height, width)
    """
    
    def __init__(self, channels: int, reduction: int = 4):
        """
        Initialize SE block.
        
        Args:
            channels: Number of input channels
            reduction: Reduction ratio for squeeze
        """
        super(Model, self).__init__()
        self.channels = channels
        squeezed_channels = max(1, channels // reduction)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, squeezed_channels)
        self.fc2 = nn.Linear(squeezed_channels, channels)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply squeeze-and-excitation.
        
        Args:
            x: Input tensor (batch, channels, height, width)
            
        Returns:
            Channel-reweighted tensor (batch, channels, height, width)
        """
        batch_size = x.shape[0]
        
        # Squeeze: global average pooling
        se = self.avg_pool(x).view(batch_size, -1)
        
        # Excitation: FC → ReLU → FC → Sigmoid
        se = self.fc1(se)
        se = self.activation(se)
        se = self.fc2(se)
        se = torch.sigmoid(se)
        
        # Scale: broadcast and multiply
        se = se.view(batch_size, self.channels, 1, 1)
        return x * se
