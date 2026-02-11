import os
import sys
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Detection Head (Anchor-based)
    
    Used by: YOLOv5
    
    Detection head: conv layers → cls/box predictions per anchor.
    
    Shapes:
        Input: (batch, in_channels, height, width)
        Output: (batch, num_anchors, height, width, num_classes + 5)
    """
    
    def __init__(self, in_channels: int, num_classes: int, num_anchors: int = 3):
        """
        Initialize detection head.
        
        Args:
            in_channels: Input channels
            num_classes: Number of object classes
            num_anchors: Number of anchors per location
        """
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.num_outputs = num_classes + 5  # cls + (x, y, w, h, obj)
        
        self.conv = nn.Conv2d(in_channels, num_anchors * self.num_outputs, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        
        x = self.conv(x)
        x = x.view(batch_size, self.num_anchors, self.num_outputs, height, width)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        
        return x
