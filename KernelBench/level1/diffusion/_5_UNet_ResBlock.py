import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    UNet Residual Block
    
    Used by: Stable Diffusion 1.5, SDXL
    
    UNet ResBlock: GroupNorm + SiLU + Conv + (time_emb projection) +
    GroupNorm + SiLU + Conv + skip connection.
    
    Shapes:
        x: (batch, in_channels, height, width)
        time_emb: (batch, time_dim)
        Output: (batch, out_channels, height, width)
    """
    
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, 
                 groups: int = 32):
        """
        Initialize UNet ResBlock.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            time_dim: Timestep embedding dimension
            groups: Groups for GroupNorm
        """
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # First conv block
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        # Time embedding projection
        self.time_proj = nn.Linear(time_dim, out_channels)
        
        # Second conv block
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        ResBlock forward pass.
        
        Args:
            x: Input (batch, in_channels, height, width)
            time_emb: Timestep embedding (batch, time_dim)
            
        Returns:
            Output (batch, out_channels, height, width)
        """
        residual = self.skip(x)
        
        # First conv block
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        time_emb = self.time_proj(F.silu(time_emb))
        h = h + time_emb[:, :, None, None]
        
        # Second conv block
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        return h + residual
