import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Stable Diffusion UNet ResBlock
    
    The core repeated residual block in UNet-based diffusion models.
    Used by: Stable Diffusion 1.5, 2.x, SDXL (UNet backbone)
    
    Architecture:
        x -> GroupNorm -> SiLU -> Conv3x3 
          -> + timestep_emb (projected)
          -> GroupNorm -> SiLU -> Dropout -> Conv3x3
          -> + skip (with optional conv for channel mismatch)
    
    Key features:
    - GroupNorm instead of BatchNorm
    - SiLU activation
    - Timestep embedding addition
    - Optional channel projection for skip connection
    """
    def __init__(self, in_channels: int, out_channels: int, temb_channels: int, 
                 groups: int = 32, dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # First conv block
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Timestep embedding projection
        self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        
        # Second conv block
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Skip connection (identity or projection)
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, in_channels, height, width)
            temb: Timestep embedding (batch, temb_channels)
        """
        # Store for skip connection
        skip = x
        
        # First block
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        
        # Add timestep embedding
        temb_proj = self.time_emb_proj(F.silu(temb))
        x = x + temb_proj[:, :, None, None]
        
        # Second block
        x = self.norm2(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        
        # Skip connection
        return x + self.skip_conv(skip)


# Benchmark configuration (SDXL UNet dimensions)
batch_size = 2
in_channels = 320
out_channels = 320
temb_channels = 1280
height = 64
width = 64

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    temb = torch.randn(batch_size, temb_channels)
    return [x, temb]

def get_init_inputs():
    return [in_channels, out_channels, temb_channels]

