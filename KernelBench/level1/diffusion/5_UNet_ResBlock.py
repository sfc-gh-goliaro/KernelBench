import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

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


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "in_channels": 320, "out_channels": 320, "height": 64, "width": 64, "time_dim": 1280},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("diffusion", "5_UNet_ResBlock")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x_shape = (p["batch_size"], p["in_channels"], p["height"], p["width"])
    time_shape = (p["batch_size"], p["time_dim"])
    x = DISTRIBUTIONS[dist_name](x_shape, dtype=dtype, device=device)
    time_emb = DISTRIBUTIONS[dist_name](time_shape, dtype=dtype, device=device)
    return [x, time_emb]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["in_channels"], p["out_channels"], p["time_dim"]]
