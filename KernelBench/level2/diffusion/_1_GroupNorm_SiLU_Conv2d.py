import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Fused GroupNorm + SiLU + Conv2d
    
    Used by: Stable Diffusion 1.5, SDXL
    
    UNet ResBlock pattern: GroupNorm + SiLU + Conv2d.
    """
    
    def __init__(self, in_channels: int, out_channels: int, groups: int = 32):
        super(Model, self).__init__()
        self.norm = nn.GroupNorm(groups, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.silu(self.norm(x)))


batch_size, channels, height, width = 8, 320, 64, 64

def get_inputs():
    return [torch.randn(batch_size, channels, height, width, device='cuda')]

def get_init_inputs():
    return [channels, channels]

