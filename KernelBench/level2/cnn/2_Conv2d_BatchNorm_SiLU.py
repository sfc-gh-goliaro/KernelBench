import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Fused Conv2d + BatchNorm + SiLU
    
    Used by: EfficientNet, ConvNeXt
    
    Modern CNN fusion: Conv2d + BatchNorm + SiLU/Swish.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.bn(self.conv(x)))


batch_size, in_channels, out_channels, height, width = 32, 64, 128, 56, 56

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width, device='cuda')]

def get_init_inputs():
    return [in_channels, out_channels]

