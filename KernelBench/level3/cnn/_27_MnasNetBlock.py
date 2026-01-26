import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    MnasNet MBConv Block (Mobile Inverted Bottleneck)
    
    The core repeated block in MnasNet architecture.
    Used by: MnasNet-0.5, MnasNet-1.0, MnasNet-1.3
    
    Architecture:
        x -> Conv 1x1 (expand) -> BN -> ReLU
          -> DepthwiseConv -> BN -> ReLU
          -> Conv 1x1 (project) -> BN
          -> + residual (if stride=1 and in_channels=out_channels)
    
    Key features:
    - Inverted bottleneck structure
    - Depthwise separable convolution
    - ReLU activation (not ReLU6)
    - No SE block (unlike EfficientNet)
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, expansion_ratio: int = 6):
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        
        expanded_channels = in_channels * expansion_ratio
        
        # Expand
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU(inplace=True),
        ) if expansion_ratio != 1 else nn.Identity()
        
        # Depthwise
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Sequential(
            nn.Conv2d(expanded_channels if expansion_ratio != 1 else in_channels,
                     expanded_channels if expansion_ratio != 1 else in_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     groups=expanded_channels if expansion_ratio != 1 else in_channels,
                     bias=False),
            nn.BatchNorm2d(expanded_channels if expansion_ratio != 1 else in_channels),
            nn.ReLU(inplace=True),
        )
        
        # Project
        self.project = nn.Sequential(
            nn.Conv2d(expanded_channels if expansion_ratio != 1 else in_channels,
                     out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        self.expansion_ratio = expansion_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        
        if self.expansion_ratio != 1:
            x = self.expand(x)
        x = self.depthwise(x)
        x = self.project(x)
        
        if self.use_residual:
            x = x + shortcut
        
        return x


# Benchmark configuration (MnasNet-1.0)
batch_size = 32
in_channels = 40
out_channels = 40
height = 28
width = 28

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels]

