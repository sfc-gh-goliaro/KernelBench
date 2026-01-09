import torch
import torch.nn as nn
import torch.nn.functional as F

class HardSwish(nn.Module):
    """HardSwish activation: x * ReLU6(x+3) / 6"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.relu6(x + 3) / 6


class HardSigmoid(nn.Module):
    """HardSigmoid activation: ReLU6(x+3) / 6"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(x + 3) / 6


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block with HardSigmoid."""
    def __init__(self, in_channels: int, squeeze_channels: int):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(squeeze_channels, in_channels, kernel_size=1)
        self.activation = HardSigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.avgpool(x)
        scale = F.relu(self.fc1(scale))
        scale = self.activation(self.fc2(scale))
        return x * scale


class Model(nn.Module):
    """
    MobileNetV3 Inverted Residual Block
    
    The core repeated block in MobileNetV3 architecture.
    Used by: MobileNetV3-Small, MobileNetV3-Large
    
    Architecture:
        x -> Conv 1x1 (expand) -> BN -> Activation
          -> DepthwiseConv -> BN -> Activation
          -> [SE Block (optional)]
          -> Conv 1x1 (project) -> BN
          -> + residual (if stride=1 and in_channels=out_channels)
    
    Key features:
    - HardSwish activation (efficient)
    - Squeeze-and-Excitation with HardSigmoid
    - Configurable expansion ratio
    - Optional SE block
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, expansion_ratio: float = 4.0, use_se: bool = True,
                 use_hs: bool = True, se_ratio: float = 0.25):
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        
        expanded_channels = int(in_channels * expansion_ratio)
        self.use_expand = expansion_ratio != 1
        
        # Activation
        activation = HardSwish() if use_hs else nn.ReLU(inplace=True)
        
        layers = []
        
        # Expand
        if self.use_expand:
            layers.extend([
                nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                activation,
            ])
        
        # Depthwise
        padding = (kernel_size - 1) // 2
        layers.extend([
            nn.Conv2d(expanded_channels if self.use_expand else in_channels, 
                     expanded_channels if self.use_expand else in_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     groups=expanded_channels if self.use_expand else in_channels, bias=False),
            nn.BatchNorm2d(expanded_channels if self.use_expand else in_channels),
            activation,
        ])
        
        self.conv = nn.Sequential(*layers)
        
        # Squeeze-and-Excitation
        se_channels = int(in_channels * se_ratio)
        self.se = SqueezeExcitation(
            expanded_channels if self.use_expand else in_channels, se_channels
        ) if use_se else nn.Identity()
        
        # Project
        project_channels = expanded_channels if self.use_expand else in_channels
        self.project = nn.Sequential(
            nn.Conv2d(project_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        
        x = self.conv(x)
        x = self.se(x)
        x = self.project(x)
        
        if self.use_residual:
            x = x + shortcut
        
        return x


# Benchmark configuration
batch_size = 32
in_channels = 40
out_channels = 40
height = 28
width = 28

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels]

