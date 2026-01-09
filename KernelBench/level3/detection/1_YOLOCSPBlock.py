import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNSiLU(nn.Module):
    """Standard Conv + BatchNorm + SiLU block."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1,
                 stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck block."""
    def __init__(self, in_channels: int, out_channels: int, shortcut: bool = True, expansion: float = 0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = ConvBNSiLU(in_channels, hidden_channels, kernel_size=1)
        self.cv2 = ConvBNSiLU(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Model(nn.Module):
    """
    YOLO CSP (Cross Stage Partial) Block
    
    The core feature extraction block in YOLOv5.
    Used by: YOLOv5, YOLOv6, YOLOv7 variants
    
    Architecture:
        x -> split into two paths:
            Path 1: Conv 1x1 (transition)
            Path 2: Conv 1x1 -> n * Bottleneck
        -> Concat -> Conv 1x1 (fusion)
    
    Key features:
    - Cross-Stage Partial connections
    - Gradient flow optimization
    - Bottleneck blocks with residuals
    """
    def __init__(self, in_channels: int, out_channels: int, num_bottlenecks: int = 3,
                 shortcut: bool = True, expansion: float = 0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        
        # Split paths
        self.cv1 = ConvBNSiLU(in_channels, hidden_channels, kernel_size=1)
        self.cv2 = ConvBNSiLU(in_channels, hidden_channels, kernel_size=1)
        
        # Bottleneck sequence
        self.bottlenecks = nn.Sequential(*[
            Bottleneck(hidden_channels, hidden_channels, shortcut, expansion=1.0)
            for _ in range(num_bottlenecks)
        ])
        
        # Fusion
        self.cv3 = ConvBNSiLU(hidden_channels * 2, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split paths
        y1 = self.cv1(x)
        y2 = self.bottlenecks(self.cv2(x))
        
        # Concat and fuse
        return self.cv3(torch.cat([y1, y2], dim=1))


# Benchmark configuration
batch_size = 8
in_channels = 256
out_channels = 256
height = 40
width = 40
num_bottlenecks = 3

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, num_bottlenecks]

