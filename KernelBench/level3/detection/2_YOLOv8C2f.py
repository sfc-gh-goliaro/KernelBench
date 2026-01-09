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
    """Bottleneck with optional shortcut."""
    def __init__(self, in_channels: int, out_channels: int, shortcut: bool = True, expansion: float = 0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = ConvBNSiLU(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.cv2 = ConvBNSiLU(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Model(nn.Module):
    """
    YOLOv8 C2f Block
    
    The core feature extraction block in YOLOv8.
    Used by: YOLOv8 (all sizes: n/s/m/l/x)
    
    Architecture:
        x -> Conv 1x1 -> split into [cv1_out, remaining]
            -> remaining through n bottlenecks (each output concatenated)
        -> Concat all (cv1_out + all bottleneck outputs) -> Conv 1x1
    
    Key features:
    - CSP-style with gradient flow splitting
    - Multiple bottleneck outputs concatenated
    - More efficient than original CSP
    """
    def __init__(self, in_channels: int, out_channels: int, num_bottlenecks: int = 2,
                 shortcut: bool = True, expansion: float = 0.5):
        super().__init__()
        self.hidden_channels = int(out_channels * expansion)
        
        # Initial conv
        self.cv1 = ConvBNSiLU(in_channels, 2 * self.hidden_channels, kernel_size=1)
        
        # Bottleneck sequence (each takes hidden_channels, outputs hidden_channels)
        self.bottlenecks = nn.ModuleList([
            Bottleneck(self.hidden_channels, self.hidden_channels, shortcut, expansion=1.0)
            for _ in range(num_bottlenecks)
        ])
        
        # Final fusion: (1 + num_bottlenecks) * hidden_channels -> out_channels
        self.cv2 = ConvBNSiLU((1 + num_bottlenecks) * self.hidden_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split after first conv
        y = self.cv1(x)
        y = y.chunk(2, dim=1)
        
        # Collect outputs: first half + each bottleneck output
        outputs = [y[0]]
        x_bottleneck = y[1]
        
        for bottleneck in self.bottlenecks:
            x_bottleneck = bottleneck(x_bottleneck)
            outputs.append(x_bottleneck)
        
        # Concat and fuse
        return self.cv2(torch.cat(outputs, dim=1))


# Benchmark configuration (YOLOv8m)
batch_size = 8
in_channels = 256
out_channels = 256
height = 40
width = 40
num_bottlenecks = 2

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, num_bottlenecks]

