import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """Fused Depthwise Conv + BN + HardSwish (MobileNetV3)."""
    
    def __init__(self, channels: int):
        super(Model, self).__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        return self.bn(self.dw(x)) * F.relu6(self.bn(self.dw(x)) + 3) / 6

batch_size, channels, h, w = 32, 96, 56, 56
def get_inputs(): return [torch.randn(batch_size, channels, h, w, device='cuda')]
def get_init_inputs(): return [channels]

