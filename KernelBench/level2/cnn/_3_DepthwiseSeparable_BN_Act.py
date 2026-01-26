import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Fused Depthwise Separable Conv + BN + Activation
    
    Used by: MobileNet, EfficientNet
    
    Depthwise conv + BN + Act + Pointwise conv + BN + Act.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(Model, self).__init__()
        self.dw_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_channels)
        self.pw_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu6(self.dw_bn(self.dw_conv(x)))
        return F.relu6(self.pw_bn(self.pw_conv(x)))


batch_size, in_channels, out_channels, height, width = 32, 64, 128, 56, 56

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width, device='cuda')]

def get_init_inputs():
    return [in_channels, out_channels]

