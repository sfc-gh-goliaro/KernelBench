import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """Fused FPN Concat + Conv (YOLO, RetinaNet)."""
    
    def __init__(self, high_ch: int, low_ch: int, out_ch: int):
        super(Model, self).__init__()
        self.reduce_high = nn.Sequential(nn.Conv2d(high_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
        self.reduce_low = nn.Sequential(nn.Conv2d(low_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
        self.fuse = nn.Sequential(nn.Conv2d(2*out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch))
    
    def forward(self, high, low):
        high = F.interpolate(F.silu(self.reduce_high(high)), size=low.shape[2:], mode='nearest')
        return F.silu(self.fuse(torch.cat([high, F.silu(self.reduce_low(low))], 1)))

def get_inputs(): return [torch.randn(8, 512, 20, 20, device='cuda'), torch.randn(8, 256, 40, 40, device='cuda')]
def get_init_inputs(): return [512, 256, 256]

