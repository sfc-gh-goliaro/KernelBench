import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """Fused CSP Block with Conv+SiLU (YOLO)."""
    
    def __init__(self, channels: int):
        super(Model, self).__init__()
        hid = channels // 2
        self.cv1 = nn.Sequential(nn.Conv2d(channels, hid, 1, bias=False), nn.BatchNorm2d(hid))
        self.cv2 = nn.Sequential(nn.Conv2d(channels, hid, 1, bias=False), nn.BatchNorm2d(hid))
        self.cv3 = nn.Sequential(nn.Conv2d(2*hid, channels, 1, bias=False), nn.BatchNorm2d(channels))
        self.block = nn.Sequential(
            nn.Conv2d(hid, hid, 3, padding=1, bias=False), nn.BatchNorm2d(hid), nn.SiLU(),
            nn.Conv2d(hid, hid, 3, padding=1, bias=False), nn.BatchNorm2d(hid), nn.SiLU())
    
    def forward(self, x):
        return F.silu(self.cv3(torch.cat([self.block(F.silu(self.cv1(x))), F.silu(self.cv2(x))], 1)))

batch_size, channels, h, w = 8, 256, 40, 40
def get_inputs(): return [torch.randn(batch_size, channels, h, w, device='cuda')]
def get_init_inputs(): return [channels]

