import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """Fused Inverted Residual + Squeeze-Excitation (MobileNetV3)."""
    
    def __init__(self, in_ch: int, out_ch: int, expand: int = 6):
        super(Model, self).__init__()
        hid = in_ch * expand
        self.expand = nn.Sequential(nn.Conv2d(in_ch, hid, 1, bias=False), nn.BatchNorm2d(hid), nn.ReLU6())
        self.dw = nn.Sequential(nn.Conv2d(hid, hid, 3, padding=1, groups=hid, bias=False), nn.BatchNorm2d(hid), nn.ReLU6())
        self.se_pool = nn.AdaptiveAvgPool2d(1)
        self.se_fc1, self.se_fc2 = nn.Linear(hid, hid//4), nn.Linear(hid//4, hid)
        self.project = nn.Sequential(nn.Conv2d(hid, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
        self.use_res = in_ch == out_ch
    
    def forward(self, x):
        out = self.dw(self.expand(x))
        b, c = out.shape[:2]
        se = torch.sigmoid(self.se_fc2(F.relu(self.se_fc1(self.se_pool(out).view(b, c))))).view(b, c, 1, 1)
        out = self.project(out * se)
        return out + x if self.use_res else out

batch_size, channels, h, w = 32, 96, 56, 56
def get_inputs(): return [torch.randn(batch_size, channels, h, w, device='cuda')]
def get_init_inputs(): return [channels, channels]

