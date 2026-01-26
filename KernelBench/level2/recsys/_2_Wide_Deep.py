import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """Fused Wide + Deep (Wide & Deep Learning)."""
    
    def __init__(self, wide_dim: int, deep_dim: int, hidden: int = 256):
        super(Model, self).__init__()
        self.wide = nn.Linear(wide_dim, 1)
        self.deep = nn.Sequential(nn.Linear(deep_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    
    def forward(self, wide_x, deep_x):
        return torch.sigmoid(self.wide(wide_x) + self.deep(deep_x))

batch_size, wide_dim, deep_dim = 4096, 1000, 416
def get_inputs(): return [torch.randn(batch_size, wide_dim, device='cuda'), torch.randn(batch_size, deep_dim, device='cuda')]
def get_init_inputs(): return [wide_dim, deep_dim]

