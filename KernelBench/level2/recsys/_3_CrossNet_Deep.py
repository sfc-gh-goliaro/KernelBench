import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """Fused Cross Network + Deep (DCN)."""
    
    def __init__(self, input_dim: int, num_cross: int = 3, hidden: int = 256):
        super(Model, self).__init__()
        self.cross_w = nn.ParameterList([nn.Parameter(torch.randn(input_dim, 1) * 0.01) for _ in range(num_cross)])
        self.cross_b = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(num_cross)])
        self.deep = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.out = nn.Linear(input_dim + hidden, 1)
    
    def forward(self, x):
        x0, xl = x, x
        for w, b in zip(self.cross_w, self.cross_b):
            xl = x0 * (xl @ w) + b + xl
        deep_out = self.deep(x)
        return torch.sigmoid(self.out(torch.cat([xl, deep_out], -1)))

batch_size, input_dim = 4096, 416
def get_inputs(): return [torch.randn(batch_size, input_dim, device='cuda')]
def get_init_inputs(): return [input_dim]

