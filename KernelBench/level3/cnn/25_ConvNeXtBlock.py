import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    """LayerNorm for 2D inputs (channels-first)."""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> normalize over C
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x


class Model(nn.Module):
    """
    ConvNeXt Block
    
    The core repeated block in ConvNeXt architecture.
    Used by: ConvNeXt-T/S/B/L/XL
    
    Architecture:
        x -> DepthwiseConv 7x7 -> LayerNorm -> Conv 1x1 -> GELU 
          -> Conv 1x1 -> * LayerScale -> + residual
    
    Key features:
    - Large 7x7 depthwise convolution
    - LayerNorm instead of BatchNorm
    - GELU activation
    - LayerScale for training stability
    - Inverted bottleneck (expand -> contract)
    """
    def __init__(self, dim: int, expansion_ratio: int = 4, layer_scale_init: float = 1e-6,
                 drop_path: float = 0.0):
        super().__init__()
        
        # Depthwise 7x7 conv
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        
        # LayerNorm (channel-wise)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        # Pointwise convs (inverted bottleneck)
        hidden_dim = dim * expansion_ratio
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.pwconv2 = nn.Linear(hidden_dim, dim)
        
        # LayerScale
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None
        
        # Stochastic depth
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        
        # Depthwise conv
        x = self.dwconv(x)
        
        # Permute to (B, H, W, C) for LayerNorm and Linear
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        
        # Inverted bottleneck MLP
        x = self.pwconv1(x)
        x = F.gelu(x)
        x = self.pwconv2(x)
        
        # LayerScale
        if self.gamma is not None:
            x = self.gamma * x
        
        # Permute back to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        # Residual with optional drop path
        x = shortcut + self.drop_path(x)
        
        return x


class StochasticDepth(nn.Module):
    """Stochastic depth (drop path) regularization."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


# Benchmark configuration (ConvNeXt-B dimensions)
batch_size = 8
dim = 128  # Stage 1 dimension
height = 56
width = 56

def get_inputs():
    return [torch.randn(batch_size, dim, height, width)]

def get_init_inputs():
    return [dim]

