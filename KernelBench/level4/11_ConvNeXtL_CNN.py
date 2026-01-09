"""
ConvNeXt-L CNN

A modern CNN implementing ConvNeXt architecture:
- 7x7 depthwise convolutions (large kernels)
- LayerNorm instead of BatchNorm
- GELU activation
- LayerScale for training stability
- Stochastic Depth regularization

Reference: ConvNeXt-L
- Channels: [192, 384, 768, 1536], Layers: [3, 3, 27, 3]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LayerNorm2d(nn.Module):
    """LayerNorm for 2D inputs (channels-first)."""
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class StochasticDepth(nn.Module):
    """Stochastic Depth (Drop Path) for regularization."""
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


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block with depthwise conv, LayerNorm, GELU, and LayerScale."""
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        # 7x7 depthwise conv
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        # LayerNorm (channels-last, then back to channels-first)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # Pointwise FFN
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        # LayerScale
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(dim)
        ) if layer_scale_init_value > 0 else None

        # Stochastic Depth
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # Depthwise conv
        x = self.dwconv(x)

        # Channels-last for LayerNorm and FFN
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)

        # FFN
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # LayerScale
        if self.gamma is not None:
            x = self.gamma * x

        # Back to channels-first
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)

        # Residual with drop path
        x = residual + self.drop_path(x)
        return x


class ConvNeXtStage(nn.Module):
    """ConvNeXt Stage with downsampling and multiple blocks."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        drop_path_rates: list,
        downsample: bool = True,
    ):
        super().__init__()

        # Downsampling layer
        if downsample:
            self.downsample = nn.Sequential(
                LayerNorm2d(in_channels, eps=1e-6),
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
            )
        else:
            self.downsample = nn.Identity()

        # ConvNeXt blocks
        self.blocks = nn.ModuleList([
            ConvNeXtBlock(out_channels, drop_path=drop_path_rates[i])
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x


class Model(nn.Module):
    """ConvNeXt-L CNN."""
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        depths: tuple = (3, 3, 27, 3),
        dims: tuple = (192, 384, 768, 1536),
        drop_path_rate: float = 0.5,
    ):
        super().__init__()

        # Stem: patchify with 4x4 conv
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6),
        )

        # Calculate drop path rates
        total_depth = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        # Build stages
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(len(depths)):
            stage = ConvNeXtStage(
                in_channels=dims[i - 1] if i > 0 else dims[0],
                out_channels=dims[i],
                depth=depths[i],
                drop_path_rates=dp_rates[cur:cur + depths[i]],
                downsample=(i > 0),
            )
            self.stages.append(stage)
            cur += depths[i]

        # Classification head
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        # Global average pooling
        x = x.mean([-2, -1])  # (B, C)
        x = self.norm(x)
        x = self.head(x)
        return x


# Configuration (reduced for benchmarking)
batch_size = 8
img_size = 224
in_channels = 3
num_classes = 1000

# ConvNeXt-Base configuration (smaller for benchmarking)
depths = (3, 3, 9, 3)
dims = (128, 256, 512, 1024)
drop_path_rate = 0.4


def get_inputs():
    return [torch.randn(batch_size, in_channels, img_size, img_size)]


def get_init_inputs():
    return [{
        'in_channels': in_channels,
        'num_classes': num_classes,
        'depths': depths,
        'dims': dims,
        'drop_path_rate': drop_path_rate,
    }]

