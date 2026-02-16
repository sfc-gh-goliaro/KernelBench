"""
ConvNeXt V2 CNN Model

Implements ConvNeXt V2 architecture aligned with HuggingFace
ConvNextV2ForImageClassification:
- Depthwise convolution + LayerNorm + pointwise MLP per block
- Global Response Normalization (GRN) layer
- Stochastic depth (drop path)
- Patch embedding stem with 4x4 conv
- Downsampling between stages with LayerNorm + 2x2 conv

HuggingFace weight structure (ConvNextV2ForImageClassification):
  convnextv2.embeddings.patch_embeddings.{weight,bias}
  convnextv2.embeddings.layernorm.{weight,bias}
  convnextv2.encoder.stages.{i}.downsampling_layer.0.{weight,bias}  (LayerNorm, i>=1)
  convnextv2.encoder.stages.{i}.downsampling_layer.1.{weight,bias}  (Conv2d, i>=1)
  convnextv2.encoder.stages.{i}.layers.{j}.dwconv.{weight,bias}
  convnextv2.encoder.stages.{i}.layers.{j}.layernorm.{weight,bias}
  convnextv2.encoder.stages.{i}.layers.{j}.pwconv1.{weight,bias}
  convnextv2.encoder.stages.{i}.layers.{j}.grn.{weight,bias}
  convnextv2.encoder.stages.{i}.layers.{j}.pwconv2.{weight,bias}
  convnextv2.layernorm.{weight,bias}
  classifier.{weight,bias}

Tested against: facebook/convnextv2-tiny-1k-224

This model uses level1 operators from KernelBench:
- LayerNorm from level1/normalization/6_LayerNorm
- GELU from level1/activations/8_GELU
- StochasticDepth from level1/mobile/4_StochasticDepth
- AdaptiveAvgPool2d from level1/pooling/7_AdaptiveAvgPool2d
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple

from ..level1.normalization._6_LayerNorm import Model as LayerNorm
from ..level1.activations._8_GELU import Model as GELU
from ..level1.mobile._4_StochasticDepth import Model as StochasticDepth
from ..level1.pooling._7_AdaptiveAvgPool2d import Model as AdaptiveAvgPool2d


# ============================================================================
# Component Modules
# ============================================================================

class ConvNextV2LayerNorm(nn.Module):
    """LayerNorm that supports channels-first (B, C, H, W) format.
    
    HF ConvNeXtV2 uses a custom LayerNorm that normalizes over the channel
    dimension. When input is (B, C, H, W), it permutes to (B, H, W, C),
    applies LayerNorm, then permutes back.
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            # channels-first: (B, C, H, W) -> normalize over C
            x = x.permute(0, 2, 3, 1)
            x = torch.nn.functional.layer_norm(x, self.normalized_shape,
                                                self.weight, self.bias, self.eps)
            x = x.permute(0, 3, 1, 2)
            return x
        else:
            return torch.nn.functional.layer_norm(x, self.normalized_shape,
                                                   self.weight, self.bias, self.eps)


class ConvNextV2GRN(nn.Module):
    """Global Response Normalization layer.
    
    GRN: x = x * (norm(x) / (mean(norm(x)) + eps)) * gamma + beta + x
    
    Weight shape in HF: (1, 1, 1, channels) for channels-last format.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, 1, 1, channels))
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, H, W, C) in channels-last format
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.weight * (x * nx) + self.bias + x


class ConvNextV2Layer(nn.Module):
    """Single ConvNeXt V2 block.
    
    Architecture: dwconv -> permute -> layernorm -> pwconv1 -> GELU -> GRN -> pwconv2 -> permute
    With residual connection and optional drop path.
    
    Key structure matches HF:
      dwconv.{weight,bias}
      layernorm.{weight,bias}
      pwconv1.{weight,bias}
      grn.{weight,bias}
      pwconv2.{weight,bias}
    
    Note: HF hardcodes eps=1e-6 for per-layer LayerNorm (not config.layer_norm_eps).
    """
    def __init__(self, dim: int, drop_path_rate: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # HF hardcodes eps=1e-6 for per-layer LayerNorm; input is channels-last (B, H, W, C)
        self.layernorm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = GELU()
        self.grn = ConvNextV2GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = StochasticDepth(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        # (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.layernorm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        # (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        x = residual + self.drop_path(x)
        return x


class ConvNextV2Stage(nn.Module):
    """ConvNeXt V2 stage with optional downsampling.
    
    Key structure matches HF:
      downsampling_layer.0.{weight,bias}  (LayerNorm, only for stages 1+)
      downsampling_layer.1.{weight,bias}  (Conv2d, only for stages 1+)
      layers.{j}.<ConvNextV2Layer keys>
    
    Note: HF hardcodes eps=1e-6 for downsampling LayerNorm (not config.layer_norm_eps).
    """
    def __init__(self, in_channels: int, out_channels: int, depth: int,
                 drop_path_rates: List[float], is_first_stage: bool = False):
        super().__init__()
        
        if is_first_stage:
            self.downsampling_layer = nn.ModuleList()
        else:
            # HF hardcodes eps=1e-6 for downsampling LayerNorm
            self.downsampling_layer = nn.ModuleList([
                ConvNextV2LayerNorm(in_channels, eps=1e-6),
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
            ])
        
        self.layers = nn.ModuleList([
            ConvNextV2Layer(out_channels, drop_path_rates[j])
            for j in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.downsampling_layer:
            x = module(x)
        for layer in self.layers:
            x = layer(x)
        return x


class ConvNextV2Embeddings(nn.Module):
    """Patch embedding stem.
    
    Key structure matches HF:
      patch_embeddings.{weight,bias}
      layernorm.{weight,bias}
    
    Note: HF hardcodes eps=1e-6 for embedding LayerNorm (not config.layer_norm_eps).
    """
    def __init__(self, num_channels: int, hidden_size: int, patch_size: int = 4):
        super().__init__()
        self.patch_embeddings = nn.Conv2d(num_channels, hidden_size,
                                           kernel_size=patch_size, stride=patch_size)
        # HF hardcodes eps=1e-6 for embedding LayerNorm
        self.layernorm = ConvNextV2LayerNorm(hidden_size, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embeddings(x)
        x = self.layernorm(x)
        return x


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    ConvNeXt V2 Image Classification Model.
    
    Aligned with HuggingFace ConvNextV2ForImageClassification.
    
    Uses level1 operators from KernelBench:
    - LayerNorm from level1/normalization/6_LayerNorm
    - GELU from level1/activations/8_GELU
    - StochasticDepth from level1/mobile/4_StochasticDepth
    - AdaptiveAvgPool2d from level1/pooling/7_AdaptiveAvgPool2d
    """

    def __init__(
        self,
        num_channels: int = 3,
        patch_size: int = 4,
        hidden_sizes: List[int] = None,
        depths: List[int] = None,
        num_labels: int = 1000,
        drop_path_rate: float = 0.0,
        layer_norm_eps: float = 1e-6,
        image_size: int = 224,
        **kwargs
    ):
        super().__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [96, 192, 384, 768]
        if depths is None:
            depths = [3, 3, 9, 3]
        
        self.num_stages = len(depths)
        
        # Generate drop path rates (linearly increasing)
        total_depth = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        
        # Embeddings (stem) - uses hardcoded eps=1e-6 internally
        self.embeddings = ConvNextV2Embeddings(num_channels, hidden_sizes[0], patch_size)
        
        # Encoder stages - uses hardcoded eps=1e-6 internally
        self.encoder = ConvNextV2Encoder(hidden_sizes, depths, dp_rates)
        
        # Final LayerNorm uses config.layer_norm_eps (typically 1e-12)
        self.layernorm = nn.LayerNorm(hidden_sizes[-1], eps=layer_norm_eps)
        self.classifier = nn.Linear(hidden_sizes[-1], num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embedding
        x = self.embeddings(x)
        
        # Encoder stages
        x = self.encoder(x)
        
        # Global average pooling: (B, C, H, W) -> (B, C)
        x = x.mean(dim=[-2, -1])
        
        # Final norm + classify
        x = self.layernorm(x)
        x = self.classifier(x)
        
        return x


class ConvNextV2Encoder(nn.Module):
    """Encoder containing all ConvNeXt V2 stages.
    
    Key structure matches HF:
      stages.{i}.<ConvNextV2Stage keys>
    """
    def __init__(self, hidden_sizes: List[int], depths: List[int],
                 dp_rates: List[float]):
        super().__init__()
        
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(len(depths)):
            in_ch = hidden_sizes[i - 1] if i > 0 else hidden_sizes[0]
            out_ch = hidden_sizes[i]
            stage = ConvNextV2Stage(
                in_channels=in_ch,
                out_channels=out_ch,
                depth=depths[i],
                drop_path_rates=dp_rates[cur:cur + depths[i]],
                is_first_stage=(i == 0),
            )
            self.stages.append(stage)
            cur += depths[i]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        return x
