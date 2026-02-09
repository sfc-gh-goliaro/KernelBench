"""
EfficientNet CNN Model

Implements EfficientNet architecture:
- Mobile Inverted Bottleneck (MBConv) blocks
- Squeeze-and-Excitation
- Compound scaling (depth, width, resolution)

Variants from Table 5:
- EfficientNet-B0: width=1.0, depth=1.0, image=224
- EfficientNet-B4: width=1.4, depth=1.8, image=380
- EfficientNet-B7: width=2.0, depth=3.1, image=600

This model uses level1 operators from KernelBench.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, List, Tuple

# Import level1 operators (used directly - no wrapping needed)
from ..level1.normalization._1_BatchNorm import Model as BatchNorm
from ..level1.activations._7_Swish import Model as Swish
from ..level1.activations._3_Sigmoid import Model as Sigmoid
from ..level1.convolutions._1_Conv2d_Standard import Model as Conv2d
from ..level1.pooling._7_AdaptiveAvgPool2d import Model as AdaptiveAvgPool2d


# ============================================================================
# Model Variants - configs loaded from HuggingFace
# ============================================================================

VARIANTS: Dict[str, str] = {
    "B0": "google/efficientnet-b0",
    "B4": "google/efficientnet-b4",
    "B7": "google/efficientnet-b7",
}

# Base architecture config (before scaling)
BASE_CONFIG = [
    # (channels, layers, stride, expand_ratio, kernel_size)
    (16, 1, 1, 1, 3),
    (24, 2, 2, 6, 3),
    (40, 2, 2, 6, 5),
    (80, 3, 2, 6, 3),
    (112, 3, 1, 6, 5),
    (192, 4, 2, 6, 5),
    (320, 1, 1, 6, 3),
]


# ============================================================================
# Component Modules (using level1 operators)
# ============================================================================

class ConvBNSiLU(nn.Module):
    """Conv2d + BatchNorm + SiLU using level1 operators."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, groups: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                             padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.swish = Swish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swish(self.bn(self.conv(x)))


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block using level1 operators."""
    def __init__(self, in_channels: int, squeeze_channels: int):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, in_channels, 1)
        self.swish = Swish()
        self.sigmoid = Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.avgpool(x)
        scale = self.swish(self.fc1(scale))
        scale = self.sigmoid(self.fc2(scale))
        return x * scale


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Convolution using level1 operators."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: int,
        se_ratio: float = 0.25,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        self.drop_rate = drop_rate
        
        expanded = in_channels * expand_ratio
        
        layers = []
        
        if expand_ratio != 1:
            layers.append(ConvBNSiLU(in_channels, expanded, 1))
        
        layers.append(ConvBNSiLU(expanded, expanded, kernel_size, stride, groups=expanded))
        
        squeeze_channels = max(1, int(in_channels * se_ratio))
        layers.append(SqueezeExcitation(expanded, squeeze_channels))
        
        layers.append(nn.Conv2d(expanded, out_channels, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        
        if self.use_residual:
            if self.training and self.drop_rate > 0:
                out = F.dropout(out, p=self.drop_rate, training=True)
            out = out + x
        
        return out


# ============================================================================
# Main Model Class
# ============================================================================

def _make_divisible(v: float, divisor: int = 8) -> int:
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Model(nn.Module):
    """
    EfficientNet CNN.
    
    Uses level1 operators from KernelBench:
    - BatchNorm from level1/normalization/1_BatchNorm
    - Swish/SiLU from level1/activations/7_Swish
    - Sigmoid from level1/activations/3_Sigmoid
    - AdaptiveAvgPool2d from level1/pooling/7_AdaptiveAvgPool2d
    
    Supports variants: B0, B4, B7 (configs loaded from HuggingFace)
    """
    
    VARIANTS = VARIANTS
    
    @classmethod
    def from_pretrained(cls, variant: str = "B0", operator_level: Optional[OperatorLevel] = None, **kwargs):
        """Create model with config loaded from HuggingFace."""
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}. Available: {list(VARIANTS.keys())}")
        hf_config = load_hf_config(VARIANTS[variant])
        hf_config.update(kwargs)
        return cls(operator_level=operator_level, **hf_config)
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        operator_level: Optional[OperatorLevel] = None,
        **kwargs
    ):
        width_mult = kwargs.get('width_mult', 1.0)
        depth_mult = kwargs.get('depth_mult', 1.0)
        image_size = kwargs.get('image_size', 224)
        num_classes = kwargs.get('num_classes', 1000)
        dropout = kwargs.get('dropout', 0.2)
        stem_channels = kwargs.get('stem_channels', 32)
        drop_connect_rate = kwargs.get('drop_connect_rate', 0.2)
        
        if config is None:
            config = ModelConfig(
                hidden_size=stem_channels,
                vocab_size=num_classes,
            )
        
        super().__init__()
        
        stem_out = _make_divisible(stem_channels * width_mult)
        self.stem = ConvBNSiLU(3, stem_out, 3, stride=2)
        
        self.stages = nn.ModuleList()
        in_channels = stem_out
        total_blocks = sum(int(math.ceil(layers * depth_mult)) for _, layers, _, _, _ in BASE_CONFIG)
        block_idx = 0
        
        for out_ch, num_layers, stride, expand_ratio, kernel_size in BASE_CONFIG:
            out_channels = _make_divisible(out_ch * width_mult)
            num_layers = int(math.ceil(num_layers * depth_mult))
            
            stage = []
            for i in range(num_layers):
                s = stride if i == 0 else 1
                drop_rate = drop_connect_rate * block_idx / total_blocks
                stage.append(MBConv(
                    in_channels, out_channels, kernel_size, s, 
                    expand_ratio, drop_rate=drop_rate
                ))
                in_channels = out_channels
                block_idx += 1
            
            self.stages.append(nn.Sequential(*stage))
        
        head_channels = _make_divisible(1280 * width_mult)
        self.head_conv = ConvBNSiLU(in_channels, head_channels, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(head_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        
        for stage in self.stages:
            x = stage(x)
        
        x = self.head_conv(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 16
image_size = 224
num_classes = 1000


def get_inputs():
    return [torch.randn(batch_size, 3, image_size, image_size)]


def get_init_inputs():
    return [{
        'width_mult': 1.0,
        'depth_mult': 1.0,
        'image_size': image_size,
        'num_classes': num_classes,
    }]
