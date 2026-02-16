"""
EfficientNet V2 Small CNN Model

Implements EfficientNetV2-S architecture aligned with timm's EfficientNet class:
- Fused MBConv (EdgeResidual) blocks in early stages
- MBConv (InvertedResidual) blocks with SE in later stages
- Conv2dSame (TF-style SAME padding) for strided convolutions
- SiLU (Swish) activation
- BatchNorm with eps=0.001

timm weight structure (tf_efficientnetv2_s.in21k):
  conv_stem.weight  (Conv2dSame)
  bn1.{weight,bias,running_mean,running_var,num_batches_tracked}
  blocks.{stage}.{block}.<block-specific keys>
  conv_head.weight
  bn2.{weight,bias,...}
  classifier.{weight,bias}

Block types:
  Stage 0: ConvBnAct - conv.weight, bn1.{...}
  Stages 1-2: EdgeResidual (Fused MBConv) - conv_exp.weight, bn1.{...}, conv_pwl.weight, bn2.{...}
  Stages 3-5: InvertedResidual (MBConv+SE) - conv_pw.weight, bn1.{...}, conv_dw.weight, bn2.{...},
      se.{conv_reduce,conv_expand}.{weight,bias}, conv_pwl.weight, bn3.{...}

This model uses level1 operators from KernelBench:
- BatchNorm from level1/normalization/1_BatchNorm
- Swish (SiLU) from level1/activations/7_Swish
- Sigmoid from level1/activations/3_Sigmoid
- AdaptiveAvgPool2d from level1/pooling/7_AdaptiveAvgPool2d
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple

from ..level1.normalization._1_BatchNorm import Model as BatchNorm
from ..level1.activations._7_Swish import Model as Swish
from ..level1.activations._3_Sigmoid import Model as Sigmoid
from ..level1.pooling._7_AdaptiveAvgPool2d import Model as AdaptiveAvgPool2d


# ============================================================================
# Component Modules (matching timm's key structure exactly)
# ============================================================================

BN_EPS = 0.001


def _same_pad(x: torch.Tensor, kernel_size: tuple, stride: tuple) -> torch.Tensor:
    """Apply TF-style SAME padding."""
    ih, iw = x.shape[-2:]
    kh, kw = kernel_size
    sh, sw = stride
    oh = (ih + sh - 1) // sh
    ow = (iw + sw - 1) // sw
    pad_h = max((oh - 1) * sh + kh - ih, 0)
    pad_w = max((ow - 1) * sw + kw - iw, 0)
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2,
                       pad_h // 2, pad_h - pad_h // 2])
    return x


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation block matching timm's key structure.
    Keys: conv_reduce.{weight,bias}, conv_expand.{weight,bias}
    """
    def __init__(self, in_channels: int, se_channels: int):
        super().__init__()
        self.conv_reduce = nn.Conv2d(in_channels, se_channels, 1, bias=True)
        self.act1 = nn.SiLU(inplace=True)
        self.conv_expand = nn.Conv2d(se_channels, in_channels, 1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        se = x.mean(dim=[-2, -1], keepdim=True)
        se = self.act1(self.conv_reduce(se))
        se = self.gate(self.conv_expand(se))
        return x * se


class ConvBnAct(nn.Module):
    """Simple Conv + BN + SiLU block matching timm's ConvBnAct key structure.
    Keys: conv.weight, bn1.{weight,bias,...}
    Includes residual connection when in_channels == out_channels and stride == 1.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1):
        super().__init__()
        self.has_skip = stride == 1 and in_channels == out_channels
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=BN_EPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        out = F.silu(self.bn1(self.conv(x)), inplace=True)
        if self.has_skip:
            out = out + shortcut
        return out


class EdgeResidual(nn.Module):
    """Fused MBConv (EdgeResidual) block matching timm's key structure.
    Keys: conv_exp.weight, bn1.{...}, conv_pwl.weight, bn2.{...}
    Uses SAME padding for stride > 1.
    """
    def __init__(self, in_channels: int, exp_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.has_residual = stride == 1 and in_channels == out_channels
        self.stride = stride
        self.kernel_size = (kernel_size, kernel_size)
        self.use_same_pad = stride > 1

        if self.use_same_pad:
            self.conv_exp = nn.Conv2d(in_channels, exp_channels, kernel_size, stride,
                                       padding=0, bias=False)
        else:
            padding = (kernel_size - 1) // 2
            self.conv_exp = nn.Conv2d(in_channels, exp_channels, kernel_size, stride,
                                       padding, bias=False)
        self.bn1 = nn.BatchNorm2d(exp_channels, eps=BN_EPS)
        self.conv_pwl = nn.Conv2d(exp_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=BN_EPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if self.use_same_pad:
            x = _same_pad(x, self.kernel_size, (self.stride, self.stride))
        out = F.silu(self.bn1(self.conv_exp(x)), inplace=True)
        out = self.bn2(self.conv_pwl(out))
        if self.has_residual:
            out = out + residual
        return out


class InvertedResidual(nn.Module):
    """MBConv block with SE matching timm's InvertedResidual key structure.
    Keys: conv_pw.weight, bn1.{...}, conv_dw.weight, bn2.{...},
          se.conv_reduce.{weight,bias}, se.conv_expand.{weight,bias},
          conv_pwl.weight, bn3.{...}
    Uses SAME padding for strided depthwise conv.
    """
    def __init__(self, in_channels: int, exp_channels: int, out_channels: int,
                 dw_kernel_size: int = 3, stride: int = 1, se_channels: int = 0):
        super().__init__()
        self.has_residual = stride == 1 and in_channels == out_channels
        self.stride = stride
        self.dw_kernel_size = (dw_kernel_size, dw_kernel_size)
        self.use_same_pad = stride > 1

        # Pointwise expansion
        self.conv_pw = nn.Conv2d(in_channels, exp_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(exp_channels, eps=BN_EPS)

        # Depthwise convolution
        if self.use_same_pad:
            self.conv_dw = nn.Conv2d(exp_channels, exp_channels, dw_kernel_size,
                                      stride, padding=0, groups=exp_channels, bias=False)
        else:
            padding = (dw_kernel_size - 1) // 2
            self.conv_dw = nn.Conv2d(exp_channels, exp_channels, dw_kernel_size,
                                      stride, padding, groups=exp_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(exp_channels, eps=BN_EPS)

        # Squeeze-and-Excitation
        self.se = SqueezeExcite(exp_channels, se_channels) if se_channels > 0 else nn.Identity()

        # Pointwise projection
        self.conv_pwl = nn.Conv2d(exp_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels, eps=BN_EPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.silu(self.bn1(self.conv_pw(x)), inplace=True)
        if self.use_same_pad:
            out = _same_pad(out, self.dw_kernel_size, (self.stride, self.stride))
        out = F.silu(self.bn2(self.conv_dw(out)), inplace=True)
        out = self.se(out)
        out = self.bn3(self.conv_pwl(out))
        if self.has_residual:
            out = out + residual
        return out


# ============================================================================
# Architecture Definition for tf_efficientnetv2_s
# ============================================================================

EFFICIENTNETV2_S_ARCH = [
    # Stage 0: ConvBnAct (Fused-MBConv1, no expansion)
    [
        ('convbnact', 24, 24, 3, 1),
        ('convbnact', 24, 24, 3, 1),
    ],
    # Stage 1: EdgeResidual (Fused-MBConv4)
    [
        ('edge', 24, 96, 48, 3, 2),
        ('edge', 48, 192, 48, 3, 1),
        ('edge', 48, 192, 48, 3, 1),
        ('edge', 48, 192, 48, 3, 1),
    ],
    # Stage 2: EdgeResidual (Fused-MBConv4)
    [
        ('edge', 48, 192, 64, 3, 2),
        ('edge', 64, 256, 64, 3, 1),
        ('edge', 64, 256, 64, 3, 1),
        ('edge', 64, 256, 64, 3, 1),
    ],
    # Stage 3: InvertedResidual (MBConv4 + SE)
    [
        ('ir', 64, 256, 128, 3, 2, 16),
        ('ir', 128, 512, 128, 3, 1, 32),
        ('ir', 128, 512, 128, 3, 1, 32),
        ('ir', 128, 512, 128, 3, 1, 32),
        ('ir', 128, 512, 128, 3, 1, 32),
        ('ir', 128, 512, 128, 3, 1, 32),
    ],
    # Stage 4: InvertedResidual (MBConv6 + SE)
    [
        ('ir', 128, 768, 160, 3, 1, 32),
        ('ir', 160, 960, 160, 3, 1, 40),
        ('ir', 160, 960, 160, 3, 1, 40),
        ('ir', 160, 960, 160, 3, 1, 40),
        ('ir', 160, 960, 160, 3, 1, 40),
        ('ir', 160, 960, 160, 3, 1, 40),
        ('ir', 160, 960, 160, 3, 1, 40),
        ('ir', 160, 960, 160, 3, 1, 40),
        ('ir', 160, 960, 160, 3, 1, 40),
    ],
    # Stage 5: InvertedResidual (MBConv6 + SE)
    [
        ('ir', 160, 960, 256, 3, 2, 40),
        ('ir', 256, 1536, 256, 3, 1, 64),
        ('ir', 256, 1536, 256, 3, 1, 64),
        ('ir', 256, 1536, 256, 3, 1, 64),
        ('ir', 256, 1536, 256, 3, 1, 64),
        ('ir', 256, 1536, 256, 3, 1, 64),
        ('ir', 256, 1536, 256, 3, 1, 64),
        ('ir', 256, 1536, 256, 3, 1, 64),
        ('ir', 256, 1536, 256, 3, 1, 64),
        ('ir', 256, 1536, 256, 3, 1, 64),
        ('ir', 256, 1536, 256, 3, 1, 64),
        ('ir', 256, 1536, 256, 3, 1, 64),
        ('ir', 256, 1536, 256, 3, 1, 64),
        ('ir', 256, 1536, 256, 3, 1, 64),
        ('ir', 256, 1536, 256, 3, 1, 64),
    ],
]


def _build_blocks(arch):
    """Build nn.Sequential stages from architecture definition."""
    stages = nn.ModuleList()
    for stage_def in arch:
        blocks = []
        for spec in stage_def:
            if spec[0] == 'convbnact':
                _, in_ch, out_ch, kernel, stride = spec
                blocks.append(ConvBnAct(in_ch, out_ch, kernel, stride))
            elif spec[0] == 'edge':
                _, in_ch, exp_ch, out_ch, kernel, stride = spec
                blocks.append(EdgeResidual(in_ch, exp_ch, out_ch, kernel, stride))
            elif spec[0] == 'ir':
                _, in_ch, exp_ch, out_ch, dw_k, stride, se_ch = spec
                blocks.append(InvertedResidual(in_ch, exp_ch, out_ch, dw_k, stride, se_ch))
        stages.append(nn.Sequential(*blocks))
    return stages


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    EfficientNetV2-S CNN.

    Aligned with timm's tf_efficientnetv2_s.in21k.
    Uses EfficientNet-style head: conv_head -> bn2 -> pool -> classifier.

    Uses level1 operators from KernelBench:
    - BatchNorm from level1/normalization/1_BatchNorm
    - Swish (SiLU) from level1/activations/7_Swish
    - Sigmoid from level1/activations/3_Sigmoid
    - AdaptiveAvgPool2d from level1/pooling/7_AdaptiveAvgPool2d
    """

    def __init__(
        self,
        num_classes: int = 21843,
        in_chans: int = 3,
        stem_size: int = 24,
        num_features: int = 1280,
        last_stage_channels: int = 256,
        **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self._stem_ks = (3, 3)
        self._stem_stride = (2, 2)

        # Stem (Conv2dSame for stride=2)
        self.conv_stem = nn.Conv2d(in_chans, stem_size, 3, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(stem_size, eps=BN_EPS)

        # Blocks
        self.blocks = nn.Sequential(*_build_blocks(EFFICIENTNETV2_S_ARCH))

        # Head
        self.conv_head = nn.Conv2d(last_stage_channels, num_features, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features, eps=BN_EPS)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _same_pad(x, self._stem_ks, self._stem_stride)
        x = F.silu(self.bn1(self.conv_stem(x)), inplace=True)
        x = self.blocks(x)
        x = F.silu(self.bn2(self.conv_head(x)), inplace=True)
        x = self.global_pool(x)
        x = self.flatten(x)
        return self.classifier(x)
