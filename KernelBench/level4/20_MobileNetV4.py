"""
MobileNetV4 Conv Medium CNN Model

Implements MobileNetV4 architecture aligned with timm's MobileNetV3 class
(which hosts MobileNetV4 variants):
- Edge Residual blocks
- Universal Inverted Residual (UIR) blocks
- MobileNetV3-style efficient head (pool before head conv)
- BatchNorm + ReLU activations

timm weight structure (mobilenetv4_conv_medium.e500_r256_in1k):
  conv_stem.weight
  bn1.{weight,bias,running_mean,running_var,num_batches_tracked}
  blocks.{stage}.{block}.<block-specific keys>
  blocks.4.0.conv.weight   (final 1x1 conv)
  blocks.4.0.bn1.{weight,bias,...}
  conv_head.weight
  norm_head.{weight,bias,running_mean,running_var,num_batches_tracked}
  classifier.{weight,bias}

Block types:
  EdgeResidual: conv_exp, bn1, conv_pwl, bn2
  UniversalInvertedResidual: dw_start.{conv,bn}, pw_exp.{conv,bn},
      dw_mid.{conv,bn}, pw_proj.{conv,bn}
  ConvBnAct: conv, bn1

This model uses level1 operators from KernelBench:
- BatchNorm from level1/normalization/1_BatchNorm
- ReLU from level1/activations/1_ReLU
- AdaptiveAvgPool2d from level1/pooling/7_AdaptiveAvgPool2d
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple

from ..level1.normalization._1_BatchNorm import Model as BatchNorm
from ..level1.activations._1_ReLU import Model as ReLU
from ..level1.pooling._7_AdaptiveAvgPool2d import Model as AdaptiveAvgPool2d


# ============================================================================
# Component Modules (matching timm's key structure exactly)
# ============================================================================

class ConvNormAct(nn.Module):
    """Conv2d + BatchNorm2d + optional ReLU, matching timm's ConvNormAct key structure.
    Keys: conv.weight, bn.{weight,bias,...}
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, groups: int = 1, act: bool = True):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.use_act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.conv(x))
        if self.use_act:
            x = F.relu(x, inplace=True)
        return x


class EdgeResidual(nn.Module):
    """Edge Residual block matching timm's EdgeResidual key structure.
    Keys: conv_exp.weight, bn1.{...}, conv_pwl.weight, bn2.{...}
    """
    def __init__(self, in_channels: int, exp_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.has_residual = stride == 1 and in_channels == out_channels
        padding = (kernel_size - 1) // 2
        self.conv_exp = nn.Conv2d(in_channels, exp_channels, kernel_size, stride,
                                   padding, bias=False)
        self.bn1 = nn.BatchNorm2d(exp_channels)
        self.conv_pwl = nn.Conv2d(exp_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv_exp(x)), inplace=True)
        out = self.bn2(self.conv_pwl(out))
        if self.has_residual:
            out = out + residual
        return out


class UniversalInvertedResidual(nn.Module):
    """Universal Inverted Residual block matching timm's key structure.
    Keys: dw_start.{conv,bn}, pw_exp.{conv,bn}, dw_mid.{conv,bn}, pw_proj.{conv,bn}
    Some components may be Identity() (no weights).
    """
    def __init__(self, in_channels: int, out_channels: int,
                 dw_start_k: int = 0, exp_channels: int = 0,
                 dw_mid_k: int = 0, stride: int = 1):
        super().__init__()
        self.has_residual = stride == 1 and in_channels == out_channels

        if dw_start_k > 0:
            self.dw_start = ConvNormAct(in_channels, in_channels, dw_start_k,
                                         stride=1, groups=in_channels, act=False)
        else:
            self.dw_start = nn.Identity()

        self.pw_exp = ConvNormAct(in_channels, exp_channels, 1, act=True)

        if dw_mid_k > 0:
            self.dw_mid = ConvNormAct(exp_channels, exp_channels, dw_mid_k,
                                       stride=stride, groups=exp_channels, act=True)
        else:
            self.dw_mid = nn.Identity()

        self.pw_proj = ConvNormAct(exp_channels, out_channels, 1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.dw_start(x)
        out = self.pw_exp(out)
        out = self.dw_mid(out)
        out = self.pw_proj(out)
        if self.has_residual:
            out = out + residual
        return out


class ConvBnAct(nn.Module):
    """Simple Conv + BN + Act block matching timm's ConvBnAct key structure.
    Keys: conv.weight, bn1.{weight,bias,...}
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1,
                 stride: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn1(self.conv(x)), inplace=True)


# ============================================================================
# Architecture Definition for mobilenetv4_conv_medium
# ============================================================================

MOBILENETV4_CONV_MEDIUM_ARCH = [
    # Stage 0: EdgeResidual
    [('edge', 32, 128, 48, 3, 2)],
    # Stage 1: UIR blocks
    [
        ('uir', 48, 80, 3, 192, 5, 2),
        ('uir', 80, 80, 3, 160, 3, 1),
    ],
    # Stage 2: UIR blocks
    [
        ('uir', 80, 160, 3, 480, 5, 2),
        ('uir', 160, 160, 3, 640, 3, 1),
        ('uir', 160, 160, 3, 640, 3, 1),
        ('uir', 160, 160, 3, 640, 5, 1),
        ('uir', 160, 160, 3, 640, 3, 1),
        ('uir', 160, 160, 3, 640, 0, 1),
        ('uir', 160, 160, 0, 320, 0, 1),
        ('uir', 160, 160, 3, 640, 0, 1),
    ],
    # Stage 3: UIR blocks
    [
        ('uir', 160, 256, 5, 960, 5, 2),
        ('uir', 256, 256, 5, 1024, 5, 1),
        ('uir', 256, 256, 3, 1024, 5, 1),
        ('uir', 256, 256, 3, 1024, 5, 1),
        ('uir', 256, 256, 0, 1024, 0, 1),
        ('uir', 256, 256, 3, 1024, 0, 1),
        ('uir', 256, 256, 3, 512, 5, 1),
        ('uir', 256, 256, 5, 1024, 5, 1),
        ('uir', 256, 256, 0, 1024, 0, 1),
        ('uir', 256, 256, 0, 1024, 0, 1),
        ('uir', 256, 256, 5, 512, 0, 1),
    ],
    # Stage 4: Final ConvBnAct
    [('convbnact', 256, 960, 1, 1)],
]


def _build_blocks(arch):
    """Build nn.Sequential stages from architecture definition."""
    stages = nn.ModuleList()
    for stage_def in arch:
        blocks = []
        for spec in stage_def:
            if spec[0] == 'edge':
                _, in_ch, exp_ch, out_ch, kernel, stride = spec
                blocks.append(EdgeResidual(in_ch, exp_ch, out_ch, kernel, stride))
            elif spec[0] == 'uir':
                _, in_ch, out_ch, dw_start_k, exp_ch, dw_mid_k, stride = spec
                blocks.append(UniversalInvertedResidual(
                    in_ch, out_ch, dw_start_k, exp_ch, dw_mid_k, stride))
            elif spec[0] == 'convbnact':
                _, in_ch, out_ch, kernel, stride = spec
                blocks.append(ConvBnAct(in_ch, out_ch, kernel, stride))
        stages.append(nn.Sequential(*blocks))
    return stages


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    MobileNetV4 Conv Medium CNN.

    Aligned with timm's mobilenetv4_conv_medium.e500_r256_in1k.
    Uses MobileNetV3 efficient head: pool -> conv_head -> norm_head -> classifier.

    Uses level1 operators from KernelBench:
    - BatchNorm from level1/normalization/1_BatchNorm
    - ReLU from level1/activations/1_ReLU
    - AdaptiveAvgPool2d from level1/pooling/7_AdaptiveAvgPool2d
    """

    def __init__(
        self,
        num_classes: int = 1000,
        in_chans: int = 3,
        stem_size: int = 32,
        num_features: int = 1280,
        head_channels: int = 960,
        **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes

        # Stem
        self.conv_stem = nn.Conv2d(in_chans, stem_size, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stem_size)

        # Blocks
        self.blocks = nn.Sequential(*_build_blocks(MOBILENETV4_CONV_MEDIUM_ARCH))

        # Head (MobileNetV3-style efficient head)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_head = nn.Conv2d(head_channels, num_features, 1, bias=False)
        self.norm_head = nn.BatchNorm2d(num_features)
        self.flatten = nn.Flatten(1)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv_stem(x)), inplace=True)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = F.relu(self.norm_head(self.conv_head(x)), inplace=True)
        x = self.flatten(x)
        return self.classifier(x)
