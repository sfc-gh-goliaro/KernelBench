"""
YOLOv8 Object Detection Model

A modern object detection model implementing YOLOv8 architecture:
- CSP backbone with C2f blocks
- SPPF (Spatial Pyramid Pooling Fast)
- FPN + PAN neck
- Anchor-free detection head

Reference: YOLOv8x
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ConvBNSiLU(nn.Module):
    """Conv2d + BatchNorm + SiLU activation."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck block."""
    def __init__(self, in_channels: int, out_channels: int, shortcut: bool = True, expansion: float = 0.5):
        super().__init__()
        hidden = int(out_channels * expansion)
        self.cv1 = ConvBNSiLU(in_channels, hidden, kernel_size=3, padding=1)
        self.cv2 = ConvBNSiLU(hidden, out_channels, kernel_size=3, padding=1)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions (YOLOv8 specific)."""
    def __init__(self, in_channels: int, out_channels: int, n: int = 1, shortcut: bool = False, expansion: float = 0.5):
        super().__init__()
        self.c = int(out_channels * expansion)
        self.cv1 = ConvBNSiLU(in_channels, 2 * self.c, kernel_size=1)
        self.cv2 = ConvBNSiLU((2 + n) * self.c, out_channels, kernel_size=1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling Fast."""
    def __init__(self, in_channels: int, out_channels: int, k: int = 5):
        super().__init__()
        c_ = in_channels // 2
        self.cv1 = ConvBNSiLU(in_channels, c_, kernel_size=1)
        self.cv2 = ConvBNSiLU(c_ * 4, out_channels, kernel_size=1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Backbone(nn.Module):
    """YOLOv8 Backbone."""
    def __init__(self, in_channels: int = 3, base_channels: int = 64, depth_multiple: float = 1.0, width_multiple: float = 1.0):
        super().__init__()

        def ch(c): return max(round(c * width_multiple), 1)
        def depth(d): return max(round(d * depth_multiple), 1)

        # Stem
        self.stem = ConvBNSiLU(in_channels, ch(base_channels), kernel_size=3, stride=2, padding=1)

        # Stage 1
        self.stage1 = nn.Sequential(
            ConvBNSiLU(ch(base_channels), ch(base_channels * 2), kernel_size=3, stride=2, padding=1),
            C2f(ch(base_channels * 2), ch(base_channels * 2), n=depth(3), shortcut=True),
        )

        # Stage 2
        self.stage2 = nn.Sequential(
            ConvBNSiLU(ch(base_channels * 2), ch(base_channels * 4), kernel_size=3, stride=2, padding=1),
            C2f(ch(base_channels * 4), ch(base_channels * 4), n=depth(6), shortcut=True),
        )

        # Stage 3
        self.stage3 = nn.Sequential(
            ConvBNSiLU(ch(base_channels * 4), ch(base_channels * 8), kernel_size=3, stride=2, padding=1),
            C2f(ch(base_channels * 8), ch(base_channels * 8), n=depth(6), shortcut=True),
        )

        # Stage 4
        self.stage4 = nn.Sequential(
            ConvBNSiLU(ch(base_channels * 8), ch(base_channels * 16), kernel_size=3, stride=2, padding=1),
            C2f(ch(base_channels * 16), ch(base_channels * 16), n=depth(3), shortcut=True),
            SPPF(ch(base_channels * 16), ch(base_channels * 16)),
        )

        self.out_channels = [ch(base_channels * 4), ch(base_channels * 8), ch(base_channels * 16)]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)   # 1/8
        p4 = self.stage3(p3)  # 1/16
        p5 = self.stage4(p4)  # 1/32
        return p3, p4, p5


class Neck(nn.Module):
    """YOLOv8 FPN + PAN Neck."""
    def __init__(self, in_channels: List[int], depth_multiple: float = 1.0):
        super().__init__()
        c3, c4, c5 = in_channels

        def depth(d): return max(round(d * depth_multiple), 1)

        # Top-down pathway (FPN)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.reduce_p5 = ConvBNSiLU(c5, c4, kernel_size=1)
        self.c2f_p4 = C2f(c4 + c4, c4, n=depth(3), shortcut=False)

        self.reduce_p4 = ConvBNSiLU(c4, c3, kernel_size=1)
        self.c2f_p3 = C2f(c3 + c3, c3, n=depth(3), shortcut=False)

        # Bottom-up pathway (PAN)
        self.down_p3 = ConvBNSiLU(c3, c3, kernel_size=3, stride=2, padding=1)
        self.c2f_n4 = C2f(c3 + c4, c4, n=depth(3), shortcut=False)

        self.down_p4 = ConvBNSiLU(c4, c4, kernel_size=3, stride=2, padding=1)
        self.c2f_n5 = C2f(c4 + c5, c5, n=depth(3), shortcut=False)

        self.out_channels = [c3, c4, c5]

    def forward(self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p3, p4, p5 = features

        # Top-down
        p5_up = self.upsample(self.reduce_p5(p5))
        p4 = self.c2f_p4(torch.cat([p5_up, p4], 1))

        p4_up = self.upsample(self.reduce_p4(p4))
        p3 = self.c2f_p3(torch.cat([p4_up, p3], 1))

        # Bottom-up
        p3_down = self.down_p3(p3)
        n4 = self.c2f_n4(torch.cat([p3_down, p4], 1))

        n4_down = self.down_p4(n4)
        n5 = self.c2f_n5(torch.cat([n4_down, p5], 1))

        return p3, n4, n5


class DetectionHead(nn.Module):
    """YOLOv8 Detection Head (anchor-free)."""
    def __init__(self, in_channels: List[int], num_classes: int = 80, reg_max: int = 16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.num_outputs = num_classes + reg_max * 4

        self.heads = nn.ModuleList()
        for c in in_channels:
            # Classification branch
            cls_conv = nn.Sequential(
                ConvBNSiLU(c, c, kernel_size=3, padding=1),
                ConvBNSiLU(c, c, kernel_size=3, padding=1),
                nn.Conv2d(c, num_classes, kernel_size=1),
            )

            # Regression branch (box + DFL)
            reg_conv = nn.Sequential(
                ConvBNSiLU(c, c, kernel_size=3, padding=1),
                ConvBNSiLU(c, c, kernel_size=3, padding=1),
                nn.Conv2d(c, reg_max * 4, kernel_size=1),
            )

            self.heads.append(nn.ModuleDict({'cls': cls_conv, 'reg': reg_conv}))

    def forward(self, features: Tuple[torch.Tensor, ...]) -> List[torch.Tensor]:
        outputs = []
        for feat, head in zip(features, self.heads):
            cls_out = head['cls'](feat)
            reg_out = head['reg'](feat)

            B, _, H, W = cls_out.shape
            cls_out = cls_out.view(B, self.num_classes, H * W).permute(0, 2, 1)
            reg_out = reg_out.view(B, self.reg_max * 4, H * W).permute(0, 2, 1)

            outputs.append(torch.cat([reg_out, cls_out], dim=-1))

        return outputs


class Model(nn.Module):
    """YOLOv8 Object Detection Model."""
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 80,
        base_channels: int = 64,
        depth_multiple: float = 1.0,
        width_multiple: float = 1.25,
        reg_max: int = 16,
    ):
        super().__init__()

        self.backbone = Backbone(in_channels, base_channels, depth_multiple, width_multiple)
        self.neck = Neck(self.backbone.out_channels, depth_multiple)
        self.head = DetectionHead(self.neck.out_channels, num_classes, reg_max)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = self.backbone(x)
        features = self.neck(features)
        outputs = self.head(features)
        return outputs


# Configuration (reduced for benchmarking)
batch_size = 4
img_size = 640
in_channels = 3
num_classes = 80

base_channels = 48
depth_multiple = 0.67
width_multiple = 0.75


def get_inputs():
    return [torch.randn(batch_size, in_channels, img_size, img_size)]


def get_init_inputs():
    return [{
        'in_channels': in_channels,
        'num_classes': num_classes,
        'base_channels': base_channels,
        'depth_multiple': depth_multiple,
        'width_multiple': width_multiple,
    }]

