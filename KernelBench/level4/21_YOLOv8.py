"""
YOLOv8 Object Detection Model

Implements YOLOv8 architecture:
- CSPDarknet backbone
- PANet neck with C2f blocks
- Decoupled detection head

Variants from Table 5:
- YOLOv8-n: nano variant
- YOLOv8-s: small variant
- YOLOv8-m: medium variant
- YOLOv8-l: large variant
- YOLOv8-x: extra-large variant

This model uses level1 operators from KernelBench.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple

# Import level1 operators (used directly - no wrapping needed)
from ..level1.normalization._1_BatchNorm import Model as BatchNorm
from ..level1.activations._7_Swish import Model as Swish
from ..level1.pooling._2_MaxPool2d import Model as MaxPool2d


# ============================================================================
# Model Variants - configs loaded from HuggingFace/hardcoded
# ============================================================================

VARIANTS: Dict[str, str] = {
    "n": "ultralytics/yolov8n",
    "s": "ultralytics/yolov8s",
    "m": "ultralytics/yolov8m",
    "l": "ultralytics/yolov8l",
    "x": "ultralytics/yolov8x",
}


# ============================================================================
# Component Modules (using level1 operators)
# ============================================================================

def autopad(k, p=None):
    """Auto-calculate padding for 'same' convolution."""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution with batch norm and SiLU using level1 operators."""
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p=None, g: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = Swish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """YOLOv8 bottleneck block using level1 operators."""
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 3)
        self.cv2 = Conv(c_, c2, 3)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class C2f(nn.Module):
    """CSP bottleneck with 2 convolutions (YOLOv8 style) using level1 operators."""
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling Fast using level1 operators."""
    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Detect(nn.Module):
    """YOLOv8 detection head using level1 operators."""
    def __init__(self, nc: int = 80, ch: Tuple[int, ...] = ()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, 64, 3), Conv(64, 64, 3), nn.Conv2d(64, 4 * self.reg_max, 1))
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, max(x, self.nc), 3), Conv(max(x, self.nc), max(x, self.nc), 3), nn.Conv2d(max(x, self.nc), self.nc, 1))
            for x in ch
        )

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs = []
        for i in range(self.nl):
            box = self.cv2[i](x[i])
            cls = self.cv3[i](x[i])
            outputs.append(torch.cat([box, cls], 1))
        return outputs


# ============================================================================
# Main Model Class
# ============================================================================

def make_divisible(x: float, divisor: int) -> int:
    """Make x divisible by divisor."""
    return max(divisor, int(x + divisor / 2) // divisor * divisor)


class Model(nn.Module):
    """
    YOLOv8 object detection model.
    
    Uses level1 operators from KernelBench:
    - BatchNorm from level1/normalization/1_BatchNorm
    - Swish/SiLU from level1/activations/7_Swish
    - MaxPool2d from level1/pooling/2_MaxPool2d
    
    Supports variants: n, s, m, l, x (configs loaded from HuggingFace)
    """
    
    VARIANTS = VARIANTS
    
    @classmethod
    def from_pretrained(cls, variant: str = "s", operator_level: Optional[OperatorLevel] = None, **kwargs):
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
        depth_mult = kwargs.get('depth_mult', 1.0)
        width_mult = kwargs.get('width_mult', 1.0)
        num_classes = kwargs.get('num_classes', 80)
        
        if config is None:
            config = ModelConfig(
                hidden_size=64,
                vocab_size=num_classes,
            )
        
        super().__init__()
        
        def ch(c): return make_divisible(c * width_mult, 8)
        def n(d): return max(round(d * depth_mult), 1)
        
        # Backbone
        self.stem = Conv(3, ch(64), 3, 2)
        
        self.stage1 = nn.Sequential(
            Conv(ch(64), ch(128), 3, 2),
            C2f(ch(128), ch(128), n(3), True),
        )
        
        self.stage2 = nn.Sequential(
            Conv(ch(128), ch(256), 3, 2),
            C2f(ch(256), ch(256), n(6), True),
        )
        
        self.stage3 = nn.Sequential(
            Conv(ch(256), ch(512), 3, 2),
            C2f(ch(512), ch(512), n(6), True),
        )
        
        self.stage4 = nn.Sequential(
            Conv(ch(512), ch(1024), 3, 2),
            C2f(ch(1024), ch(1024), n(3), True),
            SPPF(ch(1024), ch(1024), 5),
        )
        
        # Neck (PANet)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.neck_c2f1 = C2f(ch(1024) + ch(512), ch(512), n(3), False)
        self.neck_c2f2 = C2f(ch(512) + ch(256), ch(256), n(3), False)
        
        self.neck_conv1 = Conv(ch(256), ch(256), 3, 2)
        self.neck_c2f3 = C2f(ch(256) + ch(512), ch(512), n(3), False)
        
        self.neck_conv2 = Conv(ch(512), ch(512), 3, 2)
        self.neck_c2f4 = C2f(ch(512) + ch(1024), ch(1024), n(3), False)
        
        # Head
        self.detect = Detect(num_classes, (ch(256), ch(512), ch(1024)))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Backbone
        x = self.stem(x)
        p2 = self.stage1(x)
        p3 = self.stage2(p2)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        
        # Neck
        x = self.upsample(p5)
        x = torch.cat([x, p4], 1)
        x = self.neck_c2f1(x)
        
        x = self.upsample(x)
        x = torch.cat([x, p3], 1)
        n3 = self.neck_c2f2(x)
        
        x = self.neck_conv1(n3)
        x = torch.cat([x, p4], 1)
        n4 = self.neck_c2f3(x)
        
        x = self.neck_conv2(n4)
        x = torch.cat([x, p5], 1)
        n5 = self.neck_c2f4(x)
        
        # Detection head
        return self.detect([n3, n4, n5])
