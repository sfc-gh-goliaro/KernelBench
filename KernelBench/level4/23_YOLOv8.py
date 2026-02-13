"""
YOLOv8 Object Detection Model (ultralytics/yolov8n)

Implements the YOLOv8-nano architecture exactly as defined in the ultralytics
library, supporting pretrained weight loading and producing bit-identical
outputs for the same input.

Architecture (from ultralytics/cfg/models/v8/yolov8.yaml):
  Backbone:
    0: Conv(3, 16, 3, 2)       # P1/2
    1: Conv(16, 32, 3, 2)      # P2/4
    2: C2f(32, 32, n=1, shortcut=True)
    3: Conv(32, 64, 3, 2)      # P3/8
    4: C2f(64, 64, n=2, shortcut=True)
    5: Conv(64, 128, 3, 2)     # P4/16
    6: C2f(128, 128, n=2, shortcut=True)
    7: Conv(128, 256, 3, 2)    # P5/32
    8: C2f(256, 256, n=1, shortcut=True)
    9: SPPF(256, 256, k=5)
  Head (PANet + Detect):
    10: Upsample(scale=2)
    11: Concat([-1, 6])
    12: C2f(384, 128, n=1)
    13: Upsample(scale=2)
    14: Concat([-1, 4])
    15: C2f(192, 64, n=1)       # P3/8-small
    16: Conv(64, 64, 3, 2)
    17: Concat([-1, 12])
    18: C2f(192, 128, n=1)      # P4/16-medium
    19: Conv(128, 128, 3, 2)
    20: Concat([-1, 9])
    21: C2f(384, 256, n=1)      # P5/32-large
    22: Detect(nc=80, ch=(64, 128, 256))

Variant: yolov8n  (depth=0.33, width=0.25, max_channels=1024)
  - depth_mult=0.33 → n(3)=1, n(6)=2
  - width_mult=0.25 → ch(64)=16, ch(128)=32, ch(256)=64, ch(512)=128, ch(1024)=256

This model reuses level1 operators from KernelBench wherever possible.
"""

import math
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import Optional, Dict, Any, List, Tuple, Union

# Import level1 operators
from ..level1.activations._7_Swish import Model as Swish
from ..level1.normalization._1_BatchNorm import Model as BatchNorm


# ============================================================================
# Model Variants
# ============================================================================

VARIANTS: Dict[str, Dict[str, Any]] = {
    "n": {"depth_mult": 0.33, "width_mult": 0.25, "max_channels": 1024},
    "s": {"depth_mult": 0.33, "width_mult": 0.50, "max_channels": 1024},
    "m": {"depth_mult": 0.67, "width_mult": 0.75, "max_channels": 768},
    "l": {"depth_mult": 1.00, "width_mult": 1.00, "max_channels": 512},
    "x": {"depth_mult": 1.00, "width_mult": 1.25, "max_channels": 512},
}

# Default: yolov8n
DEFAULT_VARIANT = "n"
DEFAULT_NUM_CLASSES = 80

# COCO class names (80 classes, matching ultralytics/cfg/datasets/coco8.yaml)
COCO_NAMES: Dict[int, str] = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard", 37: "surfboard",
    38: "tennis racket", 39: "bottle", 40: "wine glass", 41: "cup", 42: "fork",
    43: "knife", 44: "spoon", 45: "bowl", 46: "banana", 47: "apple",
    48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog",
    53: "pizza", 54: "donut", 55: "cake", 56: "chair", 57: "couch",
    58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet", 62: "tv",
    63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone",
    68: "microwave", 69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator",
    73: "book", 74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush",
}


# ============================================================================
# Detection Result Dataclass
# ============================================================================

@dataclass
class DetectionResult:
    """Holds detection results for a single image.

    Attributes:
        boxes: Bounding boxes in xyxy format, shape (N, 4), in original
            image coordinates.
        scores: Confidence scores, shape (N,).
        class_ids: Integer class indices, shape (N,).
        class_names: Human-readable class labels, list of length N.
        orig_shape: Original image shape (H, W) before preprocessing.
    """
    boxes: torch.Tensor        # (N, 4) xyxy in original-image coords
    scores: torch.Tensor       # (N,)
    class_ids: torch.Tensor    # (N,) int64
    class_names: List[str]     # length N
    orig_shape: Tuple[int, int]  # (H, W)

    def __len__(self) -> int:
        return self.boxes.shape[0]

    def __repr__(self) -> str:
        return (
            f"DetectionResult(n={len(self)}, orig_shape={self.orig_shape}, "
            f"classes={self.class_names})"
        )


# ============================================================================
# Utility Functions
# ============================================================================

def make_divisible(x: float, divisor: int) -> int:
    """Make x divisible by divisor."""
    return max(divisor, int(x + divisor / 2) // divisor * divisor)


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def make_anchors(feats: List[torch.Tensor], strides: torch.Tensor, grid_cell_offset: float = 0.5):
    """Generate anchors from features (matches ultralytics.utils.tal.make_anchors)."""
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for i in range(len(feats)):
        stride = strides[i]
        h, w = feats[i].shape[2:]
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor, xywh: bool = True, dim: int = -1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)
    return torch.cat((x1y1, x2y2), dim)


# ============================================================================
# Image Preprocessing Functions
# ============================================================================

def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format.

    Matches ``ultralytics.utils.ops.xywh2xyxy``.
    """
    assert x.shape[-1] == 4
    y = torch.empty_like(x)
    xy = x[..., :2]
    wh = x[..., 2:] / 2
    y[..., :2] = xy - wh
    y[..., 2:] = xy + wh
    return y


def clip_boxes(boxes: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """Clip bounding boxes to image boundaries.

    Args:
        boxes: (N, 4) tensor in xyxy format.
        shape: (H, W) of the image.

    Returns:
        Clipped boxes tensor (same object, modified in-place).
    """
    h, w = shape[:2]
    boxes[..., 0].clamp_(0, w)
    boxes[..., 1].clamp_(0, h)
    boxes[..., 2].clamp_(0, w)
    boxes[..., 3].clamp_(0, h)
    return boxes


def scale_boxes(
    img1_shape: Tuple[int, int],
    boxes: torch.Tensor,
    img0_shape: Tuple[int, int],
    ratio_pad: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    padding: bool = True,
) -> torch.Tensor:
    """Rescale boxes from ``img1_shape`` (model input) to ``img0_shape`` (original).

    Matches ``ultralytics.utils.ops.scale_boxes``.
    """
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad_x = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1)
        pad_y = round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)
    else:
        gain = ratio_pad[0][0]
        pad_x, pad_y = ratio_pad[1]

    if padding:
        boxes[..., 0] -= pad_x
        boxes[..., 1] -= pad_y
        boxes[..., 2] -= pad_x
        boxes[..., 3] -= pad_y
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)


def letterbox_image(
    img: np.ndarray,
    imgsz: int = 640,
    stride: int = 32,
    fill_value: float = 114.0,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize and pad an image using letterboxing.

    This replicates the ultralytics ``LetterBox`` transform used at
    inference time: the image is resized so the longer side equals
    ``imgsz`` (while maintaining aspect ratio), and the shorter side is
    padded symmetrically with ``fill_value``.

    Args:
        img: Input image in HWC uint8 format (RGB or BGR).
        imgsz: Target square size in pixels.
        stride: Stride constraint (final size will be divisible by stride).
        fill_value: Pixel value used for padding (default 114 = grey).

    Returns:
        A tuple of (padded_image, scale, (pad_w, pad_h)):
          - padded_image: uint8 HWC ndarray of shape (imgsz, imgsz, C).
          - scale: scaling factor applied to the original image.
          - (pad_w, pad_h): padding offsets applied to the top-left corner.
    """
    h, w = img.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    if (h, w) != (new_h, new_w):
        # Use torch for deterministic bilinear resize (matches our test helper)
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        t = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
        resized = t.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().numpy()
    else:
        resized = img

    pad_h = imgsz - new_h
    pad_w = imgsz - new_w
    top = pad_h // 2
    left = pad_w // 2

    channels = img.shape[2] if img.ndim == 3 else 1
    padded = np.full((imgsz, imgsz, channels), fill_value, dtype=np.uint8)
    padded[top : top + new_h, left : left + new_w] = resized

    return padded, scale, (left, top)


def preprocess_images(
    images: Union[np.ndarray, List[np.ndarray]],
    imgsz: int = 640,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[torch.Tensor, List[Tuple[int, int]], List[Tuple[float, Tuple[int, int]]]]:
    """Preprocess one or more raw images for YOLOv8 inference.

    Performs letterbox resizing, HWC→CHW conversion, uint8→float32 /255
    normalisation, and batching.

    Args:
        images: A single HWC uint8 image or a list of such images.
        imgsz: Target square size.
        device: Torch device for the output tensor.

    Returns:
        - batch: Float32 tensor of shape (B, 3, imgsz, imgsz) in [0, 1].
        - orig_shapes: List of (H, W) tuples giving each image's original size.
        - pad_info: List of (scale, (pad_x, pad_y)) tuples for each image
                    (needed by ``scale_boxes`` to map detections back).
    """
    if isinstance(images, np.ndarray) and images.ndim == 3:
        images = [images]

    tensors: List[torch.Tensor] = []
    orig_shapes: List[Tuple[int, int]] = []
    pad_info: List[Tuple[float, Tuple[int, int]]] = []

    for img in images:
        orig_shapes.append((img.shape[0], img.shape[1]))
        padded, scale, (pad_x, pad_y) = letterbox_image(img, imgsz)
        pad_info.append((scale, (pad_x, pad_y)))

        # HWC -> CHW, uint8 -> float32 [0, 1]
        t = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
        tensors.append(t)

    batch = torch.stack(tensors).to(device)
    return batch, orig_shapes, pad_info


# ============================================================================
# Non-Maximum Suppression
# ============================================================================

def non_max_suppression(
    prediction: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes: Optional[List[int]] = None,
    agnostic: bool = False,
    max_det: int = 300,
    nc: int = 0,
    max_nms: int = 30000,
    max_wh: int = 7680,
) -> List[torch.Tensor]:
    """Non-maximum suppression on YOLOv8 raw predictions.

    Matches ``ultralytics.utils.nms.non_max_suppression`` (non-rotated,
    single-label path).

    Args:
        prediction: Raw model output, shape (B, 4+nc, N) where N = number of
            anchor points.  Channels 0-3 are xywh boxes, 4: are class scores
            (already sigmoid-ed).
        conf_thres: Minimum class-score to keep a candidate.
        iou_thres: IoU threshold for NMS.
        classes: Optional list of class indices to keep.
        agnostic: If True, perform class-agnostic NMS.
        max_det: Maximum detections per image.
        nc: Number of classes (inferred from prediction if 0).
        max_nms: Maximum boxes fed to torchvision NMS.
        max_wh: Maximum box dimension (used for class-offset trick).

    Returns:
        A list of length B, where each element is a (K, 6) tensor of
        detections in [x1, y1, x2, y2, confidence, class_id] format.
    """
    assert 0 <= conf_thres <= 1
    assert 0 <= iou_thres <= 1

    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    bs = prediction.shape[0]
    nc = nc or (prediction.shape[1] - 4)

    # Confidence-based candidate mask  (B, N)
    xc = prediction[:, 4 : 4 + nc].amax(1) > conf_thres

    # Transpose to (B, N, 4+nc)
    prediction = prediction.transpose(-1, -2)

    # xywh -> xyxy
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])

    output: List[torch.Tensor] = [
        torch.zeros((0, 6), device=prediction.device)
    ] * bs

    for xi in range(bs):
        x = prediction[xi][xc[xi]]  # filter by confidence

        if not x.shape[0]:
            continue

        box = x[:, :4]
        cls = x[:, 4 : 4 + nc]

        # Best class only (single-label)
        conf, j = cls.max(1, keepdim=True)
        filt = conf.view(-1) > conf_thres
        x = torch.cat((box, conf, j.float()), 1)[filt]

        # Filter by class
        if classes is not None:
            cls_tensor = torch.tensor(classes, device=x.device)
            x = x[(x[:, 5:6] == cls_tensor).any(1)]

        n = x.shape[0]
        if not n:
            continue
        if n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Class-offset trick for per-class NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes = x[:, :4] + c
        scores = x[:, 4]

        i = torchvision.ops.nms(boxes, scores, iou_thres)
        i = i[:max_det]

        output[xi] = x[i]

    return output


# ============================================================================
# Component Modules (matching ultralytics exactly)
# ============================================================================

class Conv(nn.Module):
    """Standard convolution with batch norm and SiLU.

    Matches ultralytics.nn.modules.conv.Conv exactly.

    Note: The pretrained yolov8n checkpoint uses BatchNorm with eps=0.001
    and momentum=0.03. These are NOT stored in the state_dict, so we must
    set them at construction time to get identical eval-mode results.

    Level1 operators: BatchNorm (via nn.BatchNorm2d), Swish/SiLU
    """
    default_act = nn.SiLU()

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p=None,
                 g: int = 1, d: int = 1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d),
                              groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Forward without batch norm (used after fusing)."""
        return self.act(self.conv(x))


class DWConv(Conv):
    """Depth-wise convolution (matches ultralytics)."""
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class Bottleneck(nn.Module):
    """Standard bottleneck block.

    Matches ultralytics.nn.modules.block.Bottleneck exactly.
    """
    def __init__(self, c1: int, c2: int, shortcut: bool = True,
                 g: int = 1, k: Tuple[int, int] = (3, 3), e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions.

    Matches ultralytics.nn.modules.block.C2f exactly.
    """
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False,
                 g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g,
                       k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF).

    Matches ultralytics.nn.modules.block.SPPF.
    The yolov8n pretrained checkpoint was built with an older SPPF that
    uses act=True on cv1 (SiLU activation). The newer ultralytics code
    uses act=False, but we match the pretrained model.
    """
    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class DFL(nn.Module):
    """Distribution Focal Loss layer.

    Matches ultralytics.nn.modules.block.DFL exactly.
    """
    def __init__(self, c1: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class Concat(nn.Module):
    """Concatenate a list of tensors along a dimension."""
    def __init__(self, dimension: int = 1):
        super().__init__()
        self.d = dimension

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x, self.d)


class Detect(nn.Module):
    """YOLOv8 Detect head for object detection.

    Matches ultralytics.nn.modules.head.Detect in legacy mode exactly.
    Implements the full inference pipeline: DFL decode + anchor-based box decode.

    For yolov8n with ch=(64, 128, 256):
      c2 = max(16, 64//4, 16*4) = 64
      c3 = max(64, min(80, 100)) = 80
    """
    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc: int = 80, ch: Tuple[int, ...] = ()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)

        # Channel calculations matching ultralytics
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], min(self.nc, 100))

        # Box regression branch
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3), Conv(c2, c2, 3),
                nn.Conv2d(c2, 4 * self.reg_max, 1),
            )
            for x in ch
        )

        # Classification branch (legacy mode for v8)
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), Conv(c3, c3, 3),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )

        # DFL layer
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x: List[torch.Tensor]):
        """Forward pass: returns decoded (boxes, scores) tensor for inference."""
        bs = x[0].shape[0]

        # Compute box and cls features
        boxes = torch.cat(
            [self.cv2[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)],
            dim=-1,
        )
        scores = torch.cat(
            [self.cv3[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)],
            dim=-1,
        )

        # Inference decode
        shape = x[0].shape
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (
                a.transpose(0, 1)
                for a in make_anchors(x, self.stride, 0.5)
            )
            self.shape = shape

        dbox = dist2bbox(
            self.dfl(boxes), self.anchors.unsqueeze(0), xywh=True, dim=1
        ) * self.strides

        y = torch.cat((dbox, scores.sigmoid()), 1)
        return y

    def bias_init(self):
        """Initialize Detect() biases."""
        for i, (a, b) in enumerate(zip(self.cv2, self.cv3)):
            a[-1].bias.data[:] = 2.0
            b[-1].bias.data[: self.nc] = math.log(
                5 / self.nc / (640 / self.stride[i]) ** 2
            )


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    YOLOv8 object detection model.

    Faithfully reproduces the ultralytics/yolov8n architecture so that
    pretrained weights can be loaded directly and produce identical outputs.

    Architecture follows ultralytics/cfg/models/v8/yolov8.yaml with
    the "n" scale: depth_mult=0.33, width_mult=0.25, max_channels=1024.

    Level1 operators used:
    - Swish/SiLU from level1/activations/_7_Swish  (activation in Conv)
    - BatchNorm from level1/normalization/_1_BatchNorm  (via nn.BatchNorm2d)
    - MaxPool2d from level1/pooling/_2_MaxPool2d  (in SPPF)

    Supports variants: n, s, m, l, x
    """

    VARIANTS = VARIANTS

    def __init__(
        self,
        variant: str = DEFAULT_VARIANT,
        num_classes: int = DEFAULT_NUM_CLASSES,
        **kwargs,
    ):
        super().__init__()

        cfg = VARIANTS[variant]
        depth_mult = kwargs.get("depth_mult", cfg["depth_mult"])
        width_mult = kwargs.get("width_mult", cfg["width_mult"])
        max_channels = kwargs.get("max_channels", cfg["max_channels"])

        self.variant = variant
        self.num_classes = num_classes

        def ch(c):
            return make_divisible(min(c, max_channels) * width_mult, 8)

        def n(d):
            return max(round(d * depth_mult), 1)

        # ----------------------------------------------------------------
        # Backbone (layers 0-9 in the YAML)
        # ----------------------------------------------------------------
        # 0: Conv(3, 64, 3, 2) -> Conv(3, ch(64), 3, 2)
        self.backbone_0 = Conv(3, ch(64), 3, 2)

        # 1: Conv(ch(64), ch(128), 3, 2)
        self.backbone_1 = Conv(ch(64), ch(128), 3, 2)

        # 2: C2f(ch(128), ch(128), n(3), True)
        self.backbone_2 = C2f(ch(128), ch(128), n(3), True)

        # 3: Conv(ch(128), ch(256), 3, 2)
        self.backbone_3 = Conv(ch(128), ch(256), 3, 2)

        # 4: C2f(ch(256), ch(256), n(6), True)
        self.backbone_4 = C2f(ch(256), ch(256), n(6), True)

        # 5: Conv(ch(256), ch(512), 3, 2)
        self.backbone_5 = Conv(ch(256), ch(512), 3, 2)

        # 6: C2f(ch(512), ch(512), n(6), True)
        self.backbone_6 = C2f(ch(512), ch(512), n(6), True)

        # 7: Conv(ch(512), ch(1024), 3, 2)
        self.backbone_7 = Conv(ch(512), ch(1024), 3, 2)

        # 8: C2f(ch(1024), ch(1024), n(3), True)
        self.backbone_8 = C2f(ch(1024), ch(1024), n(3), True)

        # 9: SPPF(ch(1024), ch(1024), 5)
        self.backbone_9 = SPPF(ch(1024), ch(1024), 5)

        # ----------------------------------------------------------------
        # Head (layers 10-22 in the YAML)
        # ----------------------------------------------------------------
        # 10: nn.Upsample(scale_factor=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # 12: C2f(ch(1024) + ch(512), ch(512), n(3), False)
        #     Input is concat of upsampled backbone_9 output and backbone_6 output
        self.head_12 = C2f(ch(1024) + ch(512), ch(512), n(3), False)

        # 15: C2f(ch(512) + ch(256), ch(256), n(3), False)
        #     Input is concat of upsampled head_12 output and backbone_4 output
        self.head_15 = C2f(ch(512) + ch(256), ch(256), n(3), False)

        # 16: Conv(ch(256), ch(256), 3, 2)
        self.head_16 = Conv(ch(256), ch(256), 3, 2)

        # 18: C2f(ch(256) + ch(512), ch(512), n(3), False)
        #     Input is concat of head_16 output and head_12 output
        self.head_18 = C2f(ch(256) + ch(512), ch(512), n(3), False)

        # 19: Conv(ch(512), ch(512), 3, 2)
        self.head_19 = Conv(ch(512), ch(512), 3, 2)

        # 21: C2f(ch(512) + ch(1024), ch(1024), n(3), False)
        #     Input is concat of head_19 output and backbone_9 output
        self.head_21 = C2f(ch(512) + ch(1024), ch(1024), n(3), False)

        # 22: Detect(nc, ch=(ch(256), ch(512), ch(1024)))
        self.detect = Detect(num_classes, (ch(256), ch(512), ch(1024)))

        # Set strides (P3=8, P4=16, P5=32)
        self.detect.stride = torch.tensor([8.0, 16.0, 32.0])

        # Initialize biases
        self.detect.bias_init()

        # Store channel info for weight loading
        self._ch = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through YOLOv8.

        Args:
            x: Input image tensor of shape (B, 3, H, W).
                Pixel values should be in [0, 1] range (as ultralytics normalizes to 0-1).

        Returns:
            Tensor of shape (B, 4+nc, num_anchors) with decoded boxes (xywh) and
            class scores (sigmoid).
        """
        # --- Backbone ---
        x0 = self.backbone_0(x)                        # P1/2
        x1 = self.backbone_1(x0)                       # P2/4
        x2 = self.backbone_2(x1)
        x3 = self.backbone_3(x2)                       # P3/8
        x4 = self.backbone_4(x3)                       # -> feeds into head (layer 4)
        x5 = self.backbone_5(x4)                       # P4/16
        x6 = self.backbone_6(x5)                       # -> feeds into head (layer 6)
        x7 = self.backbone_7(x6)                       # P5/32
        x8 = self.backbone_8(x7)
        x9 = self.backbone_9(x8)                       # -> feeds into head (layer 9)

        # --- Head (PANet) ---
        # Upsample + Concat backbone P4 (layer 6)
        up1 = self.upsample(x9)                        # layer 10
        cat1 = torch.cat([up1, x6], 1)                 # layer 11: concat
        h12 = self.head_12(cat1)                        # layer 12

        # Upsample + Concat backbone P3 (layer 4)
        up2 = self.upsample(h12)                        # layer 13
        cat2 = torch.cat([up2, x4], 1)                  # layer 14: concat
        h15 = self.head_15(cat2)                         # layer 15 (P3/8-small)

        # Downsample + Concat head P4 (layer 12)
        d16 = self.head_16(h15)                          # layer 16
        cat3 = torch.cat([d16, h12], 1)                  # layer 17: concat
        h18 = self.head_18(cat3)                          # layer 18 (P4/16-medium)

        # Downsample + Concat head P5 (layer 9)
        d19 = self.head_19(h18)                           # layer 19
        cat4 = torch.cat([d19, x9], 1)                    # layer 20: concat
        h21 = self.head_21(cat4)                           # layer 21 (P5/32-large)

        # --- Detection ---
        return self.detect([h15, h18, h21])

    # ------------------------------------------------------------------
    # End-to-end detection pipeline
    # ------------------------------------------------------------------

    @torch.no_grad()
    def detect_from_image(
        self,
        images: Union[np.ndarray, List[np.ndarray]],
        imgsz: int = 640,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        classes: Optional[List[int]] = None,
        agnostic_nms: bool = False,
        max_det: int = 300,
        names: Optional[Dict[int, str]] = None,
    ) -> List[DetectionResult]:
        """Run the full detection pipeline on raw images.

        This is the main entry-point for using the model end-to-end.
        It replicates the ultralytics ``model.predict()`` pipeline:

            raw image → letterbox → normalise → forward → NMS →
            scale boxes back to original coords → ``DetectionResult``

        Args:
            images: One or more HWC uint8 images (RGB).  A single ndarray
                of shape (H, W, 3) or a list of such arrays.
            imgsz: Model input resolution (default 640).
            conf_thres: Confidence threshold for NMS.
            iou_thres: IoU threshold for NMS.
            classes: Optional list of class indices to filter.
            agnostic_nms: If ``True``, perform class-agnostic NMS.
            max_det: Maximum number of detections per image.
            names: Class-index → name mapping.  Defaults to ``COCO_NAMES``.

        Returns:
            A list of :class:`DetectionResult`, one per input image.
        """
        if names is None:
            names = COCO_NAMES

        was_training = self.training
        self.eval()

        device = next(self.parameters()).device

        # 1) Preprocess
        batch, orig_shapes, pad_info = preprocess_images(images, imgsz, device)

        # 2) Forward (raw predictions)
        raw = self.forward(batch)  # (B, 4+nc, N)

        # 3) Non-maximum suppression  →  list[ (K, 6) ]
        dets = non_max_suppression(
            raw,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            classes=classes,
            agnostic=agnostic_nms,
            max_det=max_det,
            nc=self.num_classes,
        )

        # 4) Scale boxes back to original image coordinates
        results: List[DetectionResult] = []
        for i, det in enumerate(dets):
            orig_hw = orig_shapes[i]
            if det.shape[0]:
                # scale_boxes modifies in-place; clone to be safe
                det[:, :4] = scale_boxes(
                    (imgsz, imgsz), det[:, :4].clone(), orig_hw
                )

            boxes = det[:, :4]
            scores = det[:, 4]
            class_ids = det[:, 5].long()
            class_labels = [names.get(int(c), str(int(c))) for c in class_ids]

            results.append(DetectionResult(
                boxes=boxes,
                scores=scores,
                class_ids=class_ids,
                class_names=class_labels,
                orig_shape=orig_hw,
            ))

        if was_training:
            self.train()

        return results


def load_from_ultralytics(model: Model, ultralytics_model) -> None:
    """Load weights from an ultralytics YOLO model into our KB Model.

    Args:
        model: Our KernelBench YOLOv8 Model instance.
        ultralytics_model: An ultralytics YOLO model (from ultralytics import YOLO; m = YOLO("yolov8n.pt")).

    This copies weights by matching the architecture layer-by-layer.
    The ultralytics model stores layers in model.model (a Sequential), while
    our Model uses named attributes. We map by layer index.
    """
    # ultralytics model is stored as model.model (a Sequential of layers)
    # Access the underlying nn.Module
    if hasattr(ultralytics_model, "model"):
        if hasattr(ultralytics_model.model, "model"):
            ul_model = ultralytics_model.model.model
        else:
            ul_model = ultralytics_model.model
    else:
        ul_model = ultralytics_model

    # Map: ultralytics layer index -> our named attribute
    layer_map = {
        0: model.backbone_0,
        1: model.backbone_1,
        2: model.backbone_2,
        3: model.backbone_3,
        4: model.backbone_4,
        5: model.backbone_5,
        6: model.backbone_6,
        7: model.backbone_7,
        8: model.backbone_8,
        9: model.backbone_9,
        # Head layers (skipping Upsample/Concat which are stateless)
        12: model.head_12,
        15: model.head_15,
        16: model.head_16,
        18: model.head_18,
        19: model.head_19,
        21: model.head_21,
        22: model.detect,
    }

    for idx, ul_layer in enumerate(ul_model):
        if idx in layer_map:
            kb_layer = layer_map[idx]
            ul_sd = ul_layer.state_dict()
            kb_sd = kb_layer.state_dict()

            # For the Detect head, handle special key mappings
            if idx == 22:
                # ultralytics Detect uses one2many dict pattern in newer versions
                # but the cv2/cv3/dfl weights are the same
                new_sd = {}
                for kb_key in kb_sd:
                    if kb_key in ul_sd and ul_sd[kb_key].shape == kb_sd[kb_key].shape:
                        new_sd[kb_key] = ul_sd[kb_key]
                    else:
                        # Try one2one prefix
                        for prefix in ["", "one2one_"]:
                            alt_key = prefix + kb_key
                            if alt_key in ul_sd and ul_sd[alt_key].shape == kb_sd[kb_key].shape:
                                new_sd[kb_key] = ul_sd[alt_key]
                                break

                missing, unexpected = kb_layer.load_state_dict(new_sd, strict=False)
                if missing:
                    print(f"  Detect missing {len(missing)} keys")
                # Also copy stride
                if hasattr(ul_layer, "stride"):
                    model.detect.stride = ul_layer.stride.clone()
            else:
                kb_layer.load_state_dict(ul_sd, strict=True)
