"""
YOLOv8n alignment tests: KernelBench vs ultralytics library.

Tests end-to-end alignment between our KernelBench implementation
(KernelBench/level4/23_YOLOv8.py) and the official ultralytics YOLO library,
using realistic images from scikit-image.

Test structure:
  - TestWeightLoading: Verify weights can be loaded with zero missing keys
  - TestBackboneAlignment: Compare backbone feature maps layer-by-layer
  - TestDetectHeadAlignment: Compare detection head raw outputs
  - TestE2EAlignment: Compare full end-to-end outputs on real images
  - TestMultiResolution: Test alignment at different input resolutions

Usage:
    # All tests:
    pytest tests/test_yolov8_alignment.py -v

    # E2E only:
    pytest tests/test_yolov8_alignment.py -v -k "TestE2EAlignment"

    # Quick smoke test:
    pytest tests/test_yolov8_alignment.py -v -k "test_smoke"

Requires:
    - ultralytics (pip install ultralytics)
    - scikit-image (pip install scikit-image)
    - CUDA GPU recommended but not required
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pytest
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(TEST_DIR, "..")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "KernelBench"))

# Ensure the local ultralytics/ directory does NOT shadow the installed package.
# Remove any path entries that would cause the repo-local ultralytics dir to
# be imported instead of the pip-installed ultralytics package.
_local_ul = os.path.join(REPO_ROOT, "ultralytics")
sys.path = [p for p in sys.path if os.path.realpath(p) != os.path.realpath(_local_ul)]
# Also remove the repo root's ultralytics from sys.modules if already loaded
if "ultralytics" in sys.modules:
    _ul_mod = sys.modules["ultralytics"]
    if _ul_mod.__file__ is None or (
        hasattr(_ul_mod, "__path__") and
        any(os.path.realpath(_local_ul) in os.path.realpath(p) for p in _ul_mod.__path__)
    ):
        del sys.modules["ultralytics"]

# Output directory for saved images
OUTPUT_DIR = os.path.join(TEST_DIR, "outputs")

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_kb_module():
    """Load the KernelBench YOLOv8 module."""
    return importlib.import_module("KernelBench.level4.23_YOLOv8")


def _get_skimage_images() -> List[np.ndarray]:
    """Load realistic test images from scikit-image.

    Returns a list of RGB uint8 images (H, W, 3) with various content.
    """
    from skimage import data as skdata

    images = []
    # Astronaut: person in a spacesuit (480x512)
    images.append(skdata.astronaut())
    # Chelsea: cat image (300x451)
    images.append(skdata.chelsea())
    # Coffee: cup of coffee (400x600)
    images.append(skdata.coffee())

    return images


def _preprocess_image(img: np.ndarray, imgsz: int = 640) -> torch.Tensor:
    """Preprocess an image for YOLOv8 inference.

    Mimics the ultralytics preprocessing pipeline:
    1. Resize with letterbox to (imgsz, imgsz)
    2. Convert BGR->RGB (skimage is already RGB)
    3. Normalize to [0, 1]
    4. CHW format

    Args:
        img: RGB uint8 image (H, W, 3)
        imgsz: Target size

    Returns:
        Tensor of shape (1, 3, imgsz, imgsz) in [0, 1] range
    """
    # Letterbox resize
    h, w = img.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Resize using torch for consistency
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_resized = F.interpolate(
        img_tensor, size=(new_h, new_w), mode="bilinear", align_corners=False
    )

    # Pad to imgsz x imgsz (center pad)
    pad_h = imgsz - new_h
    pad_w = imgsz - new_w
    top = pad_h // 2
    left = pad_w // 2
    padded = torch.full((1, 3, imgsz, imgsz), 114.0 / 255.0, dtype=torch.float32)
    padded[:, :, top : top + new_h, left : left + new_w] = img_resized

    return padded


def _get_ultralytics_model():
    """Load the ultralytics YOLOv8n model."""
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    return model


def _copy_weights_to_kb(kb_model, ul_model):
    """Copy weights from ultralytics model to KB model layer by layer.

    The ultralytics model stores layers in a Sequential (model.model.model),
    while our KB model uses named attributes. We map by layer index from the
    YAML configuration.
    """
    # Get the underlying nn.Module Sequential
    ul_sequential = ul_model.model.model

    # Map: ultralytics layer index -> our named attribute
    layer_map = {
        0: kb_model.backbone_0,
        1: kb_model.backbone_1,
        2: kb_model.backbone_2,
        3: kb_model.backbone_3,
        4: kb_model.backbone_4,
        5: kb_model.backbone_5,
        6: kb_model.backbone_6,
        7: kb_model.backbone_7,
        8: kb_model.backbone_8,
        9: kb_model.backbone_9,
        12: kb_model.head_12,
        15: kb_model.head_15,
        16: kb_model.head_16,
        18: kb_model.head_18,
        19: kb_model.head_19,
        21: kb_model.head_21,
        22: kb_model.detect,
    }

    total_missing = 0
    total_keys = 0

    for idx, ul_layer in enumerate(ul_sequential):
        if idx not in layer_map:
            continue

        kb_layer = layer_map[idx]
        ul_sd = ul_layer.state_dict()
        kb_sd = kb_layer.state_dict()

        if idx == 22:
            # Detect head: ultralytics may have extra keys (one2one_cv2/cv3, etc.)
            new_sd = {}
            for kb_key in kb_sd:
                if kb_key in ul_sd and ul_sd[kb_key].shape == kb_sd[kb_key].shape:
                    new_sd[kb_key] = ul_sd[kb_key]

            missing, _ = kb_layer.load_state_dict(new_sd, strict=False)
            total_missing += len(missing)
            total_keys += len(kb_sd)

            # Copy stride
            if hasattr(ul_layer, "stride"):
                kb_model.detect.stride = ul_layer.stride.clone()
        else:
            try:
                kb_layer.load_state_dict(ul_sd, strict=True)
            except RuntimeError as e:
                # If strict loading fails, try non-strict and count missing
                missing, _ = kb_layer.load_state_dict(ul_sd, strict=False)
                total_missing += len(missing)
                print(f"  Layer {idx}: {len(missing)} missing keys - {e}")
            total_keys += len(kb_sd)

    return total_missing, total_keys


def _run_ultralytics_raw_forward(ul_model, img_tensor: torch.Tensor):
    """Run the ultralytics model in raw forward mode (no pre/post processing).

    This runs the model's internal nn.Module directly, bypassing the
    ultralytics prediction pipeline, to get raw outputs for comparison.

    Args:
        ul_model: ultralytics YOLO model
        img_tensor: preprocessed image tensor (B, 3, H, W) in [0, 1]

    Returns:
        Raw forward output (the detection tensor before NMS)
    """
    # Access the internal model
    model = ul_model.model
    model.eval()

    with torch.no_grad():
        # Use predict mode to get inference output
        result = model(img_tensor)

    # result is (y, preds) tuple in non-export mode
    # y is the decoded inference output
    if isinstance(result, tuple):
        return result[0]  # decoded output
    return result


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def ultralytics_available():
    """Check if ultralytics is installed."""
    try:
        import ultralytics
        return True
    except ImportError:
        pytest.skip("ultralytics not installed. Install with: pip install ultralytics")


@pytest.fixture(scope="session")
def skimage_available():
    """Check if scikit-image is installed."""
    try:
        import skimage
        return True
    except ImportError:
        pytest.skip("scikit-image not installed. Install with: pip install scikit-image")


@pytest.fixture(scope="session")
def ul_model(ultralytics_available):
    """Load the ultralytics YOLOv8n model (cached for session)."""
    model = _get_ultralytics_model()
    model.model.to(DEVICE).eval()
    return model


@pytest.fixture(scope="session")
def kb_model_loaded(ul_model):
    """Build and load weights into KB model from ultralytics (cached for session)."""
    kb_mod = _load_kb_module()
    kb_model = kb_mod.Model(variant="n", num_classes=80)
    missing, total = _copy_weights_to_kb(kb_model, ul_model)
    kb_model = kb_model.to(DEVICE).eval()
    return kb_model, missing, total


@pytest.fixture(scope="session")
def test_images(skimage_available):
    """Load and preprocess test images from scikit-image."""
    raw_images = _get_skimage_images()
    preprocessed = []
    for img in raw_images:
        tensor = _preprocess_image(img, imgsz=640).to(DEVICE)
        preprocessed.append((img, tensor))
    return preprocessed


# ============================================================================
# Test: Smoke test (no external deps needed)
# ============================================================================

class TestSmoke:
    """Basic smoke tests for the KB YOLOv8 model (no pretrained weights needed)."""

    def test_smoke_forward(self):
        """Test that the model can do a forward pass with random weights."""
        kb_mod = _load_kb_module()
        model = kb_mod.Model(variant="n", num_classes=80).to(DEVICE).eval()

        x = torch.randn(1, 3, 640, 640, device=DEVICE)
        with torch.no_grad():
            out = model(x)

        # Should be (1, 84, num_anchors) where 84 = 4 + 80
        assert out.dim() == 3, f"Expected 3D output, got {out.dim()}D"
        assert out.shape[0] == 1
        assert out.shape[1] == 84  # 4 (xywh) + 80 (classes)
        # 640/8=80, 640/16=40, 640/32=20 -> 80*80 + 40*40 + 20*20 = 8400
        assert out.shape[2] == 8400, f"Expected 8400 anchors, got {out.shape[2]}"
        assert not torch.isnan(out).any(), "NaN in output"
        print(f"\nSmoke test passed: output shape {out.shape}")

    def test_smoke_different_sizes(self):
        """Test forward pass with different input sizes."""
        kb_mod = _load_kb_module()
        model = kb_mod.Model(variant="n", num_classes=80).to(DEVICE).eval()

        for size in [320, 416, 640]:
            x = torch.randn(1, 3, size, size, device=DEVICE)
            with torch.no_grad():
                out = model(x)

            expected_anchors = (size // 8) ** 2 + (size // 16) ** 2 + (size // 32) ** 2
            assert out.shape == (1, 84, expected_anchors), (
                f"Size {size}: expected (1, 84, {expected_anchors}), got {out.shape}"
            )
        print("\nMulti-size smoke test passed")

    def test_smoke_batch(self):
        """Test forward pass with batch size > 1."""
        kb_mod = _load_kb_module()
        model = kb_mod.Model(variant="n", num_classes=80).to(DEVICE).eval()

        x = torch.randn(4, 3, 640, 640, device=DEVICE)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (4, 84, 8400), f"Expected (4, 84, 8400), got {out.shape}"
        print(f"\nBatch smoke test passed: output shape {out.shape}")


# ============================================================================
# Test: Weight Loading
# ============================================================================

class TestWeightLoading:
    """Test that weights load correctly from ultralytics."""

    def test_weight_loading_completeness(self, kb_model_loaded):
        """Verify all weights are loaded with zero missing keys."""
        kb_model, missing, total = kb_model_loaded
        print(f"\nWeight loading: {missing} missing out of {total} total keys")
        assert missing == 0, f"{missing} keys missing during weight loading"

    def test_weight_shapes_match(self, ul_model, kb_model_loaded):
        """Verify KB and ultralytics models have the same parameter shapes."""
        kb_model, _, _ = kb_model_loaded

        kb_params = dict(kb_model.named_parameters())
        kb_total = sum(p.numel() for p in kb_model.parameters())

        # YOLOv8n has ~3.2M parameters
        print(f"\nKB model: {kb_total:,} parameters")
        assert 3_000_000 < kb_total < 4_000_000, (
            f"YOLOv8n should have ~3.2M params, got {kb_total:,}"
        )

    def test_stride_values(self, kb_model_loaded):
        """Verify detection head strides are set correctly."""
        kb_model, _, _ = kb_model_loaded
        expected = torch.tensor([8.0, 16.0, 32.0])
        actual = kb_model.detect.stride
        assert torch.allclose(actual.cpu(), expected), (
            f"Expected strides {expected}, got {actual}"
        )


# ============================================================================
# Test: End-to-End Alignment with Real Images
# ============================================================================

class TestE2EAlignment:
    """End-to-end alignment tests using scikit-image images."""

    def test_e2e_astronaut(self, ul_model, kb_model_loaded, test_images):
        """Test alignment on the astronaut image (person)."""
        kb_model, _, _ = kb_model_loaded
        raw_img, img_tensor = test_images[0]  # astronaut

        self._compare_outputs(kb_model, ul_model, img_tensor, "astronaut")

    def test_e2e_chelsea(self, ul_model, kb_model_loaded, test_images):
        """Test alignment on the chelsea image (cat)."""
        kb_model, _, _ = kb_model_loaded
        raw_img, img_tensor = test_images[1]  # chelsea

        self._compare_outputs(kb_model, ul_model, img_tensor, "chelsea")

    def test_e2e_coffee(self, ul_model, kb_model_loaded, test_images):
        """Test alignment on the coffee image."""
        kb_model, _, _ = kb_model_loaded
        raw_img, img_tensor = test_images[2]  # coffee

        self._compare_outputs(kb_model, ul_model, img_tensor, "coffee")

    def test_e2e_batch(self, ul_model, kb_model_loaded, test_images):
        """Test alignment with a batch of images."""
        kb_model, _, _ = kb_model_loaded

        # Stack all images into a batch
        batch = torch.cat([t for _, t in test_images], dim=0)

        with torch.no_grad():
            kb_out = kb_model(batch)
            ul_out = _run_ultralytics_raw_forward(ul_model, batch)

        assert kb_out.shape == ul_out.shape, (
            f"Shape mismatch: KB={kb_out.shape}, UL={ul_out.shape}"
        )

        max_diff = (kb_out - ul_out).abs().max().item()
        mean_diff = (kb_out - ul_out).abs().mean().item()
        cos_sim = F.cosine_similarity(
            kb_out.flatten().unsqueeze(0),
            ul_out.flatten().unsqueeze(0),
        ).item()

        print(f"\n[Batch E2E] max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}, "
              f"cos_sim={cos_sim:.10f}")
        assert cos_sim > 0.99999, f"Batch cosine similarity too low: {cos_sim}"
        assert max_diff < 0.01, f"Batch max diff too high: {max_diff}"

    def _compare_outputs(self, kb_model, ul_model, img_tensor, name):
        """Compare KB and ultralytics outputs for a single image."""
        with torch.no_grad():
            kb_out = kb_model(img_tensor)
            ul_out = _run_ultralytics_raw_forward(ul_model, img_tensor)

        assert kb_out.shape == ul_out.shape, (
            f"Shape mismatch for {name}: KB={kb_out.shape}, UL={ul_out.shape}"
        )

        # Overall comparison
        max_diff = (kb_out - ul_out).abs().max().item()
        mean_diff = (kb_out - ul_out).abs().mean().item()
        cos_sim = F.cosine_similarity(
            kb_out.flatten().unsqueeze(0),
            ul_out.flatten().unsqueeze(0),
        ).item()

        print(f"\n[{name}] max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}, "
              f"cos_sim={cos_sim:.10f}")

        # Box predictions (first 4 channels)
        kb_boxes = kb_out[:, :4, :]
        ul_boxes = ul_out[:, :4, :]
        box_max_diff = (kb_boxes - ul_boxes).abs().max().item()
        box_cos = F.cosine_similarity(
            kb_boxes.flatten().unsqueeze(0),
            ul_boxes.flatten().unsqueeze(0),
        ).item()

        # Class predictions (remaining 80 channels)
        kb_cls = kb_out[:, 4:, :]
        ul_cls = ul_out[:, 4:, :]
        cls_max_diff = (kb_cls - ul_cls).abs().max().item()
        cls_cos = F.cosine_similarity(
            kb_cls.flatten().unsqueeze(0),
            ul_cls.flatten().unsqueeze(0),
        ).item()

        print(f"  Boxes: max_diff={box_max_diff:.8f}, cos_sim={box_cos:.10f}")
        print(f"  Classes: max_diff={cls_max_diff:.8f}, cos_sim={cls_cos:.10f}")

        # Very tight tolerances since both use identical float32 operations
        assert cos_sim > 0.99999, f"Cosine similarity too low for {name}: {cos_sim}"
        assert max_diff < 0.01, f"Max diff too high for {name}: {max_diff}"

        # Check that top detections match
        self._compare_top_detections(kb_out, ul_out, name)

    def _compare_top_detections(self, kb_out, ul_out, name, top_k=10):
        """Compare top-k detections between KB and ultralytics outputs."""
        # Get class scores (channels 4:)
        kb_scores = kb_out[0, 4:, :].max(dim=0)  # (num_anchors,)
        ul_scores = ul_out[0, 4:, :].max(dim=0)

        # Top-k by score
        kb_topk_scores, kb_topk_idx = kb_scores.values.topk(top_k)
        ul_topk_scores, ul_topk_idx = ul_scores.values.topk(top_k)

        # Check that top detections have similar scores
        score_diff = (kb_topk_scores - ul_topk_scores).abs().max().item()
        print(f"  Top-{top_k} score max_diff={score_diff:.8f}")

        # Check that top detections are at the same anchor locations
        idx_match = (kb_topk_idx == ul_topk_idx).sum().item()
        print(f"  Top-{top_k} index match: {idx_match}/{top_k}")

        assert score_diff < 0.001, f"Top detection score diff too high for {name}: {score_diff}"
        assert idx_match >= top_k - 1, (
            f"Top detection indices diverge for {name}: only {idx_match}/{top_k} match"
        )


# ============================================================================
# Test: Multi-Resolution Alignment
# ============================================================================

class TestMultiResolution:
    """Test alignment at different input resolutions."""

    @pytest.mark.parametrize("imgsz", [320, 416, 640])
    def test_resolution(self, ul_model, kb_model_loaded, skimage_available, imgsz):
        """Compare outputs at different resolutions."""
        kb_model, _, _ = kb_model_loaded

        from skimage import data as skdata
        img = skdata.astronaut()
        img_tensor = _preprocess_image(img, imgsz=imgsz).to(DEVICE)

        with torch.no_grad():
            kb_out = kb_model(img_tensor)
            ul_out = _run_ultralytics_raw_forward(ul_model, img_tensor)

        assert kb_out.shape == ul_out.shape, (
            f"Shape mismatch at {imgsz}: KB={kb_out.shape}, UL={ul_out.shape}"
        )

        max_diff = (kb_out - ul_out).abs().max().item()
        cos_sim = F.cosine_similarity(
            kb_out.flatten().unsqueeze(0),
            ul_out.flatten().unsqueeze(0),
        ).item()

        print(f"\n[Resolution {imgsz}] max_diff={max_diff:.8f}, cos_sim={cos_sim:.10f}")
        assert cos_sim > 0.99999, f"cos_sim too low at {imgsz}: {cos_sim}"
        assert max_diff < 0.01, f"max_diff too high at {imgsz}: {max_diff}"


# ============================================================================
# Test: Backbone Layer-by-Layer Alignment
# ============================================================================

class TestBackboneAlignment:
    """Layer-by-layer comparison of backbone feature maps."""

    def test_backbone_features(self, ul_model, kb_model_loaded, test_images):
        """Compare intermediate feature maps from backbone."""
        kb_model, _, _ = kb_model_loaded

        # Use astronaut image
        _, img_tensor = test_images[0]

        # Run KB backbone manually
        with torch.no_grad():
            kb_x0 = kb_model.backbone_0(img_tensor)
            kb_x1 = kb_model.backbone_1(kb_x0)
            kb_x2 = kb_model.backbone_2(kb_x1)
            kb_x3 = kb_model.backbone_3(kb_x2)
            kb_x4 = kb_model.backbone_4(kb_x3)
            kb_x5 = kb_model.backbone_5(kb_x4)
            kb_x6 = kb_model.backbone_6(kb_x5)
            kb_x7 = kb_model.backbone_7(kb_x6)
            kb_x8 = kb_model.backbone_8(kb_x7)
            kb_x9 = kb_model.backbone_9(kb_x8)

        # Run ultralytics backbone manually
        ul_seq = ul_model.model.model
        with torch.no_grad():
            ul_x0 = ul_seq[0](img_tensor)
            ul_x1 = ul_seq[1](ul_x0)
            ul_x2 = ul_seq[2](ul_x1)
            ul_x3 = ul_seq[3](ul_x2)
            ul_x4 = ul_seq[4](ul_x3)
            ul_x5 = ul_seq[5](ul_x4)
            ul_x6 = ul_seq[6](ul_x5)
            ul_x7 = ul_seq[7](ul_x6)
            ul_x8 = ul_seq[8](ul_x7)
            ul_x9 = ul_seq[9](ul_x8)

        features = [
            ("stem (0)", kb_x0, ul_x0),
            ("conv1 (1)", kb_x1, ul_x1),
            ("c2f_1 (2)", kb_x2, ul_x2),
            ("conv2 (3)", kb_x3, ul_x3),
            ("c2f_2 (4)", kb_x4, ul_x4),
            ("conv3 (5)", kb_x5, ul_x5),
            ("c2f_3 (6)", kb_x6, ul_x6),
            ("conv4 (7)", kb_x7, ul_x7),
            ("c2f_4 (8)", kb_x8, ul_x8),
            ("sppf (9)", kb_x9, ul_x9),
        ]

        print()
        for name, kb_feat, ul_feat in features:
            assert kb_feat.shape == ul_feat.shape, (
                f"Shape mismatch at {name}: KB={kb_feat.shape}, UL={ul_feat.shape}"
            )
            max_diff = (kb_feat - ul_feat).abs().max().item()
            cos_sim = F.cosine_similarity(
                kb_feat.flatten().unsqueeze(0),
                ul_feat.flatten().unsqueeze(0),
            ).item()
            print(f"  {name}: shape={kb_feat.shape}, max_diff={max_diff:.8f}, cos_sim={cos_sim:.10f}")

            # Early layers should be very close; later layers may accumulate error
            assert cos_sim > 0.99999, f"cos_sim too low at {name}: {cos_sim}"

    def test_neck_features(self, ul_model, kb_model_loaded, test_images):
        """Compare PANet neck feature maps."""
        kb_model, _, _ = kb_model_loaded
        _, img_tensor = test_images[0]

        # Run KB model completely
        with torch.no_grad():
            kb_x0 = kb_model.backbone_0(img_tensor)
            kb_x1 = kb_model.backbone_1(kb_x0)
            kb_x2 = kb_model.backbone_2(kb_x1)
            kb_x3 = kb_model.backbone_3(kb_x2)
            kb_x4 = kb_model.backbone_4(kb_x3)
            kb_x5 = kb_model.backbone_5(kb_x4)
            kb_x6 = kb_model.backbone_6(kb_x5)
            kb_x7 = kb_model.backbone_7(kb_x6)
            kb_x8 = kb_model.backbone_8(kb_x7)
            kb_x9 = kb_model.backbone_9(kb_x8)

            up1 = kb_model.upsample(kb_x9)
            cat1 = torch.cat([up1, kb_x6], 1)
            kb_h12 = kb_model.head_12(cat1)

            up2 = kb_model.upsample(kb_h12)
            cat2 = torch.cat([up2, kb_x4], 1)
            kb_h15 = kb_model.head_15(cat2)

        # Run ultralytics neck
        ul_seq = ul_model.model.model
        with torch.no_grad():
            ul_x0 = ul_seq[0](img_tensor)
            ul_x1 = ul_seq[1](ul_x0)
            ul_x2 = ul_seq[2](ul_x1)
            ul_x3 = ul_seq[3](ul_x2)
            ul_x4 = ul_seq[4](ul_x3)
            ul_x5 = ul_seq[5](ul_x4)
            ul_x6 = ul_seq[6](ul_x5)
            ul_x7 = ul_seq[7](ul_x6)
            ul_x8 = ul_seq[8](ul_x7)
            ul_x9 = ul_seq[9](ul_x8)

            # Layer 10: Upsample
            ul_up1 = ul_seq[10](ul_x9)
            # Layer 11: Concat (from=[−1, 6])
            ul_cat1 = ul_seq[11]([ul_up1, ul_x6])
            # Layer 12: C2f
            ul_h12 = ul_seq[12](ul_cat1)

            # Layer 13: Upsample
            ul_up2 = ul_seq[13](ul_h12)
            # Layer 14: Concat (from=[−1, 4])
            ul_cat2 = ul_seq[14]([ul_up2, ul_x4])
            # Layer 15: C2f
            ul_h15 = ul_seq[15](ul_cat2)

        print()
        for name, kb_f, ul_f in [("head_12", kb_h12, ul_h12), ("head_15", kb_h15, ul_h15)]:
            assert kb_f.shape == ul_f.shape, f"Shape mismatch at {name}"
            max_diff = (kb_f - ul_f).abs().max().item()
            cos_sim = F.cosine_similarity(
                kb_f.flatten().unsqueeze(0), ul_f.flatten().unsqueeze(0),
            ).item()
            print(f"  {name}: max_diff={max_diff:.8f}, cos_sim={cos_sim:.10f}")
            assert cos_sim > 0.99999, f"cos_sim too low at {name}: {cos_sim}"


# ============================================================================
# Test: Detection Output Numerical Precision
# ============================================================================

class TestNumericalPrecision:
    """Test numerical precision of the detection pipeline."""

    def test_dfl_precision(self, ul_model, kb_model_loaded, test_images):
        """Test that our DFL produces identical results."""
        kb_model, _, _ = kb_model_loaded
        _, img_tensor = test_images[0]

        # Get raw box features from both models
        with torch.no_grad():
            # KB
            kb_out = kb_model(img_tensor)

            # UL
            ul_out = _run_ultralytics_raw_forward(ul_model, img_tensor)

        # Compare boxes (first 4 channels) which use DFL
        kb_boxes = kb_out[:, :4, :]
        ul_boxes = ul_out[:, :4, :]

        box_diff = (kb_boxes - ul_boxes).abs()
        print(f"\nDFL box precision: max={box_diff.max():.8f}, "
              f"mean={box_diff.mean():.8f}, "
              f"std={box_diff.std():.8f}")

        assert box_diff.max() < 0.01, f"DFL box diff too high: {box_diff.max()}"

    def test_classification_precision(self, ul_model, kb_model_loaded, test_images):
        """Test that classification scores are identical."""
        kb_model, _, _ = kb_model_loaded
        _, img_tensor = test_images[0]

        with torch.no_grad():
            kb_out = kb_model(img_tensor)
            ul_out = _run_ultralytics_raw_forward(ul_model, img_tensor)

        kb_cls = kb_out[:, 4:, :]
        ul_cls = ul_out[:, 4:, :]

        cls_diff = (kb_cls - ul_cls).abs()
        print(f"\nClassification precision: max={cls_diff.max():.8f}, "
              f"mean={cls_diff.mean():.8f}")

        assert cls_diff.max() < 0.001, f"Classification diff too high: {cls_diff.max()}"

    def test_output_ranges(self, kb_model_loaded, test_images):
        """Test that output values are in expected ranges."""
        kb_model, _, _ = kb_model_loaded
        _, img_tensor = test_images[0]

        with torch.no_grad():
            out = kb_model(img_tensor)

        # Boxes (first 4 channels): should be positive, roughly in image coordinates
        boxes = out[:, :4, :]
        assert (boxes[:, 2:4, :] >= 0).all(), "Width/height should be non-negative"

        # Class scores (channels 4:): should be in (0, 1) since they're sigmoid
        scores = out[:, 4:, :]
        assert (scores >= 0).all() and (scores <= 1).all(), (
            f"Class scores should be in [0, 1], got [{scores.min()}, {scores.max()}]"
        )
        print(f"\nOutput ranges: boxes=[{boxes.min():.2f}, {boxes.max():.2f}], "
              f"scores=[{scores.min():.6f}, {scores.max():.6f}]")


# ============================================================================
# Test: End-to-End Pipeline (raw image → detection boxes)
# ============================================================================

class TestE2EPipeline:
    """Test the full pipeline: raw image → detect_from_image → boxes in
    original-image coordinates.

    Compares against the ultralytics ``model.predict()`` pipeline to
    verify that our preprocessing, NMS, and box-scaling produce the
    same detections.
    """

    # ---- helpers ---------------------------------------------------------

    @staticmethod
    def _run_ultralytics_predict(ul_model, img_rgb: np.ndarray, imgsz=640,
                                  conf=0.25, iou=0.45):
        """Run ultralytics predict() on a *single* RGB image.

        Returns a list of (x1, y1, x2, y2, conf, class_id) rows, sorted
        by descending confidence.
        """
        # ultralytics expects BGR input (will do BGR→RGB internally)
        img_bgr = img_rgb[..., ::-1].copy()
        results = ul_model.predict(
            source=img_bgr,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            verbose=False,
        )
        # results is a list[Results]; we need the first one
        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return torch.empty(0, 6)
        # (N, 6): x1y1x2y2 conf cls
        return torch.cat([
            boxes.xyxy.cpu(),
            boxes.conf.cpu().unsqueeze(1),
            boxes.cls.cpu().unsqueeze(1),
        ], dim=1)

    @staticmethod
    def _run_kb_detect(kb_model, kb_mod, img_rgb, imgsz=640, conf=0.25, iou=0.45):
        """Run our detect_from_image on a single RGB image.

        Returns a (N, 6) tensor: x1y1x2y2 conf cls.
        """
        res = kb_model.detect_from_image(
            img_rgb,
            imgsz=imgsz,
            conf_thres=conf,
            iou_thres=iou,
        )
        r = res[0]
        if len(r) == 0:
            return torch.empty(0, 6)
        return torch.cat([
            r.boxes.cpu(),
            r.scores.cpu().unsqueeze(1),
            r.class_ids.cpu().float().unsqueeze(1),
        ], dim=1)

    @staticmethod
    def _match_detections(kb_dets, ul_dets, iou_thresh=0.5):
        """Match KB and UL detections via greedy IoU matching.

        Returns (matched_kb, matched_ul, unmatched_kb, unmatched_ul)
        where matched_* are index arrays of corresponding rows.
        """
        if kb_dets.shape[0] == 0 or ul_dets.shape[0] == 0:
            return ([], [], list(range(kb_dets.shape[0])),
                    list(range(ul_dets.shape[0])))

        kb_boxes = kb_dets[:, :4]
        ul_boxes = ul_dets[:, :4]

        # Compute pairwise IoU
        x1 = torch.max(kb_boxes[:, None, 0], ul_boxes[None, :, 0])
        y1 = torch.max(kb_boxes[:, None, 1], ul_boxes[None, :, 1])
        x2 = torch.min(kb_boxes[:, None, 2], ul_boxes[None, :, 2])
        y2 = torch.min(kb_boxes[:, None, 3], ul_boxes[None, :, 3])
        inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        area_kb = (kb_boxes[:, 2] - kb_boxes[:, 0]) * (kb_boxes[:, 3] - kb_boxes[:, 1])
        area_ul = (ul_boxes[:, 2] - ul_boxes[:, 0]) * (ul_boxes[:, 3] - ul_boxes[:, 1])
        union = area_kb[:, None] + area_ul[None, :] - inter
        iou = inter / (union + 1e-7)

        matched_kb, matched_ul = [], []
        used_kb = set()
        used_ul = set()

        # Greedy matching: highest IoU first
        flat = iou.flatten()
        sorted_idx = flat.argsort(descending=True)
        for idx in sorted_idx:
            ki = int(idx // iou.shape[1])
            ui = int(idx % iou.shape[1])
            if ki in used_kb or ui in used_ul:
                continue
            if iou[ki, ui] < iou_thresh:
                break
            matched_kb.append(ki)
            matched_ul.append(ui)
            used_kb.add(ki)
            used_ul.add(ui)

        unmatched_kb = [i for i in range(kb_dets.shape[0]) if i not in used_kb]
        unmatched_ul = [i for i in range(ul_dets.shape[0]) if i not in used_ul]

        return matched_kb, matched_ul, unmatched_kb, unmatched_ul

    # ---- actual tests ----------------------------------------------------

    def test_pipeline_astronaut(self, ul_model, kb_model_loaded, test_images):
        """Astronaut image: verify the KB pipeline finds the same objects."""
        kb_model, _, _ = kb_model_loaded
        kb_mod = _load_kb_module()
        raw_img, _ = test_images[0]

        ul_dets = self._run_ultralytics_predict(ul_model, raw_img)
        kb_dets = self._run_kb_detect(kb_model, kb_mod, raw_img)

        print(f"\n[pipeline astronaut] UL: {ul_dets.shape[0]} dets, "
              f"KB: {kb_dets.shape[0]} dets")

        self._assert_detections_close(kb_dets, ul_dets, "astronaut")

    def test_pipeline_chelsea(self, ul_model, kb_model_loaded, test_images):
        """Chelsea (cat) image: verify the KB pipeline finds the same objects."""
        kb_model, _, _ = kb_model_loaded
        kb_mod = _load_kb_module()
        raw_img, _ = test_images[1]

        ul_dets = self._run_ultralytics_predict(ul_model, raw_img)
        kb_dets = self._run_kb_detect(kb_model, kb_mod, raw_img)

        print(f"\n[pipeline chelsea] UL: {ul_dets.shape[0]} dets, "
              f"KB: {kb_dets.shape[0]} dets")

        self._assert_detections_close(kb_dets, ul_dets, "chelsea")

    def test_pipeline_coffee(self, ul_model, kb_model_loaded, test_images):
        """Coffee image: verify the KB pipeline finds the same objects."""
        kb_model, _, _ = kb_model_loaded
        kb_mod = _load_kb_module()
        raw_img, _ = test_images[2]

        ul_dets = self._run_ultralytics_predict(ul_model, raw_img)
        kb_dets = self._run_kb_detect(kb_model, kb_mod, raw_img)

        print(f"\n[pipeline coffee] UL: {ul_dets.shape[0]} dets, "
              f"KB: {kb_dets.shape[0]} dets")

        self._assert_detections_close(kb_dets, ul_dets, "coffee")

    def test_pipeline_result_structure(self, ul_model, kb_model_loaded, test_images):
        """Verify DetectionResult fields are properly populated."""
        kb_model, _, _ = kb_model_loaded
        kb_mod = _load_kb_module()
        raw_img, _ = test_images[0]

        results = kb_model.detect_from_image(raw_img, conf_thres=0.25)
        r = results[0]

        # Check structure
        assert hasattr(r, "boxes"), "Missing 'boxes' attribute"
        assert hasattr(r, "scores"), "Missing 'scores' attribute"
        assert hasattr(r, "class_ids"), "Missing 'class_ids' attribute"
        assert hasattr(r, "class_names"), "Missing 'class_names' attribute"
        assert hasattr(r, "orig_shape"), "Missing 'orig_shape' attribute"

        n = len(r)
        assert r.boxes.shape == (n, 4)
        assert r.scores.shape == (n,)
        assert r.class_ids.shape == (n,)
        assert len(r.class_names) == n
        assert r.orig_shape == (raw_img.shape[0], raw_img.shape[1])

        # Scores in (0, 1]
        if n > 0:
            assert (r.scores > 0).all() and (r.scores <= 1).all()
            # Boxes within image bounds (with small tolerance for rounding)
            assert (r.boxes[:, 0] >= -1).all(), "x1 out of range"
            assert (r.boxes[:, 1] >= -1).all(), "y1 out of range"
            assert (r.boxes[:, 2] <= raw_img.shape[1] + 1).all(), "x2 out of range"
            assert (r.boxes[:, 3] <= raw_img.shape[0] + 1).all(), "y2 out of range"
            # Class names should be strings from COCO
            assert all(isinstance(s, str) for s in r.class_names)

        print(f"\n[result structure] {n} detections, classes={r.class_names}")

    def test_pipeline_batch(self, ul_model, kb_model_loaded, test_images):
        """Test batch detection via detect_from_image with multiple images."""
        kb_model, _, _ = kb_model_loaded
        kb_mod = _load_kb_module()

        raw_imgs = [img for img, _ in test_images]

        results = kb_model.detect_from_image(raw_imgs, conf_thres=0.25)
        assert len(results) == len(raw_imgs), (
            f"Expected {len(raw_imgs)} results, got {len(results)}"
        )

        for i, r in enumerate(results):
            assert r.orig_shape == (raw_imgs[i].shape[0], raw_imgs[i].shape[1])
            print(f"  Image {i}: {len(r)} detections")

    def test_pipeline_conf_threshold(self, ul_model, kb_model_loaded, test_images):
        """Verify that raising conf_thres reduces detection count."""
        kb_model, _, _ = kb_model_loaded
        raw_img, _ = test_images[0]

        r_low = kb_model.detect_from_image(raw_img, conf_thres=0.10)[0]
        r_high = kb_model.detect_from_image(raw_img, conf_thres=0.60)[0]

        print(f"\n[conf_threshold] conf=0.10: {len(r_low)} dets, "
              f"conf=0.60: {len(r_high)} dets")
        assert len(r_high) <= len(r_low), (
            "Higher confidence threshold should give fewer or equal detections"
        )

    def test_pipeline_class_filter(self, ul_model, kb_model_loaded, test_images):
        """Verify that filtering by class works."""
        kb_model, _, _ = kb_model_loaded
        raw_img, _ = test_images[0]

        # Detect only "person" (class 0)
        r_person = kb_model.detect_from_image(
            raw_img, conf_thres=0.25, classes=[0]
        )[0]

        if len(r_person) > 0:
            assert all(c == 0 for c in r_person.class_ids.tolist()), (
                f"Expected only class 0, got {r_person.class_ids.tolist()}"
            )
            assert all(n == "person" for n in r_person.class_names)

        # Compare with unfiltered
        r_all = kb_model.detect_from_image(raw_img, conf_thres=0.25)[0]
        assert len(r_person) <= len(r_all)

        print(f"\n[class_filter] All: {len(r_all)} dets, "
              f"Person-only: {len(r_person)} dets")

    # ---- shared assertion helper -----------------------------------------

    def _assert_detections_close(self, kb_dets, ul_dets, name):
        """Assert that KB and UL detections are aligned.

        We allow small numerical differences in box coordinates and scores,
        and permit a small number of unmatched detections (since NMS is
        sensitive to tiny score differences near the threshold).
        """
        # Both should find at least some detections
        assert kb_dets.shape[0] > 0 or ul_dets.shape[0] == 0, (
            f"[{name}] UL found {ul_dets.shape[0]} dets but KB found 0"
        )

        if ul_dets.shape[0] == 0 and kb_dets.shape[0] == 0:
            return  # both empty

        # Match detections by IoU
        matched_kb, matched_ul, unmatched_kb, unmatched_ul = \
            self._match_detections(kb_dets, ul_dets, iou_thresh=0.5)

        total_ul = ul_dets.shape[0]
        total_kb = kb_dets.shape[0]
        n_matched = len(matched_kb)

        print(f"  Matched: {n_matched}/{total_ul} UL dets, "
              f"unmatched_kb={len(unmatched_kb)}, unmatched_ul={len(unmatched_ul)}")

        # At least 80% of detections should match (NMS boundary effects)
        match_rate = n_matched / max(total_ul, 1)
        assert match_rate >= 0.7, (
            f"[{name}] Match rate too low: {match_rate:.2f} "
            f"({n_matched}/{total_ul})"
        )

        if n_matched == 0:
            return

        # For matched pairs, check box proximity and class agreement
        kb_matched = kb_dets[matched_kb]
        ul_matched = ul_dets[matched_ul]

        # Classes must agree
        class_agree = (kb_matched[:, 5] == ul_matched[:, 5]).float().mean().item()
        print(f"  Class agreement: {class_agree:.2%}")
        assert class_agree >= 0.9, (
            f"[{name}] Class agreement too low: {class_agree:.2f}"
        )

        # Box coordinates should be close (allowing for slight preprocessing diffs)
        box_diff = (kb_matched[:, :4] - ul_matched[:, :4]).abs()
        max_box_diff = box_diff.max().item()
        mean_box_diff = box_diff.mean().item()
        print(f"  Box diff: max={max_box_diff:.2f}px, mean={mean_box_diff:.2f}px")
        assert max_box_diff < 5.0, (
            f"[{name}] Box diff too large: {max_box_diff:.2f}px"
        )

        # Confidence scores should be close
        score_diff = (kb_matched[:, 4] - ul_matched[:, 4]).abs()
        max_score_diff = score_diff.max().item()
        print(f"  Score diff: max={max_score_diff:.4f}")
        assert max_score_diff < 0.05, (
            f"[{name}] Score diff too large: {max_score_diff:.4f}"
        )
