"""
Test model alignment with timm (PyTorch Image Models) library.

This test validates that KernelBench model implementations produce
outputs matching the timm implementation using real images.

Supports:
- MobileNetV4: timm/mobilenetv4_conv_medium.e500_r256_in1k
- EfficientNetV2: timm/tf_efficientnetv2_s.in21k

Usage:
    pytest tests/test_timm_alignment.py --model-name timm/mobilenetv4_conv_medium.e500_r256_in1k -v -s
    pytest tests/test_timm_alignment.py --model-name timm/tf_efficientnetv2_s.in21k -v -s
"""

import pytest
import torch
import sys
import os
import importlib
from typing import List, Tuple, Optional, Dict
import requests
from io import BytesIO
from PIL import Image

# Set up HuggingFace environment
os.environ["HF_HOME"] = "/home/yak/data-fast/huggingface"
HF_TOKEN_PATH = "/home/yak/data-fast/huggingface/token"
if os.path.exists(HF_TOKEN_PATH):
    with open(HF_TOKEN_PATH, "r") as f:
        os.environ["HF_TOKEN"] = f.read().strip()

# Add paths for imports
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(TEST_DIR, '..')
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'KernelBench'))

# Skip all tests if timm is not available
timm = pytest.importorskip("timm")

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32  # timm models work best in fp32 for alignment

# Tolerance thresholds (strict: implementations should be bit-exact in fp32)
RTOL_MEAN = 1e-5
RTOL_MAX = 1e-3
ATOL = 1e-6

# ============================================================================
# Model to KernelBench Implementation Mapping
# ============================================================================

MODEL_TO_IMPLEMENTATION: Dict[str, str] = {
    "timm/mobilenetv4_conv_medium.e500_r256_in1k": "KernelBench.level4.20_MobileNetV4",
    "timm/tf_efficientnetv2_s.in21k": "KernelBench.level4.22_EfficientNetV2",
}

# Model-specific configs
MODEL_CONFIGS: Dict[str, dict] = {
    "timm/mobilenetv4_conv_medium.e500_r256_in1k": {
        "num_classes": 1000,
        "in_chans": 3,
        "stem_size": 32,
        "num_features": 1280,
        "head_channels": 960,
    },
    "timm/tf_efficientnetv2_s.in21k": {
        "num_classes": 21843,
        "in_chans": 3,
        "stem_size": 24,
        "num_features": 1280,
        "last_stage_channels": 256,
    },
}

# Test images - publicly accessible COCO val2017 URLs
TEST_IMAGE_URLS = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "http://images.cocodataset.org/val2017/000000281759.jpg",
    "http://images.cocodataset.org/val2017/000000397133.jpg",
    "http://images.cocodataset.org/val2017/000000252219.jpg",
    "http://images.cocodataset.org/val2017/000000087038.jpg",
]


def _load_test_images() -> List[Tuple[Image.Image, str]]:
    """Download and return (PIL image, URL) pairs."""
    images = []
    for url in TEST_IMAGE_URLS:
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            images.append((img, url))
        except Exception as e:
            print(f"  Warning: Could not load image from {url}: {e}")
            continue
    if not images:
        raise RuntimeError("Could not load any test images from URLs")
    return images


def _get_timm_model_name(model_name: str) -> str:
    """Strip 'timm/' prefix to get the actual timm model name."""
    if model_name.startswith("timm/"):
        return model_name[5:]
    return model_name


def copy_timm_weights_to_kb(timm_model, kb_model):
    """Copy weights from timm model to KernelBench model.
    
    Both models should have matching state dict key structures.
    Handles running_mean/running_var/num_batches_tracked buffers.
    """
    timm_state = timm_model.state_dict()
    kb_state = kb_model.state_dict()
    
    copied = 0
    missing = []
    
    for kb_key, kb_tensor in kb_state.items():
        if kb_key in timm_state:
            timm_tensor = timm_state[kb_key]
            if kb_tensor.shape == timm_tensor.shape:
                kb_tensor.copy_(timm_tensor)
                copied += 1
            else:
                missing.append(f"{kb_key} (shape mismatch: KB={kb_tensor.shape} vs timm={timm_tensor.shape})")
        else:
            missing.append(kb_key)
    
    if missing:
        print(f"  Warning: {len(missing)} KB weights not found in timm model:")
        for m in missing[:10]:
            print(f"    - {m}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")
    
    print(f"  Copied {copied} weights/buffers")
    kb_model.load_state_dict(kb_state)


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_addoption(parser):
    parser.addoption(
        "--model-name",
        action="store",
        default="timm/mobilenetv4_conv_medium.e500_r256_in1k",
        help="timm model name (e.g., timm/mobilenetv4_conv_medium.e500_r256_in1k)",
    )


@pytest.fixture(scope="module")
def loaded_models(request):
    """Load timm and KernelBench models for comparison."""
    model_name = request.config.getoption("--model-name")
    
    if model_name not in MODEL_TO_IMPLEMENTATION:
        pytest.skip(f"No KernelBench implementation for {model_name}. "
                     f"Available: {list(MODEL_TO_IMPLEMENTATION.keys())}")
    
    timm_name = _get_timm_model_name(model_name)
    
    print(f"\nLoading timm model: {timm_name}")
    timm_model = timm.create_model(timm_name, pretrained=True)
    timm_model = timm_model.to(device=DEVICE, dtype=DTYPE)
    timm_model.eval()
    
    # Get the data config for preprocessing
    data_config = timm.data.resolve_model_data_config(timm_model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    
    print(f"Loading KernelBench model: {MODEL_TO_IMPLEMENTATION[model_name]}")
    kb_module = importlib.import_module(MODEL_TO_IMPLEMENTATION[model_name])
    kb_config = MODEL_CONFIGS[model_name]
    kb_model = kb_module.Model(**kb_config)
    
    # Copy weights
    print("Copying weights from timm to KB...")
    copy_timm_weights_to_kb(timm_model, kb_model)
    kb_model = kb_model.to(device=DEVICE, dtype=DTYPE)
    kb_model.eval()
    
    print(f"Models loaded successfully. Config: {kb_config}")
    
    return timm_model, kb_model, transform, model_name


# ============================================================================
# Alignment Tests
# ============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefill_alignment(loaded_models):
    """Test that KB model produces matching outputs with timm model on real images."""
    timm_model, kb_model, transform, model_name = loaded_models
    
    print("\n" + "="*70)
    print(f"Testing Alignment for {model_name}")
    print("="*70)
    
    test_images = _load_test_images()
    print(f"  Loaded {len(test_images)} test images")
    
    for i, (pil_image, image_url) in enumerate(test_images):
        # Preprocess with timm's transform
        input_tensor = transform(pil_image).unsqueeze(0).to(device=DEVICE, dtype=DTYPE)
        
        with torch.no_grad():
            timm_logits = timm_model(input_tensor)
            kb_logits = kb_model(input_tensor)
        
        timm_flat = timm_logits.float()
        kb_flat = kb_logits.float()
        
        abs_diff = (timm_flat - kb_flat).abs()
        max_abs_diff = abs_diff.max().item()
        mean_abs_diff = abs_diff.mean().item()
        
        denominator = torch.maximum(timm_flat.abs(), kb_flat.abs()) + 1e-8
        rel_diff = abs_diff / denominator
        max_rel_diff = rel_diff.max().item()
        mean_rel_diff = rel_diff.mean().item()
        
        timm_top = timm_flat.argmax(dim=-1).item()
        kb_top = kb_flat.argmax(dim=-1).item()
        top_match = timm_top == kb_top
        
        mean_ok = mean_rel_diff < RTOL_MEAN
        max_ok = max_rel_diff < RTOL_MAX
        is_pass = top_match and mean_ok and max_ok
        status = "PASS" if is_pass else "FAIL"
        
        img_w, img_h = pil_image.size
        print(f"\n  [{i}] {status}: image ({img_w}x{img_h}, URL: ...{image_url[-20:]})")
        print(f"      abs_diff: max={max_abs_diff:.2e}, mean={mean_abs_diff:.2e}")
        print(f"      rel_diff: max={max_rel_diff:.2e}, mean={mean_rel_diff:.2e}")
        print(f"      timm top: {timm_top} | KB top: {kb_top} (match={top_match})")
        
        assert top_match, f"Top predictions differ: timm={timm_top} vs KB={kb_top}"
        assert mean_ok, f"Mean relative diff {mean_rel_diff:.2e} exceeds tolerance {RTOL_MEAN}"
    
    print("\n" + "-"*70)
    print(f"All alignment tests passed for {model_name}!")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_random_input_alignment(loaded_models):
    """Test alignment with random input tensors of various sizes."""
    timm_model, kb_model, transform, model_name = loaded_models
    
    print("\n" + "="*70)
    print(f"Testing Random Input Alignment for {model_name}")
    print("="*70)
    
    # Get the expected input size from the data config
    data_config = timm.data.resolve_model_data_config(timm_model)
    input_size = data_config.get('input_size', (3, 224, 224))
    
    # Test with different batch sizes
    for batch_size in [1, 2, 4]:
        x = torch.randn(batch_size, *input_size, device=DEVICE, dtype=DTYPE)
        
        with torch.no_grad():
            timm_out = timm_model(x)
            kb_out = kb_model(x)
        
        abs_diff = (timm_out.float() - kb_out.float()).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        
        # Check top-1 predictions match for each sample
        timm_preds = timm_out.argmax(dim=-1)
        kb_preds = kb_out.argmax(dim=-1)
        preds_match = (timm_preds == kb_preds).all().item()
        
        print(f"\n  Batch size {batch_size}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, "
              f"preds_match={preds_match}")
        
        assert mean_diff < ATOL, f"Mean diff {mean_diff:.2e} too large for batch_size={batch_size}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"] + sys.argv[1:])
