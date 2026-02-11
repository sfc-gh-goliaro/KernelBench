"""
Stable Diffusion XL alignment tests: KernelBench UNet vs HuggingFace diffusers.

Two test suites in one file:

1. **UNet-level alignment** (TestUNetAlignment)
   - Loads standalone UNet models in float32
   - Compares forward-pass outputs given random tensors
   - Tests: weight transfer, single forward, multiple timesteps, batch,
     different resolutions, output shape, determinism

2. **End-to-end pipeline alignment** (TestEndToEndGeneration)
   - Loads the full StableDiffusionXLPipeline in fp16
   - Swaps in the KB UNet as a drop-in replacement
   - Runs real text prompts through both pipelines and compares final images
   - Tests: various prompts, 1024x1024, multiple seeds, negative prompts

Usage:
    # Run all tests:
    pytest tests/test_sd_hf_alignment.py -v

    # UNet-level tests only:
    pytest tests/test_sd_hf_alignment.py -v -k "TestUNet"

    # End-to-end tests only:
    pytest tests/test_sd_hf_alignment.py -v -k "TestEndToEnd"

    # Save generated images for visual inspection:
    SAVE_IMAGES=1 pytest tests/test_sd_hf_alignment.py -v -k "TestEndToEnd"

Requires:
    - diffusers library
    - CUDA GPU
    - HuggingFace authentication for stabilityai/stable-diffusion-xl-base-1.0
"""

import pytest
import torch
import numpy as np
import sys
import os
import importlib

# ---------------------------------------------------------------------------
# HuggingFace environment
# ---------------------------------------------------------------------------
os.environ["HF_HOME"] = "/home/yak/data-fast/huggingface"
HF_TOKEN_PATH = "/home/yak/data-fast/huggingface/token"
if os.path.exists(HF_TOKEN_PATH):
    with open(HF_TOKEN_PATH, "r") as f:
        os.environ["HF_TOKEN"] = f.read().strip()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(TEST_DIR, "..")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "KernelBench"))

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
diffusers = pytest.importorskip("diffusers")

from diffusers import UNet2DConditionModel, StableDiffusionXLPipeline, EulerDiscreteScheduler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# UNet-level tolerances (float32)
UNET_RTOL_MEAN = 5e-3
UNET_MAX_ABS_DIFF = 1e-2

# E2E pixel tolerances (fp16, 0-255 range)
PIXEL_MSE_THRESHOLD = 1.0
PIXEL_MAX_DIFF_THRESHOLD = 20

# Image saving
SAVE_IMAGES = os.environ.get("SAVE_IMAGES", "0") == "1"
OUTPUT_DIR = os.path.join(TEST_DIR, "outputs")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_kb_unet_from_config(hf_config):
    """Instantiate a KernelBench Model using the config extracted from a HF UNet."""
    kb_module = importlib.import_module("KernelBench.level4.17_StableDiffusion")
    return kb_module.Model(
        sample_size=hf_config.sample_size,
        in_channels=hf_config.in_channels,
        out_channels=hf_config.out_channels,
        down_block_types=tuple(hf_config.down_block_types),
        up_block_types=tuple(hf_config.up_block_types),
        block_out_channels=tuple(hf_config.block_out_channels),
        layers_per_block=hf_config.layers_per_block,
        norm_num_groups=hf_config.norm_num_groups,
        norm_eps=hf_config.norm_eps,
        cross_attention_dim=hf_config.cross_attention_dim,
        transformer_layers_per_block=list(hf_config.transformer_layers_per_block),
        attention_head_dim=list(hf_config.attention_head_dim),
        use_linear_projection=hf_config.use_linear_projection,
        addition_embed_type=hf_config.addition_embed_type,
        addition_time_embed_dim=hf_config.addition_time_embed_dim,
        projection_class_embeddings_input_dim=hf_config.projection_class_embeddings_input_dim,
        flip_sin_to_cos=hf_config.flip_sin_to_cos,
        freq_shift=hf_config.freq_shift,
        act_fn=hf_config.act_fn,
        mid_block_scale_factor=hf_config.mid_block_scale_factor,
    )


def _kb_key_to_hf_key(kb_key: str) -> str:
    """
    Map a KernelBench state_dict key to the corresponding HuggingFace key.

    KB wraps some nn modules inside level1 operators which adds an extra
    segment to the parameter path:
      - GroupNorm  (level1) stores self.gn  → adds ".gn."
      - LayerNorm  (level1) stores self.ln  → adds ".ln."
      - Conv2d     (level1) stores self.conv2d → adds ".conv2d."

    Linear (level1) stores self.weight / self.bias directly, so its keys
    are identical to nn.Linear — no mapping needed.

    This function strips the wrapper segments to recover the HF key.
    """
    # Order matters: replace the most specific patterns first
    for wrapper_seg in (".conv2d.", ".gn.", ".ln."):
        if wrapper_seg in kb_key:
            kb_key = kb_key.replace(wrapper_seg, ".")
            break
    return kb_key


def _copy_weights(src_model, dst_model):
    """
    Copy state_dict from *src_model* (HF) to *dst_model* (KB) by matching keys.

    Handles the systematic key renaming introduced by KB's level1 operator
    wrappers (GroupNorm → .gn., LayerNorm → .ln., Conv2d → .conv2d.).

    Returns (copied, missing_in_src, shape_mismatch, extra_in_src).
    """
    src_sd = src_model.state_dict()
    dst_sd = dst_model.state_dict()

    # Build reverse mapping: hf_key → src_tensor  (already available as src_sd)
    # Build forward mapping: for each KB key, find the corresponding HF key
    copied = 0
    missing_in_src = []
    shape_mismatch = []
    matched_src_keys = set()

    for kb_key, dst_tensor in dst_sd.items():
        hf_key = _kb_key_to_hf_key(kb_key)
        if hf_key in src_sd:
            src_tensor = src_sd[hf_key]
            if dst_tensor.shape == src_tensor.shape:
                dst_tensor.copy_(src_tensor)
                copied += 1
                matched_src_keys.add(hf_key)
            else:
                shape_mismatch.append(
                    f"{kb_key} -> {hf_key} (dst={dst_tensor.shape} vs src={src_tensor.shape})"
                )
        else:
            missing_in_src.append(f"{kb_key} (tried HF key: {hf_key})")

    extra_in_src = [k for k in src_sd if k not in matched_src_keys]

    dst_model.load_state_dict(dst_sd)

    print(f"  Copied {copied} weights")
    if missing_in_src:
        print(f"  Missing in source ({len(missing_in_src)}): {missing_in_src[:10]}")
    if shape_mismatch:
        print(f"  Shape mismatches ({len(shape_mismatch)}): {shape_mismatch[:10]}")
    if extra_in_src:
        print(f"  Extra in source ({len(extra_in_src)}): {extra_in_src[:10]}")

    return copied, missing_in_src, shape_mismatch, extra_in_src


def _compare_tensors(hf_tensor, kb_tensor):
    """Compute abs/rel diff stats between two tensors. Returns dict of metrics."""
    hf_f = hf_tensor.float()
    kb_f = kb_tensor.float()

    abs_diff = (hf_f - kb_f).abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()

    denom = torch.maximum(hf_f.abs(), kb_f.abs()) + 1e-8
    rel_diff = abs_diff / denom
    max_rel = rel_diff.max().item()
    mean_rel = rel_diff.mean().item()

    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "max_rel": max_rel,
        "mean_rel": mean_rel,
    }


def _make_unet_inputs(batch_size, height, width, cross_attention_dim,
                      timestep_val, device, dtype, seq_len=77):
    """Create a standard set of UNet inputs for alignment testing."""
    sample = torch.randn(batch_size, 4, height, width, device=device, dtype=dtype)
    timestep = torch.tensor([timestep_val], device=device, dtype=torch.long)
    encoder_hidden_states = torch.randn(
        batch_size, seq_len, cross_attention_dim, device=device, dtype=dtype
    )
    text_embeds = torch.randn(batch_size, 1280, device=device, dtype=dtype)
    time_ids = torch.tensor(
        [[1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0]] * batch_size,
        device=device, dtype=dtype,
    )
    added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
    return sample, timestep, encoder_hidden_states, added_cond_kwargs


# ===========================================================================
# Fixture: standalone UNet models (float32, for precise UNet-level alignment)
# ===========================================================================

@pytest.fixture(scope="module")
def unet_models():
    """Load HF and KB UNets in float32 with copied weights."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    print(f"\nLoading SDXL UNet from {MODEL_ID} (float32) ...")

    hf_model = UNet2DConditionModel.from_pretrained(
        MODEL_ID, subfolder="unet", torch_dtype=torch.float32,
    ).to(DEVICE).eval()
    print(f"  HF UNet: {sum(p.numel() for p in hf_model.parameters()):,} params")

    hf_config = hf_model.config
    kb_model = _build_kb_unet_from_config(hf_config)
    kb_model = kb_model.to(device=DEVICE, dtype=torch.float32)

    copied, missing, mismatched, extra = _copy_weights(hf_model, kb_model)
    kb_model.eval()
    print(f"  KB UNet: {sum(p.numel() for p in kb_model.parameters()):,} params")

    return hf_model, kb_model, hf_config, copied, missing, mismatched, extra


# ===========================================================================
# Fixture: full SDXL pipelines (fp16, for e2e generation)
# ===========================================================================

@pytest.fixture(scope="module")
def sdxl_pipelines():
    """
    Load HF SDXL pipeline in fp16, build a second pipeline with the KB UNet
    swapped in.  Shares VAE + text encoders to save VRAM.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    print(f"\nLoading SDXL pipeline from {MODEL_ID} (fp16) ...")

    pipe_hf = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    )
    pipe_hf.scheduler = EulerDiscreteScheduler.from_config(pipe_hf.scheduler.config)
    pipe_hf = pipe_hf.to(DEVICE)
    print(f"  HF pipeline loaded  "
          f"(UNet: {sum(p.numel() for p in pipe_hf.unet.parameters()):,} params)")

    # Build KB UNet
    hf_config = pipe_hf.unet.config
    kb_unet = _build_kb_unet_from_config(hf_config)

    # Copy weights
    copied, missing, mismatched, extra = _copy_weights(pipe_hf.unet, kb_unet)
    assert len(missing) == 0, f"Missing weights: {missing[:10]}"
    assert len(mismatched) == 0, f"Shape mismatches: {mismatched[:10]}"
    assert len(extra) == 0, f"Extra HF weights: {extra[:10]}"

    kb_unet = kb_unet.to(device=DEVICE, dtype=torch.float16).eval()
    print(f"  KB UNet created & weights copied  "
          f"({sum(p.numel() for p in kb_unet.parameters()):,} params)")

    pipe_kb = StableDiffusionXLPipeline(
        vae=pipe_hf.vae,
        text_encoder=pipe_hf.text_encoder,
        text_encoder_2=pipe_hf.text_encoder_2,
        tokenizer=pipe_hf.tokenizer,
        tokenizer_2=pipe_hf.tokenizer_2,
        unet=kb_unet,
        scheduler=EulerDiscreteScheduler.from_config(pipe_hf.scheduler.config),
    ).to(DEVICE)
    print("  KB pipeline assembled (shared VAE + text encoders)")

    return pipe_hf, pipe_kb


# ===========================================================================
# E2E helpers
# ===========================================================================

def _pil_to_np(img) -> np.ndarray:
    return np.array(img).astype(np.float32)


def _compare_images(img_a, img_b, tag: str = ""):
    arr_a, arr_b = _pil_to_np(img_a), _pil_to_np(img_b)
    assert arr_a.shape == arr_b.shape, f"Shape mismatch: {arr_a.shape} vs {arr_b.shape}"
    diff = arr_a - arr_b
    mse = float(np.mean(diff ** 2))
    max_diff = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    print(f"  [{tag}] shape={arr_a.shape}  pixel MSE={mse:.4f}  "
          f"mean_abs={mean_abs:.4f}  max_diff={max_diff:.1f}")
    return mse, max_diff


def _save_image(img, name: str):
    if SAVE_IMAGES:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, name)
        img.save(path)
        print(f"  Saved: {path}")


def _generate_pair(pipe_hf, pipe_kb, prompt, seed=42, num_steps=20,
                   height=512, width=512, guidance_scale=7.5,
                   negative_prompt=""):
    gen_hf = torch.Generator(device=DEVICE).manual_seed(seed)
    img_hf = pipe_hf(
        prompt=prompt, negative_prompt=negative_prompt,
        height=height, width=width,
        num_inference_steps=num_steps, guidance_scale=guidance_scale,
        generator=gen_hf,
    ).images[0]

    gen_kb = torch.Generator(device=DEVICE).manual_seed(seed)
    img_kb = pipe_kb(
        prompt=prompt, negative_prompt=negative_prompt,
        height=height, width=width,
        num_inference_steps=num_steps, guidance_scale=guidance_scale,
        generator=gen_kb,
    ).images[0]

    return img_hf, img_kb


# ###########################################################################
#  UNet-Level Alignment Tests (float32)
# ###########################################################################

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestUNetAlignment:
    """Compare UNet forward passes with random inputs in float32."""

    def test_weight_transfer_completeness(self, unet_models):
        """All weights transfer: no missing keys, no shape mismatches."""
        hf_model, kb_model, _, copied, missing, mismatched, extra = unet_models

        hf_n = sum(p.numel() for p in hf_model.parameters())
        kb_n = sum(p.numel() for p in kb_model.parameters())

        print(f"\n  HF params: {hf_n:,}  KB params: {kb_n:,}  copied: {copied}")
        assert len(missing) == 0, f"Missing: {missing[:10]}"
        assert len(mismatched) == 0, f"Mismatched: {mismatched[:10]}"
        assert hf_n == kb_n, f"Param count: HF={hf_n:,} vs KB={kb_n:,}"

    def test_forward_alignment(self, unet_models):
        """Single forward pass produces matching outputs."""
        hf_model, kb_model, hf_config, *_ = unet_models

        torch.manual_seed(42)
        sample, timestep, enc_hs, added = _make_unet_inputs(
            1, 64, 64, hf_config.cross_attention_dim, 100, DEVICE, torch.float32,
        )

        with torch.no_grad():
            hf_out = hf_model(sample=sample, timestep=timestep,
                              encoder_hidden_states=enc_hs,
                              added_cond_kwargs=added, return_dict=False)[0]
            kb_out = kb_model(sample=sample, timestep=timestep,
                              encoder_hidden_states=enc_hs,
                              added_cond_kwargs=added, return_dict=False)[0]

        assert hf_out.shape == kb_out.shape
        m = _compare_tensors(hf_out, kb_out)
        print(f"\n  abs: max={m['max_abs']:.2e} mean={m['mean_abs']:.2e}  "
              f"rel: max={m['max_rel']:.2e} mean={m['mean_rel']:.2e}")
        assert m["mean_rel"] < UNET_RTOL_MEAN
        assert m["max_abs"] < UNET_MAX_ABS_DIFF

    def test_forward_multiple_timesteps(self, unet_models):
        """Alignment holds across the full diffusion schedule."""
        hf_model, kb_model, hf_config, *_ = unet_models

        torch.manual_seed(123)
        sample, _, enc_hs, added = _make_unet_inputs(
            1, 64, 64, hf_config.cross_attention_dim, 0, DEVICE, torch.float32,
        )

        all_pass = True
        for t in [0, 50, 250, 500, 750, 999]:
            ts = torch.tensor([t], device=DEVICE, dtype=torch.long)
            with torch.no_grad():
                hf_out = hf_model(sample=sample, timestep=ts,
                                  encoder_hidden_states=enc_hs,
                                  added_cond_kwargs=added, return_dict=False)[0]
                kb_out = kb_model(sample=sample, timestep=ts,
                                  encoder_hidden_states=enc_hs,
                                  added_cond_kwargs=added, return_dict=False)[0]
            m = _compare_tensors(hf_out, kb_out)
            ok = m["mean_rel"] < UNET_RTOL_MEAN and m["max_abs"] < UNET_MAX_ABS_DIFF
            if not ok:
                all_pass = False
            print(f"  t={t:4d}: {'PASS' if ok else 'FAIL'}  "
                  f"abs max={m['max_abs']:.2e}  rel mean={m['mean_rel']:.2e}")

        assert all_pass, "Some timesteps failed"

    def test_forward_batch(self, unet_models):
        """Alignment with batch size 2."""
        hf_model, kb_model, hf_config, *_ = unet_models

        torch.manual_seed(456)
        sample, timestep, enc_hs, added = _make_unet_inputs(
            2, 64, 64, hf_config.cross_attention_dim, 500, DEVICE, torch.float32,
        )

        with torch.no_grad():
            hf_out = hf_model(sample=sample, timestep=timestep,
                              encoder_hidden_states=enc_hs,
                              added_cond_kwargs=added, return_dict=False)[0]
            kb_out = kb_model(sample=sample, timestep=timestep,
                              encoder_hidden_states=enc_hs,
                              added_cond_kwargs=added, return_dict=False)[0]

        m = _compare_tensors(hf_out, kb_out)
        print(f"\n  shape={hf_out.shape}  abs max={m['max_abs']:.2e}  "
              f"rel mean={m['mean_rel']:.2e}")
        assert m["mean_rel"] < UNET_RTOL_MEAN
        assert m["max_abs"] < UNET_MAX_ABS_DIFF

    def test_forward_different_resolutions(self, unet_models):
        """Alignment at 32x32, 64x64, 96x96 spatial sizes."""
        hf_model, kb_model, hf_config, *_ = unet_models

        all_pass = True
        for h, w in [(32, 32), (64, 64), (96, 96)]:
            torch.manual_seed(789)
            sample, timestep, enc_hs, added = _make_unet_inputs(
                1, h, w, hf_config.cross_attention_dim, 250, DEVICE, torch.float32,
            )
            with torch.no_grad():
                hf_out = hf_model(sample=sample, timestep=timestep,
                                  encoder_hidden_states=enc_hs,
                                  added_cond_kwargs=added, return_dict=False)[0]
                kb_out = kb_model(sample=sample, timestep=timestep,
                                  encoder_hidden_states=enc_hs,
                                  added_cond_kwargs=added, return_dict=False)[0]
            m = _compare_tensors(hf_out, kb_out)
            ok = m["mean_rel"] < UNET_RTOL_MEAN and m["max_abs"] < UNET_MAX_ABS_DIFF
            if not ok:
                all_pass = False
            print(f"  {h}x{w}: {'PASS' if ok else 'FAIL'}  "
                  f"abs max={m['max_abs']:.2e}  rel mean={m['mean_rel']:.2e}")

        assert all_pass, "Some resolutions failed"

    def test_output_shape(self, unet_models):
        """Output shape is (B, out_channels, H, W), no NaN/Inf."""
        _, kb_model, hf_config, *_ = unet_models

        torch.manual_seed(0)
        sample, timestep, enc_hs, added = _make_unet_inputs(
            1, 64, 64, hf_config.cross_attention_dim, 500, DEVICE, torch.float32,
        )
        with torch.no_grad():
            out = kb_model(sample=sample, timestep=timestep,
                           encoder_hidden_states=enc_hs,
                           added_cond_kwargs=added, return_dict=False)[0]

        expected = (1, hf_config.out_channels, 64, 64)
        assert out.shape == expected, f"Expected {expected}, got {out.shape}"
        assert not torch.isnan(out).any(), "NaN in output"
        assert not torch.isinf(out).any(), "Inf in output"

    def test_deterministic(self, unet_models):
        """Identical inputs produce identical outputs across two runs."""
        _, kb_model, hf_config, *_ = unet_models

        torch.manual_seed(42)
        sample, timestep, enc_hs, added = _make_unet_inputs(
            1, 64, 64, hf_config.cross_attention_dim, 500, DEVICE, torch.float32,
        )
        with torch.no_grad():
            out1 = kb_model(sample=sample, timestep=timestep,
                            encoder_hidden_states=enc_hs,
                            added_cond_kwargs=added, return_dict=False)[0]
            out2 = kb_model(sample=sample, timestep=timestep,
                            encoder_hidden_states=enc_hs,
                            added_cond_kwargs=added, return_dict=False)[0]

        diff = (out1 - out2).abs().max().item()
        print(f"\n  Max diff between two runs: {diff:.2e}")
        assert diff == 0.0 or diff < 1e-6


# ###########################################################################
#  End-to-End Pipeline Tests (fp16, realistic prompts)
# ###########################################################################

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestEndToEndGeneration:
    """Full SDXL pipeline: text encode -> denoise -> VAE decode -> compare images."""

    def test_astronaut(self, sdxl_pipelines):
        """Prompt: 'An astronaut riding a green horse'."""
        pipe_hf, pipe_kb = sdxl_pipelines
        prompt = "An astronaut riding a green horse"
        print(f"\n  Prompt: \"{prompt}\"")

        img_hf, img_kb = _generate_pair(pipe_hf, pipe_kb, prompt=prompt,
                                         seed=42, num_steps=20)
        _save_image(img_hf, "astronaut_hf.png")
        _save_image(img_kb, "astronaut_kb.png")
        mse, max_diff = _compare_images(img_hf, img_kb, tag="astronaut")

        assert mse < PIXEL_MSE_THRESHOLD
        assert max_diff < PIXEL_MAX_DIFF_THRESHOLD

    def test_landscape(self, sdxl_pipelines):
        """Prompt: landscape / sunset."""
        pipe_hf, pipe_kb = sdxl_pipelines
        prompt = "A beautiful sunset over a mountain lake, photorealistic, 8k"
        print(f"\n  Prompt: \"{prompt}\"")

        img_hf, img_kb = _generate_pair(pipe_hf, pipe_kb, prompt=prompt,
                                         seed=123, num_steps=20)
        _save_image(img_hf, "landscape_hf.png")
        _save_image(img_kb, "landscape_kb.png")
        mse, max_diff = _compare_images(img_hf, img_kb, tag="landscape")

        assert mse < PIXEL_MSE_THRESHOLD
        assert max_diff < PIXEL_MAX_DIFF_THRESHOLD

    def test_portrait(self, sdxl_pipelines):
        """Prompt: portrait / studio."""
        pipe_hf, pipe_kb = sdxl_pipelines
        prompt = "Portrait of a cat wearing a tiny top hat, studio lighting"
        print(f"\n  Prompt: \"{prompt}\"")

        img_hf, img_kb = _generate_pair(pipe_hf, pipe_kb, prompt=prompt,
                                         seed=7, num_steps=20)
        _save_image(img_hf, "portrait_hf.png")
        _save_image(img_kb, "portrait_kb.png")
        mse, max_diff = _compare_images(img_hf, img_kb, tag="portrait")

        assert mse < PIXEL_MSE_THRESHOLD
        assert max_diff < PIXEL_MAX_DIFF_THRESHOLD

    def test_high_res(self, sdxl_pipelines):
        """1024x1024 native SDXL resolution."""
        pipe_hf, pipe_kb = sdxl_pipelines
        prompt = "An astronaut riding a green horse"
        print(f"\n  Prompt: \"{prompt}\" (1024x1024)")

        img_hf, img_kb = _generate_pair(pipe_hf, pipe_kb, prompt=prompt,
                                         seed=42, num_steps=20,
                                         height=1024, width=1024)
        _save_image(img_hf, "astronaut_1024_hf.png")
        _save_image(img_kb, "astronaut_1024_kb.png")
        mse, max_diff = _compare_images(img_hf, img_kb, tag="1024x1024")

        assert mse < PIXEL_MSE_THRESHOLD
        assert max_diff < PIXEL_MAX_DIFF_THRESHOLD

    def test_different_seeds(self, sdxl_pipelines):
        """Alignment holds across multiple random seeds."""
        pipe_hf, pipe_kb = sdxl_pipelines
        prompt = "An astronaut riding a green horse"
        print(f"\n  Prompt: \"{prompt}\" (multiple seeds)")

        all_pass = True
        for seed in [0, 99, 2024, 314159]:
            img_hf, img_kb = _generate_pair(pipe_hf, pipe_kb, prompt=prompt,
                                             seed=seed, num_steps=15)
            _save_image(img_hf, f"seed{seed}_hf.png")
            _save_image(img_kb, f"seed{seed}_kb.png")
            mse, max_diff = _compare_images(img_hf, img_kb, tag=f"seed={seed}")
            if mse >= PIXEL_MSE_THRESHOLD or max_diff >= PIXEL_MAX_DIFF_THRESHOLD:
                all_pass = False

        assert all_pass, "One or more seeds failed pixel alignment"

    def test_with_negative_prompt(self, sdxl_pipelines):
        """Classifier-free guidance with a negative prompt."""
        pipe_hf, pipe_kb = sdxl_pipelines
        prompt = "A futuristic city skyline at night, neon lights, cyberpunk"
        negative_prompt = "blurry, low quality, watermark"
        print(f"\n  Prompt: \"{prompt}\"")
        print(f"  Negative: \"{negative_prompt}\"")

        img_hf, img_kb = _generate_pair(pipe_hf, pipe_kb, prompt=prompt,
                                         negative_prompt=negative_prompt,
                                         seed=55, num_steps=20,
                                         guidance_scale=9.0)
        _save_image(img_hf, "cyberpunk_hf.png")
        _save_image(img_kb, "cyberpunk_kb.png")
        mse, max_diff = _compare_images(img_hf, img_kb, tag="negative_prompt")

        assert mse < PIXEL_MSE_THRESHOLD
        assert max_diff < PIXEL_MAX_DIFF_THRESHOLD
