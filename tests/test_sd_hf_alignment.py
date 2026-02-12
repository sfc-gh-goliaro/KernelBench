"""
Diffusion model alignment tests: KernelBench vs HuggingFace diffusers.

A single, model-agnostic test suite driven by the ``--diffusion-model-name``
CLI parameter.  Each supported model family registers a "spec" that tells the
generic tests how to:
  - load the HF denoiser and build the KB counterpart,
  - create random inputs for model-level alignment,
  - run a forward pass (different call signatures),
  - build an end-to-end pipeline with the KB denoiser swapped in.

Currently supported model families:

  * stabilityai/stable-diffusion-xl-base-1.0   (SDXL, UNet)
  * stabilityai/stable-diffusion-3.5-large     (SD3.5, MMDiT transformer)

Adding a new diffusion model only requires writing a new ``ModelSpec`` and
registering it in ``MODEL_REGISTRY``.

Usage:
    # SDXL:
    pytest tests/test_sd_hf_alignment.py -v \\
        --model-name stabilityai/stable-diffusion-xl-base-1.0

    # SD3.5:
    pytest tests/test_sd_hf_alignment.py -v \\
        --model-name stabilityai/stable-diffusion-3.5-large

    # Model-level tests only:
    pytest tests/test_sd_hf_alignment.py -v -k "TestModelAlignment" \\
        --model-name stabilityai/stable-diffusion-xl-base-1.0

    # End-to-end tests only:
    pytest tests/test_sd_hf_alignment.py -v -k "TestE2E" \\
        --model-name stabilityai/stable-diffusion-xl-base-1.0

    # Save generated images for visual inspection:
    pytest tests/test_sd_hf_alignment.py -v -k "TestE2E" \\
        --model-name stabilityai/stable-diffusion-xl-base-1.0 --save-images

Requires:
    - diffusers library
    - CUDA GPU (40 GB+ VRAM for SD3.5 e2e tests)
    - HuggingFace authentication for gated models
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch

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

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# E2E pixel tolerances (fp16, 0-255 range)
PIXEL_MSE_THRESHOLD = 1.0
PIXEL_MAX_DIFF_THRESHOLD = 20

# Image saving (set by --save-images flag; resolved at fixture time)
OUTPUT_DIR = os.path.join(TEST_DIR, "outputs")


# ###########################################################################
#  ModelSpec: per-model configuration
# ###########################################################################

@dataclass
class ModelSpec:
    """Everything the generic tests need to know about a diffusion model."""

    # --- identifiers -------------------------------------------------------
    model_id: str                       # HF repo id
    short_name: str                     # prefix for saved images / prints

    # --- model-level alignment (float32) -----------------------------------
    rtol_mean: float = 5e-3
    max_abs_diff: float = 1e-2

    # --- e2e defaults ------------------------------------------------------
    e2e_guidance_scale: float = 7.5
    e2e_default_height: int = 512
    e2e_default_width: int = 512
    e2e_native_height: Optional[int] = None   # for high-res test; None = skip
    e2e_native_width: Optional[int] = None
    e2e_seeds: List[int] = field(default_factory=lambda: [0, 99, 2024])

    # --- timestep values for multi-timestep test ---------------------------
    timestep_values: List[float] = field(
        default_factory=lambda: [0, 50, 250, 500, 750, 999])

    # --- resolution grid for multi-resolution test -------------------------
    resolution_grid: List[Tuple[int, int]] = field(
        default_factory=lambda: [(32, 32), (64, 64), (96, 96)])

    # --- callbacks (set by register helpers below) -------------------------
    # load_hf_denoiser(model_id) -> (hf_model, hf_config)
    load_hf_denoiser: Optional[Callable] = field(default=None, repr=False)
    # build_kb_denoiser(hf_config) -> kb_model
    build_kb_denoiser: Optional[Callable] = field(default=None, repr=False)
    # make_inputs(batch, h, w, hf_config, timestep_val, device, dtype) -> dict
    make_inputs: Optional[Callable] = field(default=None, repr=False)
    # forward(model, inputs) -> tensor
    forward_fn: Optional[Callable] = field(default=None, repr=False)
    # out_channels(hf_config) -> int
    get_out_channels: Optional[Callable] = field(default=None, repr=False)
    # load_hf_pipeline(model_id) -> pipe_hf
    load_hf_pipeline: Optional[Callable] = field(default=None, repr=False)
    # build_kb_pipeline(pipe_hf, kb_denoiser) -> pipe_kb
    build_kb_pipeline: Optional[Callable] = field(default=None, repr=False)
    # get_hf_denoiser_from_pipeline(pipe_hf) -> hf_denoiser_module
    get_pipeline_denoiser: Optional[Callable] = field(default=None, repr=False)
    # get_hf_config_from_pipeline(pipe_hf) -> hf_config
    get_pipeline_config: Optional[Callable] = field(default=None, repr=False)


# ###########################################################################
#  Model registry
# ###########################################################################

MODEL_REGISTRY: Dict[str, ModelSpec] = {}


def _register(spec: ModelSpec) -> None:
    MODEL_REGISTRY[spec.model_id] = spec


# ---------------------------------------------------------------------------
#  SDXL
# ---------------------------------------------------------------------------

def _sdxl_load_hf_denoiser(model_id: str):
    from diffusers import UNet2DConditionModel
    hf = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=torch.float32,
    ).to(DEVICE).eval()
    return hf, hf.config


def _sdxl_build_kb(hf_config):
    kb_mod = importlib.import_module("KernelBench.level4.17_StableDiffusion")
    return kb_mod.StableDiffusionXL(
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


def _sdxl_make_inputs(batch, h, w, hf_config, ts_val, device, dtype):
    sample = torch.randn(batch, 4, h, w, device=device, dtype=dtype)
    timestep = torch.tensor([ts_val], device=device, dtype=torch.long)
    enc_hs = torch.randn(batch, 77, hf_config.cross_attention_dim,
                         device=device, dtype=dtype)
    text_embeds = torch.randn(batch, 1280, device=device, dtype=dtype)
    time_ids = torch.tensor(
        [[1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0]] * batch,
        device=device, dtype=dtype)
    return dict(sample=sample, timestep=timestep,
                encoder_hidden_states=enc_hs,
                added_cond_kwargs={"text_embeds": text_embeds,
                                   "time_ids": time_ids})


def _sdxl_forward(model, inputs):
    return model(**inputs, return_dict=False)[0]


def _sdxl_load_pipeline(model_id):
    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True)
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config)
    return pipe.to(DEVICE)


def _sdxl_build_kb_pipeline(pipe_hf, kb_denoiser):
    from diffusers import EulerDiscreteScheduler
    kb_pipe_mod = importlib.import_module("KernelBench.level4.17_StableDiffusion")
    kb_clip_mod = importlib.import_module("KernelBench.level3.encoder._5_CLIPTextEncoder")
    kb_vae_mod = importlib.import_module("KernelBench.level3.vae._4_VAEDecoder")

    # --- KB text_encoder (CLIPTextModel) ---
    hf_te1 = pipe_hf.text_encoder
    te1_cfg = hf_te1.config
    kb_te1 = kb_clip_mod.CLIPTextModel(
        vocab_size=te1_cfg.vocab_size,
        hidden_size=te1_cfg.hidden_size,
        intermediate_size=te1_cfg.intermediate_size,
        num_hidden_layers=te1_cfg.num_hidden_layers,
        num_attention_heads=te1_cfg.num_attention_heads,
        max_position_embeddings=te1_cfg.max_position_embeddings,
        hidden_act=te1_cfg.hidden_act,
        layer_norm_eps=te1_cfg.layer_norm_eps,
        projection_dim=te1_cfg.projection_dim,
    )
    copied, missing, mismatch, extra = _copy_weights(hf_te1, kb_te1)
    assert len(missing) == 0, f"text_encoder missing: {missing[:5]}"
    assert len(mismatch) == 0, f"text_encoder mismatch: {mismatch[:5]}"
    kb_te1 = kb_te1.to(device=DEVICE, dtype=torch.float16).eval()

    # --- KB text_encoder_2 (CLIPTextModelWithProjection) ---
    hf_te2 = pipe_hf.text_encoder_2
    te2_cfg = hf_te2.config
    kb_te2 = kb_clip_mod.CLIPTextModelWithProjection(
        vocab_size=te2_cfg.vocab_size,
        hidden_size=te2_cfg.hidden_size,
        intermediate_size=te2_cfg.intermediate_size,
        num_hidden_layers=te2_cfg.num_hidden_layers,
        num_attention_heads=te2_cfg.num_attention_heads,
        max_position_embeddings=te2_cfg.max_position_embeddings,
        hidden_act=te2_cfg.hidden_act,
        layer_norm_eps=te2_cfg.layer_norm_eps,
        projection_dim=te2_cfg.projection_dim,
    )
    copied, missing, mismatch, extra = _copy_weights(hf_te2, kb_te2)
    assert len(missing) == 0, f"text_encoder_2 missing: {missing[:5]}"
    assert len(mismatch) == 0, f"text_encoder_2 mismatch: {mismatch[:5]}"
    kb_te2 = kb_te2.to(device=DEVICE, dtype=torch.float16).eval()

    # --- KB VAE decoder ---
    hf_vae = pipe_hf.vae
    vae_cfg = hf_vae.config
    kb_vae = kb_vae_mod.VAEDecoder(
        latent_channels=vae_cfg.latent_channels,
        out_channels=vae_cfg.out_channels,
        block_out_channels=tuple(vae_cfg.block_out_channels),
        layers_per_block=vae_cfg.layers_per_block,
        norm_num_groups=vae_cfg.norm_num_groups,
        scaling_factor=vae_cfg.scaling_factor,
        shift_factor=getattr(vae_cfg, "shift_factor", None),
        force_upcast=getattr(vae_cfg, "force_upcast", True),
        use_post_quant_conv=getattr(vae_cfg, "use_post_quant_conv", True),
    )
    # Copy only decoder + post_quant_conv weights from full VAE
    copied, missing, mismatch, extra = _copy_weights(hf_vae, kb_vae)
    assert len(missing) == 0, f"VAE missing: {missing[:5]}"
    assert len(mismatch) == 0, f"VAE mismatch: {mismatch[:5]}"
    kb_vae = kb_vae.to(device=DEVICE, dtype=torch.float16).eval()

    return kb_pipe_mod.StableDiffusionXLPipeline(
        vae=kb_vae,
        text_encoder=kb_te1,
        text_encoder_2=kb_te2,
        tokenizer=pipe_hf.tokenizer,
        tokenizer_2=pipe_hf.tokenizer_2,
        unet=kb_denoiser,
        scheduler=EulerDiscreteScheduler.from_config(
            pipe_hf.scheduler.config),
    ).to(DEVICE)


_register(ModelSpec(
    model_id="stabilityai/stable-diffusion-xl-base-1.0",
    short_name="sdxl",
    rtol_mean=5e-3,
    max_abs_diff=1e-2,
    e2e_guidance_scale=7.5,
    e2e_default_height=512,
    e2e_default_width=512,
    e2e_native_height=1024,
    e2e_native_width=1024,
    e2e_seeds=[0, 99, 2024, 314159],
    timestep_values=[0, 50, 250, 500, 750, 999],
    load_hf_denoiser=_sdxl_load_hf_denoiser,
    build_kb_denoiser=_sdxl_build_kb,
    make_inputs=_sdxl_make_inputs,
    forward_fn=_sdxl_forward,
    get_out_channels=lambda cfg: cfg.out_channels,
    load_hf_pipeline=_sdxl_load_pipeline,
    build_kb_pipeline=_sdxl_build_kb_pipeline,
    get_pipeline_denoiser=lambda pipe: pipe.unet,
    get_pipeline_config=lambda pipe: pipe.unet.config,
))


# ---------------------------------------------------------------------------
#  SD3.5
# ---------------------------------------------------------------------------

def _sd35_load_hf_denoiser(model_id: str):
    from diffusers import SD3Transformer2DModel
    hf = SD3Transformer2DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.float32,
    ).to(DEVICE).eval()
    return hf, hf.config


def _sd35_build_kb(hf_config):
    kb_mod = importlib.import_module("KernelBench.level4.18_StableDiffusion35")
    return kb_mod.StableDiffusion35(
        sample_size=hf_config.sample_size,
        patch_size=hf_config.patch_size,
        in_channels=hf_config.in_channels,
        num_layers=hf_config.num_layers,
        attention_head_dim=hf_config.attention_head_dim,
        num_attention_heads=hf_config.num_attention_heads,
        joint_attention_dim=hf_config.joint_attention_dim,
        caption_projection_dim=hf_config.caption_projection_dim,
        pooled_projection_dim=hf_config.pooled_projection_dim,
        out_channels=hf_config.out_channels,
        pos_embed_max_size=hf_config.pos_embed_max_size,
        dual_attention_layers=tuple(hf_config.dual_attention_layers),
        qk_norm=hf_config.qk_norm,
    )


def _sd35_make_inputs(batch, h, w, hf_config, ts_val, device, dtype):
    hidden_states = torch.randn(batch, 16, h, w, device=device, dtype=dtype)
    enc_hs = torch.randn(batch, 77, hf_config.joint_attention_dim,
                         device=device, dtype=dtype)
    pooled = torch.randn(batch, hf_config.pooled_projection_dim,
                         device=device, dtype=dtype)
    timestep = torch.tensor([ts_val] * batch, device=device, dtype=dtype)
    return dict(hidden_states=hidden_states,
                encoder_hidden_states=enc_hs,
                pooled_projections=pooled,
                timestep=timestep)


def _sd35_forward(model, inputs):
    return model(**inputs, return_dict=False)[0]


def _sd35_load_pipeline(model_id):
    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id, torch_dtype=torch.float16)
    return pipe.to(DEVICE)


def _sd35_build_kb_pipeline(pipe_hf, kb_denoiser):
    kb_pipe_mod = importlib.import_module("KernelBench.level4.18_StableDiffusion35")
    kb_clip_mod = importlib.import_module("KernelBench.level3.encoder._5_CLIPTextEncoder")
    kb_vae_mod = importlib.import_module("KernelBench.level3.vae._4_VAEDecoder")
    kb_t5_mod = importlib.import_module("KernelBench.level3.encoder._6_T5Encoder")

    # --- KB text_encoder (CLIPTextModelWithProjection) ---
    hf_te1 = pipe_hf.text_encoder
    te1_cfg = hf_te1.config
    kb_te1 = kb_clip_mod.CLIPTextModelWithProjection(
        vocab_size=te1_cfg.vocab_size,
        hidden_size=te1_cfg.hidden_size,
        intermediate_size=te1_cfg.intermediate_size,
        num_hidden_layers=te1_cfg.num_hidden_layers,
        num_attention_heads=te1_cfg.num_attention_heads,
        max_position_embeddings=te1_cfg.max_position_embeddings,
        hidden_act=te1_cfg.hidden_act,
        layer_norm_eps=te1_cfg.layer_norm_eps,
        projection_dim=te1_cfg.projection_dim,
    )
    copied, missing, mismatch, extra = _copy_weights(hf_te1, kb_te1)
    assert len(missing) == 0, f"text_encoder missing: {missing[:5]}"
    assert len(mismatch) == 0, f"text_encoder mismatch: {mismatch[:5]}"
    kb_te1 = kb_te1.to(device=DEVICE, dtype=torch.float16).eval()

    # --- KB text_encoder_2 (CLIPTextModelWithProjection) ---
    hf_te2 = pipe_hf.text_encoder_2
    te2_cfg = hf_te2.config
    kb_te2 = kb_clip_mod.CLIPTextModelWithProjection(
        vocab_size=te2_cfg.vocab_size,
        hidden_size=te2_cfg.hidden_size,
        intermediate_size=te2_cfg.intermediate_size,
        num_hidden_layers=te2_cfg.num_hidden_layers,
        num_attention_heads=te2_cfg.num_attention_heads,
        max_position_embeddings=te2_cfg.max_position_embeddings,
        hidden_act=te2_cfg.hidden_act,
        layer_norm_eps=te2_cfg.layer_norm_eps,
        projection_dim=te2_cfg.projection_dim,
    )
    copied, missing, mismatch, extra = _copy_weights(hf_te2, kb_te2)
    assert len(missing) == 0, f"text_encoder_2 missing: {missing[:5]}"
    assert len(mismatch) == 0, f"text_encoder_2 mismatch: {mismatch[:5]}"
    kb_te2 = kb_te2.to(device=DEVICE, dtype=torch.float16).eval()

    # --- KB text_encoder_3 (T5Encoder) ---
    hf_te3 = pipe_hf.text_encoder_3
    te3_cfg = hf_te3.config
    kb_te3 = kb_t5_mod.T5Encoder(
        vocab_size=te3_cfg.vocab_size,
        d_model=te3_cfg.d_model,
        d_kv=te3_cfg.d_kv,
        d_ff=te3_cfg.d_ff,
        num_heads=te3_cfg.num_heads,
        num_layers=te3_cfg.num_layers,
        relative_attention_num_buckets=te3_cfg.relative_attention_num_buckets,
        relative_attention_max_distance=te3_cfg.relative_attention_max_distance,
        dropout_rate=te3_cfg.dropout_rate,
        layer_norm_epsilon=te3_cfg.layer_norm_epsilon,
    )
    copied, missing, mismatch, extra = _copy_weights(hf_te3, kb_te3)
    assert len(missing) == 0, f"text_encoder_3 missing: {missing[:5]}"
    assert len(mismatch) == 0, f"text_encoder_3 mismatch: {mismatch[:5]}"
    kb_te3 = kb_te3.to(device=DEVICE, dtype=torch.float16).eval()

    # --- KB VAE decoder ---
    hf_vae = pipe_hf.vae
    vae_cfg = hf_vae.config
    kb_vae = kb_vae_mod.VAEDecoder(
        latent_channels=vae_cfg.latent_channels,
        out_channels=vae_cfg.out_channels,
        block_out_channels=tuple(vae_cfg.block_out_channels),
        layers_per_block=vae_cfg.layers_per_block,
        norm_num_groups=vae_cfg.norm_num_groups,
        scaling_factor=vae_cfg.scaling_factor,
        shift_factor=getattr(vae_cfg, "shift_factor", None),
        force_upcast=getattr(vae_cfg, "force_upcast", True),
        use_post_quant_conv=getattr(vae_cfg, "use_post_quant_conv", False),
    )
    copied, missing, mismatch, extra = _copy_weights(hf_vae, kb_vae)
    assert len(missing) == 0, f"VAE missing: {missing[:5]}"
    assert len(mismatch) == 0, f"VAE mismatch: {mismatch[:5]}"
    kb_vae = kb_vae.to(device=DEVICE, dtype=torch.float16).eval()

    return kb_pipe_mod.StableDiffusion3Pipeline(
        transformer=kb_denoiser,
        scheduler=pipe_hf.scheduler,
        vae=kb_vae,
        text_encoder=kb_te1,
        tokenizer=pipe_hf.tokenizer,
        text_encoder_2=kb_te2,
        tokenizer_2=pipe_hf.tokenizer_2,
        text_encoder_3=kb_te3,
        tokenizer_3=pipe_hf.tokenizer_3,
    ).to(DEVICE)


_register(ModelSpec(
    model_id="stabilityai/stable-diffusion-3.5-large",
    short_name="sd35",
    rtol_mean=5e-3,
    max_abs_diff=5e-2,
    e2e_guidance_scale=7.0,
    e2e_default_height=512,
    e2e_default_width=512,
    e2e_native_height=None,       # no special high-res test
    e2e_native_width=None,
    e2e_seeds=[0, 99, 2024],
    timestep_values=[0.0, 100.0, 500.0, 900.0, 999.0],
    load_hf_denoiser=_sd35_load_hf_denoiser,
    build_kb_denoiser=_sd35_build_kb,
    make_inputs=_sd35_make_inputs,
    forward_fn=_sd35_forward,
    get_out_channels=lambda cfg: cfg.out_channels,
    load_hf_pipeline=_sd35_load_pipeline,
    build_kb_pipeline=_sd35_build_kb_pipeline,
    get_pipeline_denoiser=lambda pipe: pipe.transformer,
    get_pipeline_config=lambda pipe: pipe.transformer.config,
))


# ###########################################################################
#  Shared helpers
# ###########################################################################

def _kb_key_to_hf_key(kb_key: str) -> str:
    """Map a KernelBench state_dict key to the corresponding HuggingFace key.

    KB wraps some nn modules inside level1 operators which adds an extra
    segment to the parameter path:
      - GroupNorm   (level1) stores self.gn        -> adds ".gn."
      - LayerNorm   (level1) stores self.ln        -> adds ".ln."
      - Conv2d      (level1) stores self.conv2d    -> adds ".conv2d."
      - Embedding   (level1) stores self.embedding -> adds ".embedding."

    This function strips the wrapper segments to recover the HF key.
    """
    for wrapper_seg in (".conv2d.", ".gn.", ".ln.", ".embedding."):
        if wrapper_seg in kb_key:
            kb_key = kb_key.replace(wrapper_seg, ".")
            break
    return kb_key


def _copy_weights(src_model, dst_model):
    """Copy state_dict from *src_model* (HF) to *dst_model* (KB).

    Returns (copied, missing_in_src, shape_mismatch, extra_in_src).
    """
    src_sd = src_model.state_dict()
    dst_sd = dst_model.state_dict()

    copied = 0
    missing_in_src: List[str] = []
    shape_mismatch: List[str] = []
    matched_src_keys: set = set()

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
                    f"{kb_key} -> {hf_key} "
                    f"(dst={dst_tensor.shape} vs src={src_tensor.shape})")
        else:
            missing_in_src.append(f"{kb_key} (tried HF key: {hf_key})")

    extra_in_src = [k for k in src_sd if k not in matched_src_keys]
    dst_model.load_state_dict(dst_sd)

    print(f"  Copied {copied} weights")
    if missing_in_src:
        print(f"  Missing in source ({len(missing_in_src)}): "
              f"{missing_in_src[:10]}")
    if shape_mismatch:
        print(f"  Shape mismatches ({len(shape_mismatch)}): "
              f"{shape_mismatch[:10]}")
    if extra_in_src:
        print(f"  Extra in source ({len(extra_in_src)}): "
              f"{extra_in_src[:10]}")

    return copied, missing_in_src, shape_mismatch, extra_in_src


def _compare_tensors(hf_tensor, kb_tensor):
    """Compute abs/rel diff stats between two tensors."""
    hf_f = hf_tensor.float()
    kb_f = kb_tensor.float()
    abs_diff = (hf_f - kb_f).abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    denom = torch.maximum(hf_f.abs(), kb_f.abs()) + 1e-8
    rel_diff = abs_diff / denom
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "max_rel": rel_diff.max().item(),
        "mean_rel": rel_diff.mean().item(),
    }


def _pil_to_np(img) -> np.ndarray:
    return np.array(img).astype(np.float32)


def _compare_images(img_a, img_b, tag: str = ""):
    arr_a, arr_b = _pil_to_np(img_a), _pil_to_np(img_b)
    assert arr_a.shape == arr_b.shape, \
        f"Shape mismatch: {arr_a.shape} vs {arr_b.shape}"
    diff = arr_a - arr_b
    mse = float(np.mean(diff ** 2))
    max_diff = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    print(f"  [{tag}] shape={arr_a.shape}  pixel MSE={mse:.4f}  "
          f"mean_abs={mean_abs:.4f}  max_diff={max_diff:.1f}")
    return mse, max_diff


def _save_image(img, name: str, save: bool = False):
    if save:
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
#  Resolve the active ModelSpec from the CLI option
# ###########################################################################

def _get_spec(request) -> ModelSpec:
    """Resolve the ModelSpec from ``--model-name``."""
    model_name = request.config.getoption("--model-name")
    if model_name not in MODEL_REGISTRY:
        supported = "\n  ".join(sorted(MODEL_REGISTRY.keys()))
        pytest.fail(
            f"Unsupported diffusion model: {model_name}\n"
            f"Supported models:\n  {supported}\n"
            f"Register a new ModelSpec in test_sd_hf_alignment.py to add it.")
    return MODEL_REGISTRY[model_name]


# ###########################################################################
#  Fixtures
# ###########################################################################

@pytest.fixture(scope="module")
def spec(request) -> ModelSpec:
    """The active ModelSpec for this test run."""
    return _get_spec(request)


@pytest.fixture(scope="module")
def save_images(request) -> bool:
    """Whether to save generated images (``--save-images`` flag)."""
    return request.config.getoption("--save-images")


@pytest.fixture(scope="module")
def denoiser_models(spec):
    """Load HF and KB denoisers in float32 with copied weights."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    print(f"\nLoading {spec.short_name} denoiser from {spec.model_id} "
          f"(float32) ...")

    hf_model, hf_config = spec.load_hf_denoiser(spec.model_id)
    print(f"  HF denoiser: "
          f"{sum(p.numel() for p in hf_model.parameters()):,} params")

    kb_model = spec.build_kb_denoiser(hf_config)
    kb_model = kb_model.to(device=DEVICE, dtype=torch.float32)

    copied, missing, mismatched, extra = _copy_weights(hf_model, kb_model)
    kb_model.eval()
    print(f"  KB denoiser: "
          f"{sum(p.numel() for p in kb_model.parameters()):,} params")

    return hf_model, kb_model, hf_config, copied, missing, mismatched, extra


@pytest.fixture(scope="module")
def pipelines(spec):
    """Load HF pipeline in fp16, build a second with KB denoiser swapped in."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    print(f"\nLoading {spec.short_name} pipeline from {spec.model_id} "
          f"(fp16) ...")

    pipe_hf = spec.load_hf_pipeline(spec.model_id)
    hf_denoiser = spec.get_pipeline_denoiser(pipe_hf)
    print(f"  HF pipeline loaded  "
          f"(denoiser: {sum(p.numel() for p in hf_denoiser.parameters()):,} "
          f"params)")

    hf_config = spec.get_pipeline_config(pipe_hf)
    kb_denoiser = spec.build_kb_denoiser(hf_config)

    copied, missing, mismatched, extra = _copy_weights(
        hf_denoiser, kb_denoiser)
    assert len(missing) == 0, f"Missing weights: {missing[:10]}"
    assert len(mismatched) == 0, f"Shape mismatches: {mismatched[:10]}"
    assert len(extra) == 0, f"Extra HF weights: {extra[:10]}"

    kb_denoiser = kb_denoiser.to(device=DEVICE, dtype=torch.float16).eval()
    print(f"  KB denoiser created & weights copied  "
          f"({sum(p.numel() for p in kb_denoiser.parameters()):,} params)")

    pipe_kb = spec.build_kb_pipeline(pipe_hf, kb_denoiser)
    print("  KB pipeline assembled (KB denoiser + KB text encoders + KB VAE)")

    return pipe_hf, pipe_kb


# ###########################################################################
#  Model-Level Alignment Tests (float32)
# ###########################################################################

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestModelAlignment:
    """Compare denoiser forward passes with random inputs in float32."""

    def test_weight_transfer(self, spec, denoiser_models):
        """All weights transfer: no missing keys, no shape mismatches."""
        hf_model, kb_model, _, copied, missing, mismatched, extra = \
            denoiser_models
        hf_n = sum(p.numel() for p in hf_model.parameters())
        kb_n = sum(p.numel() for p in kb_model.parameters())
        print(f"\n  HF params: {hf_n:,}  KB params: {kb_n:,}  "
              f"copied: {copied}")
        assert len(missing) == 0, f"Missing: {missing[:10]}"
        assert len(mismatched) == 0, f"Mismatched: {mismatched[:10]}"
        assert hf_n == kb_n, f"Param count: HF={hf_n:,} vs KB={kb_n:,}"

    def test_forward(self, spec, denoiser_models):
        """Single forward pass produces matching outputs."""
        hf_model, kb_model, hf_config, *_ = denoiser_models
        torch.manual_seed(42)
        inputs = spec.make_inputs(1, 64, 64, hf_config,
                                  spec.timestep_values[len(spec.timestep_values) // 2],
                                  DEVICE, torch.float32)
        with torch.no_grad():
            hf_out = spec.forward_fn(hf_model, inputs)
            kb_out = spec.forward_fn(kb_model, inputs)
        assert hf_out.shape == kb_out.shape
        m = _compare_tensors(hf_out, kb_out)
        print(f"\n  abs: max={m['max_abs']:.2e} mean={m['mean_abs']:.2e}  "
              f"rel: max={m['max_rel']:.2e} mean={m['mean_rel']:.2e}")
        assert m["mean_rel"] < spec.rtol_mean
        assert m["max_abs"] < spec.max_abs_diff

    def test_multiple_timesteps(self, spec, denoiser_models):
        """Alignment holds across multiple timestep values."""
        hf_model, kb_model, hf_config, *_ = denoiser_models
        torch.manual_seed(123)
        # Build inputs once with a dummy timestep; we'll override per iteration
        base_inputs = spec.make_inputs(1, 64, 64, hf_config, 0, DEVICE,
                                       torch.float32)
        all_pass = True
        for t_val in spec.timestep_values:
            # Replace the timestep in the inputs dict
            inputs = dict(base_inputs)
            ts_key = "timestep"
            if ts_key in inputs:
                old_ts = inputs[ts_key]
                if old_ts.dtype in (torch.long, torch.int32, torch.int64):
                    inputs[ts_key] = torch.tensor(
                        [int(t_val)], device=DEVICE, dtype=old_ts.dtype)
                else:
                    inputs[ts_key] = torch.tensor(
                        [t_val] * old_ts.shape[0], device=DEVICE,
                        dtype=old_ts.dtype)
            with torch.no_grad():
                hf_out = spec.forward_fn(hf_model, inputs)
                kb_out = spec.forward_fn(kb_model, inputs)
            m = _compare_tensors(hf_out, kb_out)
            ok = (m["mean_rel"] < spec.rtol_mean
                  and m["max_abs"] < spec.max_abs_diff)
            if not ok:
                all_pass = False
            print(f"  t={t_val!s:>6s}: {'PASS' if ok else 'FAIL'}  "
                  f"abs max={m['max_abs']:.2e}  rel mean={m['mean_rel']:.2e}")
        assert all_pass, "Some timesteps failed"

    def test_batch(self, spec, denoiser_models):
        """Alignment with batch size 2."""
        hf_model, kb_model, hf_config, *_ = denoiser_models
        torch.manual_seed(456)
        inputs = spec.make_inputs(2, 64, 64, hf_config,
                                  spec.timestep_values[len(spec.timestep_values) // 2],
                                  DEVICE, torch.float32)
        with torch.no_grad():
            hf_out = spec.forward_fn(hf_model, inputs)
            kb_out = spec.forward_fn(kb_model, inputs)
        m = _compare_tensors(hf_out, kb_out)
        print(f"\n  shape={hf_out.shape}  abs max={m['max_abs']:.2e}  "
              f"rel mean={m['mean_rel']:.2e}")
        assert m["mean_rel"] < spec.rtol_mean
        assert m["max_abs"] < spec.max_abs_diff

    def test_resolutions(self, spec, denoiser_models):
        """Alignment at multiple spatial sizes."""
        hf_model, kb_model, hf_config, *_ = denoiser_models
        all_pass = True
        for h, w in spec.resolution_grid:
            torch.manual_seed(789)
            inputs = spec.make_inputs(
                1, h, w, hf_config,
                spec.timestep_values[len(spec.timestep_values) // 2],
                DEVICE, torch.float32)
            with torch.no_grad():
                hf_out = spec.forward_fn(hf_model, inputs)
                kb_out = spec.forward_fn(kb_model, inputs)
            m = _compare_tensors(hf_out, kb_out)
            ok = (m["mean_rel"] < spec.rtol_mean
                  and m["max_abs"] < spec.max_abs_diff)
            if not ok:
                all_pass = False
            print(f"  {h}x{w}: {'PASS' if ok else 'FAIL'}  "
                  f"abs max={m['max_abs']:.2e}  rel mean={m['mean_rel']:.2e}")
        assert all_pass, "Some resolutions failed"

    def test_output_shape(self, spec, denoiser_models):
        """Output shape is (B, out_channels, H, W), no NaN/Inf."""
        _, kb_model, hf_config, *_ = denoiser_models
        torch.manual_seed(0)
        inputs = spec.make_inputs(1, 64, 64, hf_config,
                                  spec.timestep_values[len(spec.timestep_values) // 2],
                                  DEVICE, torch.float32)
        with torch.no_grad():
            out = spec.forward_fn(kb_model, inputs)
        expected = (1, spec.get_out_channels(hf_config), 64, 64)
        assert out.shape == expected, f"Expected {expected}, got {out.shape}"
        assert not torch.isnan(out).any(), "NaN in output"
        assert not torch.isinf(out).any(), "Inf in output"

    def test_deterministic(self, spec, denoiser_models):
        """Identical inputs produce identical outputs across two runs."""
        _, kb_model, hf_config, *_ = denoiser_models
        torch.manual_seed(42)
        inputs = spec.make_inputs(1, 64, 64, hf_config,
                                  spec.timestep_values[len(spec.timestep_values) // 2],
                                  DEVICE, torch.float32)
        with torch.no_grad():
            out1 = spec.forward_fn(kb_model, inputs)
            out2 = spec.forward_fn(kb_model, inputs)
        diff = (out1 - out2).abs().max().item()
        print(f"\n  Max diff between two runs: {diff:.2e}")
        assert diff == 0.0 or diff < 1e-6


# ###########################################################################
#  End-to-End Pipeline Tests (fp16)
# ###########################################################################

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestE2E:
    """Full pipeline: text encode -> denoise -> VAE decode -> compare images."""

    def test_astronaut(self, spec, pipelines, save_images):
        pipe_hf, pipe_kb = pipelines
        prompt = "An astronaut riding a green horse"
        print(f"\n  Prompt: \"{prompt}\"")
        img_hf, img_kb = _generate_pair(
            pipe_hf, pipe_kb, prompt=prompt, seed=42, num_steps=20,
            height=spec.e2e_default_height, width=spec.e2e_default_width,
            guidance_scale=spec.e2e_guidance_scale)
        _save_image(img_hf, f"{spec.short_name}_astronaut_hf.png", save_images)
        _save_image(img_kb, f"{spec.short_name}_astronaut_kb.png", save_images)
        mse, max_diff = _compare_images(img_hf, img_kb, tag="astronaut")
        assert mse < PIXEL_MSE_THRESHOLD
        assert max_diff < PIXEL_MAX_DIFF_THRESHOLD

    def test_landscape(self, spec, pipelines, save_images):
        pipe_hf, pipe_kb = pipelines
        prompt = "A beautiful sunset over a mountain lake, photorealistic, 8k"
        print(f"\n  Prompt: \"{prompt}\"")
        img_hf, img_kb = _generate_pair(
            pipe_hf, pipe_kb, prompt=prompt, seed=123, num_steps=20,
            height=spec.e2e_default_height, width=spec.e2e_default_width,
            guidance_scale=spec.e2e_guidance_scale)
        _save_image(img_hf, f"{spec.short_name}_landscape_hf.png", save_images)
        _save_image(img_kb, f"{spec.short_name}_landscape_kb.png", save_images)
        mse, max_diff = _compare_images(img_hf, img_kb, tag="landscape")
        assert mse < PIXEL_MSE_THRESHOLD
        assert max_diff < PIXEL_MAX_DIFF_THRESHOLD

    def test_portrait(self, spec, pipelines, save_images):
        pipe_hf, pipe_kb = pipelines
        prompt = "Portrait of a cat wearing a tiny top hat, studio lighting"
        print(f"\n  Prompt: \"{prompt}\"")
        img_hf, img_kb = _generate_pair(
            pipe_hf, pipe_kb, prompt=prompt, seed=7, num_steps=20,
            height=spec.e2e_default_height, width=spec.e2e_default_width,
            guidance_scale=spec.e2e_guidance_scale)
        _save_image(img_hf, f"{spec.short_name}_portrait_hf.png", save_images)
        _save_image(img_kb, f"{spec.short_name}_portrait_kb.png", save_images)
        mse, max_diff = _compare_images(img_hf, img_kb, tag="portrait")
        assert mse < PIXEL_MSE_THRESHOLD
        assert max_diff < PIXEL_MAX_DIFF_THRESHOLD

    def test_high_res(self, spec, pipelines, save_images):
        """Native high-resolution generation (model-specific, skipped if N/A)."""
        if spec.e2e_native_height is None:
            pytest.skip(f"No native high-res test for {spec.short_name}")
        pipe_hf, pipe_kb = pipelines
        prompt = "An astronaut riding a green horse"
        h, w = spec.e2e_native_height, spec.e2e_native_width
        print(f"\n  Prompt: \"{prompt}\" ({h}x{w})")
        img_hf, img_kb = _generate_pair(
            pipe_hf, pipe_kb, prompt=prompt, seed=42, num_steps=20,
            height=h, width=w, guidance_scale=spec.e2e_guidance_scale)
        _save_image(img_hf, f"{spec.short_name}_highres_hf.png", save_images)
        _save_image(img_kb, f"{spec.short_name}_highres_kb.png", save_images)
        mse, max_diff = _compare_images(img_hf, img_kb,
                                         tag=f"{h}x{w}")
        assert mse < PIXEL_MSE_THRESHOLD
        assert max_diff < PIXEL_MAX_DIFF_THRESHOLD

    def test_different_seeds(self, spec, pipelines, save_images):
        pipe_hf, pipe_kb = pipelines
        prompt = "An astronaut riding a green horse"
        print(f"\n  Prompt: \"{prompt}\" (multiple seeds)")
        all_pass = True
        for seed in spec.e2e_seeds:
            img_hf, img_kb = _generate_pair(
                pipe_hf, pipe_kb, prompt=prompt, seed=seed, num_steps=15,
                height=spec.e2e_default_height, width=spec.e2e_default_width,
                guidance_scale=spec.e2e_guidance_scale)
            _save_image(img_hf, f"{spec.short_name}_seed{seed}_hf.png",
                        save_images)
            _save_image(img_kb, f"{spec.short_name}_seed{seed}_kb.png",
                        save_images)
            mse, max_diff = _compare_images(img_hf, img_kb,
                                             tag=f"seed={seed}")
            if (mse >= PIXEL_MSE_THRESHOLD
                    or max_diff >= PIXEL_MAX_DIFF_THRESHOLD):
                all_pass = False
        assert all_pass, "One or more seeds failed pixel alignment"

    def test_negative_prompt(self, spec, pipelines, save_images):
        pipe_hf, pipe_kb = pipelines
        prompt = "A futuristic city skyline at night, neon lights, cyberpunk"
        negative_prompt = "blurry, low quality, watermark"
        print(f"\n  Prompt: \"{prompt}\"")
        print(f"  Negative: \"{negative_prompt}\"")
        img_hf, img_kb = _generate_pair(
            pipe_hf, pipe_kb, prompt=prompt,
            negative_prompt=negative_prompt,
            seed=55, num_steps=20,
            height=spec.e2e_default_height, width=spec.e2e_default_width,
            guidance_scale=spec.e2e_guidance_scale)
        _save_image(img_hf, f"{spec.short_name}_cyberpunk_hf.png", save_images)
        _save_image(img_kb, f"{spec.short_name}_cyberpunk_kb.png", save_images)
        mse, max_diff = _compare_images(img_hf, img_kb,
                                         tag="negative_prompt")
        assert mse < PIXEL_MSE_THRESHOLD
        assert max_diff < PIXEL_MAX_DIFF_THRESHOLD
