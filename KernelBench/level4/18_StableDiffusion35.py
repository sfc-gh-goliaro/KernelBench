"""
Stable Diffusion 3.5 Large — SD3Transformer2DModel (MMDiT architecture)

Implements the SD3Transformer2DModel matching
stabilityai/stable-diffusion-3.5-large from HuggingFace diffusers.

Architecture (SD3.5-large config):
- 16-channel latent input/output (VAE with 16 channels)
- patch_size: 2
- inner_dim: 2432 (38 heads × 64 head_dim)
- num_layers: 38 JointTransformerBlocks
- joint_attention_dim: 4096 (T5-XXL encoder)
- caption_projection_dim: 2432
- pooled_projection_dim: 2048
- pos_embed_max_size: 192
- qk_norm: "rms_norm" (RMSNorm on Q/K before attention)
- dual_attention_layers: () (no dual attention in SD3.5-large)

Key differences from SDXL (UNet):
- Transformer-based (not UNet)
- MMDiT: joint attention between image tokens and text tokens
- Patch embedding: Conv2d patchify + 2D sincos positional encoding
- AdaLayerNormZero conditioning (shift/scale/gate from timestep embedding)
- AdaLayerNormContinuous for final output normalization
- GELU(approximate="tanh") activation in FeedForward, not GEGLU
- RMSNorm on Q/K (qk_norm)

The forward signature matches diffusers:
    forward(hidden_states, encoder_hidden_states, pooled_projections, timestep, ...)

This model delegates all primitive computations to level1 operators.
"""

import inspect
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import Optional, Dict, Any, Tuple, List, Union

# ============================================================================
# Level1 operator imports
# ============================================================================

# Parameter-owning operators
from KernelBench.level1.normalization._6_LayerNorm import Model as LayerNorm
from KernelBench.level1.normalization._4_RMSNorm import Model as RMSNorm
from KernelBench.level1.matmul._10_Linear import Model as Linear
from KernelBench.level1.vision._1_PatchEmbed2D import Model as PatchEmbed2D

# Diffusion-specific operators
from KernelBench.level1.diffusion._1_AdaLN import Model as AdaLNContinuous
from KernelBench.level1.diffusion._2_AdaLN_Zero import Model as AdaLNZero
from KernelBench.level1.diffusion._3_TimestepEmbedding import Model as TimestepEmbedding
from KernelBench.level1.diffusion._7_SinusoidalTimesteps import Model as SinusoidalTimesteps

# Parameter-free operators
from KernelBench.level1.activations._7_Swish import Model as Swish
from KernelBench.level1.activations._8_GELU import Model as GELUAct
from KernelBench.level1.attention._2_Attention import ScaledDotProductAttention
from KernelBench.level1.regularization._1_Dropout import Model as Dropout


# ============================================================================
# 2D sinusoidal positional embedding (matches diffusers get_2d_sincos_pos_embed)
# ============================================================================

def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """1D sincos positional embedding from a grid of positions."""
    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=torch.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega

    pos = pos.reshape(-1)
    out = torch.outer(pos.to(torch.float64), omega)
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
    return emb


def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, base_size: int = 16,
                              interpolation_scale: float = 1.0) -> torch.Tensor:
    """2D sincos positional embedding matching diffusers PatchEmbed."""
    grid_h = torch.arange(grid_size, dtype=torch.float32) / (grid_size / base_size) / interpolation_scale
    grid_w = torch.arange(grid_size, dtype=torch.float32) / (grid_size / base_size) / interpolation_scale
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0).reshape(2, 1, grid_size, grid_size)

    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0].reshape(-1))
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1].reshape(-1))
    emb = torch.cat([emb_h, emb_w], dim=1)
    return emb


# ============================================================================
# PatchEmbed (matches diffusers PatchEmbed for SD3)
# ============================================================================

class PatchEmbed(PatchEmbed2D):
    """SD3 PatchEmbed = PatchEmbed2D (level1) + cropped 2D sincos positional encoding.

    Inherits Conv2d + flatten + transpose from PatchEmbed2D, adds SD3-specific
    pre-computed positional embeddings with center-crop at runtime."""

    def __init__(self, height: int = 128, width: int = 128, patch_size: int = 2,
                 in_channels: int = 16, embed_dim: int = 2432,
                 pos_embed_max_size: int = 192):
        super().__init__(img_size=height, patch_size=patch_size,
                         in_channels=in_channels, embed_dim=embed_dim,
                         flatten=True, bias=True, proj_name="proj")
        self.height = height // patch_size
        self.width = width // patch_size
        self.base_size = height // patch_size
        self.pos_embed_max_size = pos_embed_max_size

        # Pre-compute positional embeddings for max size
        pos_embed = _get_2d_sincos_pos_embed(embed_dim, pos_embed_max_size,
                                              base_size=self.base_size)
        self.register_buffer("pos_embed", pos_embed.float().unsqueeze(0), persistent=True)

    def cropped_pos_embed(self, height: int, width: int) -> torch.Tensor:
        """Crop positional embeddings to match input spatial size."""
        height = height // self.patch_size
        width = width // self.patch_size
        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2
        spatial = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
        spatial = spatial[:, top:top + height, left:left + width, :]
        return spatial.reshape(1, -1, spatial.shape[-1])

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        height, width = latent.shape[-2:]
        latent, _ = super().forward(latent)  # PatchEmbed2D: Conv2d + flatten + transpose
        pos_embed = self.cropped_pos_embed(height, width)
        return (latent + pos_embed).to(latent.dtype)


# ============================================================================
# CombinedTimestepTextProjEmbeddings (matches diffusers)
# ============================================================================

class TextProjection(nn.Module):
    """Matches diffusers PixArtAlphaTextProjection with act_fn='silu'.
    Uses Linear + Swish level1 ops."""

    def __init__(self, in_features: int, hidden_size: int):
        super().__init__()
        self.linear_1 = Linear(in_features, hidden_size, bias=True)
        self.act_1 = Swish()
        self.linear_2 = Linear(hidden_size, hidden_size, bias=True)

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class CombinedTimestepTextProjEmbeddings(nn.Module):
    """Matches diffusers CombinedTimestepTextProjEmbeddings.
    Combines sinusoidal timestep embedding + pooled text projection.
    Uses TimestepEmbeddingOp level1 op for the MLP."""

    def __init__(self, embedding_dim: int, pooled_projection_dim: int):
        super().__init__()
        self.time_proj = SinusoidalTimesteps(num_channels=256, flip_sin_to_cos=True,
                                              downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256,
                                                    time_embed_dim=embedding_dim)
        self.text_embedder = TextProjection(pooled_projection_dim, embedding_dim)

    def forward(self, timestep: torch.Tensor,
                pooled_projection: torch.Tensor) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(
            timesteps_proj.to(dtype=pooled_projection.dtype))
        pooled_projections = self.text_embedder(pooled_projection)
        return timesteps_emb + pooled_projections


# ============================================================================
# Adaptive Layer Norms — delegated to level1 operators
# AdaLayerNormZero → AdaLNZero (level1/diffusion/_2_AdaLN_Zero)
# AdaLayerNormContinuous → AdaLNContinuous (level1/diffusion/_1_AdaLN)
# ============================================================================


# ============================================================================
# FeedForward (matches diffusers FeedForward with gelu-approximate)
# ============================================================================

class GELUProjection(nn.Module):
    """Matches diffusers GELU activation module: Linear + GELU(approximate='tanh').
    Uses Linear + GELUAct level1 ops."""

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = Linear(dim_in, dim_out, bias=bias)
        self.gelu = GELUAct(approximate='tanh')

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states


class FeedForward(nn.Module):
    """Matches diffusers FeedForward with activation_fn='gelu-approximate'.
    Uses level1 ops for all computation."""

    def __init__(self, dim: int, dim_out: Optional[int] = None, mult: int = 4,
                 dropout: float = 0.0, bias: bool = True):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        self.net = nn.ModuleList([
            GELUProjection(dim, inner_dim, bias=bias),
            Dropout(p=dropout),
            Linear(inner_dim, dim_out, bias=bias),
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


# ============================================================================
# Joint Attention (matches diffusers JointAttnProcessor2_0 + Attention)
# ============================================================================

class JointAttention(nn.Module):
    """
    Joint attention for MMDiT: concatenates image and text tokens for shared
    attention, then splits outputs.

    Uses Linear, RMSNorm, ScaledDotProductAttention, Dropout level1 ops.
    """

    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int,
                 context_pre_only: bool = False, qk_norm: Optional[str] = None):
        super().__init__()
        self.heads = num_attention_heads
        self.head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.context_pre_only = context_pre_only

        # Image projections
        self.to_q = Linear(dim, self.inner_dim, bias=True)
        self.to_k = Linear(dim, self.inner_dim, bias=True)
        self.to_v = Linear(dim, self.inner_dim, bias=True)

        # Context (text) projections
        self.add_q_proj = Linear(dim, self.inner_dim, bias=True)
        self.add_k_proj = Linear(dim, self.inner_dim, bias=True)
        self.add_v_proj = Linear(dim, self.inner_dim, bias=True)

        # QK normalization
        if qk_norm == "rms_norm":
            self.norm_q = RMSNorm(attention_head_dim, eps=1e-6, learnable_weight=True, dim=-1)
            self.norm_k = RMSNorm(attention_head_dim, eps=1e-6, learnable_weight=True, dim=-1)
            self.norm_added_q = RMSNorm(attention_head_dim, eps=1e-6, learnable_weight=True, dim=-1)
            self.norm_added_k = RMSNorm(attention_head_dim, eps=1e-6, learnable_weight=True, dim=-1)
        else:
            self.norm_q = None
            self.norm_k = None
            self.norm_added_q = None
            self.norm_added_k = None

        # Output projections
        self.to_out = nn.ModuleList([
            Linear(self.inner_dim, dim, bias=True),
            Dropout(p=0.0),
        ])

        if not context_pre_only:
            self.to_add_out = Linear(self.inner_dim, dim, bias=True)
        else:
            self.to_add_out = None

        # Attention kernel
        self.sdpa = ScaledDotProductAttention(mode="sdpa")

    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: Optional[torch.Tensor] = None):
        batch_size = hidden_states.shape[0]
        head_dim = self.head_dim

        # Image Q/K/V
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        # Context Q/K/V (joint attention)
        if encoder_hidden_states is not None:
            ctx_query = self.add_q_proj(encoder_hidden_states)
            ctx_key = self.add_k_proj(encoder_hidden_states)
            ctx_value = self.add_v_proj(encoder_hidden_states)

            ctx_query = ctx_query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            ctx_key = ctx_key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            ctx_value = ctx_value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

            if self.norm_added_q is not None:
                ctx_query = self.norm_added_q(ctx_query)
            if self.norm_added_k is not None:
                ctx_key = self.norm_added_k(ctx_key)

            # Concatenate image and context tokens for joint attention
            query = torch.cat([query, ctx_query], dim=2)
            key = torch.cat([key, ctx_key], dim=2)
            value = torch.cat([value, ctx_value], dim=2)

        # Attention
        hidden_out = self.sdpa(query, key, value)
        hidden_out = hidden_out.transpose(1, 2).reshape(batch_size, -1, self.inner_dim)
        hidden_out = hidden_out.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split image and context outputs
            residual_len = hidden_states.shape[1]
            hidden_out, encoder_out = hidden_out[:, :residual_len], hidden_out[:, residual_len:]

            if not self.context_pre_only:
                encoder_out = self.to_add_out(encoder_out)

        # Output projection for image
        for module in self.to_out:
            hidden_out = module(hidden_out)

        if encoder_hidden_states is not None:
            return hidden_out, encoder_out
        else:
            return hidden_out


# ============================================================================
# JointTransformerBlock (matches diffusers JointTransformerBlock for SD3)
# ============================================================================

class JointTransformerBlock(nn.Module):
    """
    MMDiT transformer block with joint image-text attention.

    For SD3.5-large: no dual attention, qk_norm='rms_norm'.
    Last block has context_pre_only=True (no context FFN/residual).
    """

    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int,
                 context_pre_only: bool = False, qk_norm: Optional[str] = None,
                 use_dual_attention: bool = False):
        super().__init__()
        self.use_dual_attention = use_dual_attention
        self.context_pre_only = context_pre_only

        # Image norm (AdaLN-Zero) — level1 op with 6 output chunks
        self.norm1 = AdaLNZero(dim, num_output_chunks=6)

        # Context norm
        if context_pre_only:
            self.norm1_context = AdaLNContinuous(
                dim, dim, elementwise_affine=False, eps=1e-6, bias=True)
        else:
            self.norm1_context = AdaLNZero(dim, num_output_chunks=6)

        # Joint attention
        self.attn = JointAttention(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            context_pre_only=context_pre_only,
            qk_norm=qk_norm,
        )

        # Image FFN
        self.norm2 = LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.ff = FeedForward(dim=dim, dim_out=dim)

        # Context FFN (not present in last block)
        if not context_pre_only:
            self.norm2_context = LayerNorm(dim, eps=1e-6, elementwise_affine=False)
            self.ff_context = FeedForward(dim=dim, dim_out=dim)
        else:
            self.norm2_context = None
            self.ff_context = None

    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                temb: torch.Tensor,
                joint_attention_kwargs: Optional[Dict[str, Any]] = None):
        # 1. Norms
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb)

        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
        else:
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = \
                self.norm1_context(encoder_hidden_states, emb=temb)

        # 2. Joint attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
        )

        # 3. Image: attention residual + FFN
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output

        # 4. Context: attention residual + FFN (unless context_pre_only)
        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder = self.norm2_context(encoder_hidden_states)
            norm_encoder = norm_encoder * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            context_ff = self.ff_context(norm_encoder)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff

        return encoder_hidden_states, hidden_states


# ============================================================================
# Main Model: SD3Transformer2DModel
# ============================================================================

class StableDiffusion35(nn.Module):
    """
    SD3Transformer2DModel matching stabilityai/stable-diffusion-3.5-large.

    MMDiT (Multimodal Diffusion Transformer) architecture with:
    - Patch embedding for latent images
    - Joint image-text attention blocks
    - AdaLayerNorm conditioning from timestep + pooled text
    - RMSNorm on Q/K
    - GELU(tanh) feed-forward networks

    All primitive computations are delegated to level1 operators.
    """

    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 38,
        attention_head_dim: int = 64,
        num_attention_heads: int = 38,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 2432,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 192,
        dual_attention_layers: Tuple[int, ...] = (),
        qk_norm: Optional[str] = "rms_norm",
    ):
        super().__init__()
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        # Store config for pipeline compatibility
        self._config = type("Config", (), {
            "sample_size": sample_size,
            "patch_size": patch_size,
            "in_channels": in_channels,
            "num_layers": num_layers,
            "attention_head_dim": attention_head_dim,
            "num_attention_heads": num_attention_heads,
            "joint_attention_dim": joint_attention_dim,
            "caption_projection_dim": caption_projection_dim,
            "pooled_projection_dim": pooled_projection_dim,
            "out_channels": out_channels,
            "pos_embed_max_size": pos_embed_max_size,
            "dual_attention_layers": dual_attention_layers,
            "qk_norm": qk_norm,
        })()

        # Patch embedding
        self.pos_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,
        )

        # Timestep + pooled text conditioning
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim,
            pooled_projection_dim=pooled_projection_dim,
        )

        # Context (text) linear projection
        self.context_embedder = Linear(joint_attention_dim, caption_projection_dim, bias=True)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            JointTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                context_pre_only=(i == num_layers - 1),
                qk_norm=qk_norm,
                use_dual_attention=(i in dual_attention_layers),
            )
            for i in range(num_layers)
        ])

        # Final output: adaptive norm + linear projection (level1 AdaLN op)
        self.norm_out = AdaLNContinuous(
            self.inner_dim, self.inner_dim,
            elementwise_affine=False, eps=1e-6)
        self.proj_out = Linear(self.inner_dim,
                               patch_size * patch_size * self.out_channels, bias=True)

    @property
    def config(self):
        """Config object for pipeline compatibility."""
        return self._config

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the model parameters (for pipeline compatibility)."""
        return next(self.parameters()).dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
    ) -> Union[torch.Tensor, Tuple]:
        """
        Forward pass matching diffusers SD3Transformer2DModel.

        Args:
            hidden_states: Noisy latent (B, C, H, W)
            encoder_hidden_states: Text encoder hidden states (B, seq_len, joint_attention_dim)
            pooled_projections: Pooled text embeddings (B, pooled_projection_dim)
            timestep: Diffusion timestep
            return_dict: If True, return dict with 'sample' key

        Returns:
            If return_dict=False: (sample,) tuple
            If return_dict=True: dict with 'sample' key
        """
        height, width = hidden_states.shape[-2:]

        # 1. Patch embed + positional encoding
        hidden_states = self.pos_embed(hidden_states)

        # 2. Timestep + text conditioning
        temb = self.time_text_embed(timestep, pooled_projections)

        # 3. Project context
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # 4. Transformer blocks
        for index_block, block in enumerate(self.transformer_blocks):
            is_skip = skip_layers is not None and index_block in skip_layers
            if not is_skip:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # ControlNet residual
            if block_controlnet_hidden_states is not None and not block.context_pre_only:
                interval = len(self.transformer_blocks) / len(block_controlnet_hidden_states)
                hidden_states = hidden_states + block_controlnet_hidden_states[
                    int(index_block / interval)]

        # 5. Final norm + projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # 6. Unpatchify
        patch_size = self._config.patch_size
        p_height = height // patch_size
        p_width = width // patch_size

        hidden_states = hidden_states.reshape(
            hidden_states.shape[0], p_height, p_width, patch_size, patch_size, self.out_channels
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            hidden_states.shape[0], self.out_channels,
            p_height * patch_size, p_width * patch_size
        )

        if not return_dict:
            return (output,)

        return {"sample": output}


# Backward-compatible alias
Model = StableDiffusion35


# ============================================================================
# SD3.5 Pipeline — full text-to-image generation
# ============================================================================
# Implements the same logic as diffusers StableDiffusion3Pipeline.__call__
# but without inheriting from DiffusionPipeline.
#
# External HF components used:
#   - text_encoder / text_encoder_2  (CLIPTextModelWithProjection)
#   - text_encoder_3                 (T5EncoderModel)
#   - tokenizer / tokenizer_2       (CLIPTokenizer)
#   - tokenizer_3                   (T5TokenizerFast)
#   - vae                           (AutoencoderKL)
#   - scheduler                     (FlowMatchEulerDiscreteScheduler)
#
# The denoiser is our own StableDiffusion35 (SD3Transformer2DModel).
# ============================================================================


def _calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096,
                     base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _retrieve_timesteps(scheduler, num_inference_steps, device,
                        timesteps=None, sigmas=None, **kwargs):
    """Call scheduler.set_timesteps and return (timesteps, num_inference_steps)."""
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class _PipelineOutput:
    """Minimal output object with ``.images`` attribute for pipeline compatibility."""
    __slots__ = ("images",)

    def __init__(self, images):
        self.images = images


class StableDiffusion3Pipeline:
    """KernelBench SD3.5 text-to-image pipeline.

    Drop-in replacement for ``diffusers.StableDiffusion3Pipeline`` with the
    same ``__call__`` interface (subset of parameters that matter for
    generation).  Uses our :class:`StableDiffusion35` transformer as the
    denoiser.
    """

    def __init__(
        self,
        transformer: "StableDiffusion35",
        scheduler,
        vae,
        text_encoder,
        text_encoder_2,
        tokenizer,
        tokenizer_2,
        text_encoder_3=None,
        tokenizer_3=None,
    ):
        self.transformer = transformer
        self.scheduler = scheduler
        self.vae = vae
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.text_encoder_3 = text_encoder_3
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.tokenizer_3 = tokenizer_3

        # Derived constants
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if hasattr(self.vae, "config") and hasattr(self.vae.config, "block_out_channels")
            else 8
        )
        self.default_sample_size = (
            self.transformer._config.sample_size
            if hasattr(self.transformer, "_config") and hasattr(self.transformer._config, "sample_size")
            else 128
        )
        self.patch_size = (
            self.transformer._config.patch_size
            if hasattr(self.transformer, "_config") and hasattr(self.transformer._config, "patch_size")
            else 2
        )
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length
            if self.tokenizer is not None
            else 77
        )

    # ------------------------------------------------------------------
    # Device helpers
    # ------------------------------------------------------------------
    def to(self, device):
        self.transformer = self.transformer.to(device)
        self.vae = self.vae.to(device)
        self.text_encoder = self.text_encoder.to(device)
        self.text_encoder_2 = self.text_encoder_2.to(device)
        if self.text_encoder_3 is not None:
            self.text_encoder_3 = self.text_encoder_3.to(device)
        return self

    @property
    def device(self):
        return next(self.transformer.parameters()).device

    @property
    def dtype(self):
        return next(self.transformer.parameters()).dtype

    # ------------------------------------------------------------------
    # CLIP text encoding
    # ------------------------------------------------------------------
    def _get_clip_prompt_embeds(self, prompt, device, clip_skip=None,
                                clip_model_index=0):
        """Encode prompt with one of the CLIP text encoders.

        Returns (prompt_embeds, pooled_prompt_embeds).
        """
        tokenizer = [self.tokenizer, self.tokenizer_2][clip_model_index]
        text_encoder = [self.text_encoder, self.text_encoder_2][clip_model_index]

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        outputs = text_encoder(text_input_ids, output_hidden_states=True)
        pooled = outputs[0]

        if clip_skip is None:
            embeds = outputs.hidden_states[-2]
        else:
            embeds = outputs.hidden_states[-(clip_skip + 2)]

        embeds = embeds.to(dtype=self.text_encoder.dtype, device=device)
        return embeds, pooled

    # ------------------------------------------------------------------
    # T5 text encoding
    # ------------------------------------------------------------------
    def _get_t5_prompt_embeds(self, prompt, max_sequence_length=256,
                               device=None):
        """Encode prompt with T5 text encoder."""
        device = device or self.device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if self.text_encoder_3 is None:
            return torch.zeros(
                (batch_size, max_sequence_length,
                 self.transformer._config.joint_attention_dim),
                device=device, dtype=self.dtype)

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        embeds = self.text_encoder_3(text_input_ids)[0]
        embeds = embeds.to(dtype=self.text_encoder_3.dtype, device=device)
        return embeds

    # ------------------------------------------------------------------
    # Full prompt encoding
    # ------------------------------------------------------------------
    def _encode_prompt(self, prompt, negative_prompt=None, device=None,
                       max_sequence_length=256):
        """Encode prompt with all three text encoders.

        Returns (prompt_embeds, negative_prompt_embeds,
                 pooled_prompt_embeds, negative_pooled_prompt_embeds).
        """
        device = device or self.device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        # CLIP embeddings
        prompt_embed_1, pooled_1 = self._get_clip_prompt_embeds(
            prompt, device, clip_model_index=0)
        prompt_embed_2, pooled_2 = self._get_clip_prompt_embeds(
            prompt, device, clip_model_index=1)
        clip_prompt_embeds = torch.cat([prompt_embed_1, prompt_embed_2], dim=-1)

        # T5 embeddings
        t5_prompt_embeds = self._get_t5_prompt_embeds(
            prompt, max_sequence_length=max_sequence_length, device=device)

        # Pad CLIP to match T5 dim, then concat along sequence
        clip_prompt_embeds = F.pad(
            clip_prompt_embeds,
            (0, t5_prompt_embeds.shape[-1] - clip_prompt_embeds.shape[-1]))
        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embeds], dim=-2)
        pooled_prompt_embeds = torch.cat([pooled_1, pooled_2], dim=-1)

        # Negative prompt
        negative_prompt = negative_prompt or ""
        negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
        if len(negative_prompt) == 1 and batch_size > 1:
            negative_prompt = negative_prompt * batch_size

        neg_embed_1, neg_pooled_1 = self._get_clip_prompt_embeds(
            negative_prompt, device, clip_model_index=0)
        neg_embed_2, neg_pooled_2 = self._get_clip_prompt_embeds(
            negative_prompt, device, clip_model_index=1)
        neg_clip = torch.cat([neg_embed_1, neg_embed_2], dim=-1)

        t5_neg = self._get_t5_prompt_embeds(
            negative_prompt, max_sequence_length=max_sequence_length,
            device=device)

        neg_clip = F.pad(neg_clip, (0, t5_neg.shape[-1] - neg_clip.shape[-1]))
        negative_prompt_embeds = torch.cat([neg_clip, t5_neg], dim=-2)
        negative_pooled_prompt_embeds = torch.cat(
            [neg_pooled_1, neg_pooled_2], dim=-1)

        return (prompt_embeds, negative_prompt_embeds,
                pooled_prompt_embeds, negative_pooled_prompt_embeds)

    # ------------------------------------------------------------------
    # Latent preparation
    # ------------------------------------------------------------------
    def _prepare_latents(self, batch_size, num_channels, height, width,
                         dtype, device, generator):
        shape = (
            batch_size,
            num_channels,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        return torch.randn(shape, generator=generator, device=device,
                           dtype=dtype)

    # ------------------------------------------------------------------
    # VAE decode + image postprocessing
    # ------------------------------------------------------------------
    def _decode_latents(self, latents):
        """Decode latents to images via VAE."""
        latents = (latents / self.vae.config.scaling_factor
                   + self.vae.config.shift_factor)
        image = self.vae.decode(latents, return_dict=False)[0]
        return image

    @staticmethod
    def _postprocess(image: torch.Tensor) -> List[Image.Image]:
        """Convert (B,C,H,W) float tensor in [-1,1] to list of PIL images."""
        image = (image * 0.5 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        return [Image.fromarray(img) for img in image]

    # ------------------------------------------------------------------
    # __call__  — main generation entry point
    # ------------------------------------------------------------------
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        max_sequence_length: int = 256,
        mu: Optional[float] = None,
    ):
        """Generate images from text prompts.

        Mirrors the core interface of
        ``diffusers.StableDiffusion3Pipeline.__call__``.
        """
        device = self.device

        # 0. Defaults
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        do_cfg = guidance_scale > 1.0

        # 1. Encode prompt
        (prompt_embeds, negative_prompt_embeds,
         pooled_prompt_embeds, negative_pooled_prompt_embeds
         ) = self._encode_prompt(prompt, negative_prompt, device,
                                 max_sequence_length)

        if do_cfg:
            prompt_embeds = torch.cat(
                [negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 2. Prepare latents
        num_channels = self.transformer._config.in_channels
        latents = self._prepare_latents(
            batch_size, num_channels, height, width,
            prompt_embeds.dtype, device, generator)

        # 3. Prepare timesteps (with dynamic shifting if configured)
        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            _, _, lat_h, lat_w = latents.shape
            image_seq_len = ((lat_h // self.patch_size)
                             * (lat_w // self.patch_size))
            mu = _calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.16),
            )
            scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu

        timesteps, num_inference_steps = _retrieve_timesteps(
            self.scheduler, num_inference_steps, device, **scheduler_kwargs)

        # 4. Denoising loop
        for i, t in enumerate(timesteps):
            latent_model_input = (torch.cat([latents] * 2)
                                  if do_cfg else latents)
            timestep = t.expand(latent_model_input.shape[0])

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]

            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = (noise_pred_uncond
                              + guidance_scale * (noise_pred_text - noise_pred_uncond))

            latents_dtype = latents.dtype
            latents = self.scheduler.step(
                noise_pred, t, latents, return_dict=False)[0]
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

        # 5. VAE decode
        if output_type == "latent":
            image = latents
        else:
            image = self._decode_latents(latents)
            images = self._postprocess(image)

        if output_type == "latent":
            if not return_dict:
                return (image,)
            return _PipelineOutput(images=image)

        if not return_dict:
            return (images,)
        return _PipelineOutput(images=images)
