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

import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Any, Tuple, List, Union

# ============================================================================
# Level1 operator imports
# ============================================================================

# Parameter-owning operators
from KernelBench.level1.normalization._6_LayerNorm import Model as LayerNorm
from KernelBench.level1.normalization._4_RMSNorm import Model as RMSNorm
from KernelBench.level1.matmul._10_Linear import Model as Linear
from KernelBench.level1.convolutions._8_Conv2d_Square import Model as Conv2d

# Parameter-free operators
from KernelBench.level1.activations._7_Swish import Model as Swish
from KernelBench.level1.activations._8_GELU import Model as GELUAct
from KernelBench.level1.attention._2_Attention import ScaledDotProductAttention
from KernelBench.level1.regularization._1_Dropout import Model as Dropout
from KernelBench.level1.diffusion._7_SinusoidalTimesteps import Model as SinusoidalTimesteps


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

class PatchEmbed(nn.Module):
    """2D image to patch embedding with cropped sincos positional encoding.
    Uses Conv2d level1 op for patch projection."""

    def __init__(self, height: int = 128, width: int = 128, patch_size: int = 2,
                 in_channels: int = 16, embed_dim: int = 2432,
                 pos_embed_max_size: int = 192):
        super().__init__()
        self.patch_size = patch_size
        self.height = height // patch_size
        self.width = width // patch_size
        self.base_size = height // patch_size
        self.pos_embed_max_size = pos_embed_max_size

        self.proj = Conv2d(in_channels, embed_dim, kernel_size=patch_size,
                           stride=patch_size, padding=0, bias=True)

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
        latent = self.proj(latent)
        latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC
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


class TimestepEmbedding(nn.Module):
    """Matches diffusers TimestepEmbedding. Uses Linear + Swish level1 ops."""

    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = Linear(in_channels, time_embed_dim, bias=True)
        self.act = Swish()
        self.linear_2 = Linear(time_embed_dim, time_embed_dim, bias=True)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class CombinedTimestepTextProjEmbeddings(nn.Module):
    """Matches diffusers CombinedTimestepTextProjEmbeddings.
    Combines sinusoidal timestep embedding + pooled text projection."""

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
# Adaptive Layer Norms (matches diffusers AdaLayerNormZero, AdaLayerNormContinuous)
# ============================================================================

class AdaLayerNormZero(nn.Module):
    """Matches diffusers AdaLayerNormZero.
    Produces shift/scale/gate for MSA and MLP from timestep embedding.
    Uses Linear, Swish, LayerNorm level1 ops."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.silu = Swish()
        self.linear = Linear(embedding_dim, 6 * embedding_dim, bias=True)
        self.norm = LayerNorm(embedding_dim, eps=1e-6, elementwise_affine=False)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormContinuous(nn.Module):
    """Matches diffusers AdaLayerNormContinuous.
    Adaptive layer norm with continuous conditioning.
    Uses Linear, Swish, LayerNorm level1 ops."""

    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int,
                 elementwise_affine: bool = False, eps: float = 1e-6, bias: bool = True):
        super().__init__()
        self.silu = Swish()
        self.linear = Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        self.norm = LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


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

        # Image norm (AdaLN-Zero)
        self.norm1 = AdaLayerNormZero(dim)

        # Context norm
        if context_pre_only:
            self.norm1_context = AdaLayerNormContinuous(
                dim, dim, elementwise_affine=False, eps=1e-6, bias=True)
        else:
            self.norm1_context = AdaLayerNormZero(dim)

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

        # Final output: adaptive norm + linear projection
        self.norm_out = AdaLayerNormContinuous(
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
