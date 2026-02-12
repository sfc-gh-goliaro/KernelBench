"""
FLUX.1 — FluxTransformer2DModel (Rectified Flow Transformer)

Implements the FluxTransformer2DModel matching
black-forest-labs/FLUX.1-dev from HuggingFace diffusers.

Architecture (FLUX.1-dev config):
- 64-channel latent input/output (VAE with 16 channels, packed 2x2 -> 64)
- patch_size: 1
- inner_dim: 3072 (24 heads × 128 head_dim)
- num_layers: 19 (dual-stream FluxTransformerBlock)
- num_single_layers: 38 (single-stream FluxSingleTransformerBlock)
- joint_attention_dim: 4096 (T5-XXL encoder)
- pooled_projection_dim: 768 (CLIP pooler_output)
- guidance_embeds: True (guidance-distilled variant)
- axes_dims_rope: (16, 56, 56) for rotary position embeddings

Key differences from SD3.5 (MMDiT):
- No Conv2d patch embedding; uses Linear x_embedder
- Rotary Position Embeddings (RoPE) instead of 2D sincos
- Dual-stream blocks (joint img+txt attention) + single-stream blocks
- FluxAttention: RMSNorm on Q/K, RoPE applied before attention
- GELU(approximate='tanh') in FeedForward
- AdaLayerNormZero for dual blocks, AdaLayerNormZeroSingle for single blocks
- AdaLayerNormContinuous for final output normalization
- Guidance embedding for guidance-distilled models
- Latent packing: 2x2 patches packed into sequence dimension

The forward signature matches diffusers:
    forward(hidden_states, encoder_hidden_states, pooled_projections,
            timestep, img_ids, txt_ids, guidance, ...)

This model delegates all primitive computations to level1 operators.
"""

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
# Rotary Position Embeddings (matches diffusers FluxPosEmbed)
# ============================================================================

def _get_1d_rotary_pos_embed(dim: int, pos: torch.Tensor, theta: float = 10000.0,
                              freqs_dtype=torch.float64):
    """Compute 1D rotary positional embedding (cos, sin) for given positions.

    Returns (cos, sin) each of shape [S, dim] with repeat_interleave pattern.
    Matches diffusers get_1d_rotary_pos_embed with use_real=True,
    repeat_interleave_real=True.
    """
    assert dim % 2 == 0
    freqs = (
        1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype,
                                        device=pos.device) / dim))
    )  # [D/2]
    freqs = torch.outer(pos, freqs)  # [S, D/2]
    freqs_cos = freqs.cos().repeat_interleave(
        2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
    freqs_sin = freqs.sin().repeat_interleave(
        2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
    return freqs_cos, freqs_sin


class FluxPosEmbed(nn.Module):
    """Rotary positional embedding for FLUX.

    Computes per-axis 1D rotary embeddings and concatenates them.
    Matches diffusers FluxPosEmbed exactly.
    """

    def __init__(self, theta: int = 10000, axes_dim: Tuple[int, ...] = (16, 56, 56)):
        super().__init__()
        self.theta = theta
        self.axes_dim = list(axes_dim)

    def forward(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ids: Position indices (seq_len, n_axes)

        Returns:
            (freqs_cos, freqs_sin) each of shape (seq_len, sum(axes_dim))
        """
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        freqs_dtype = torch.float64
        for i in range(n_axes):
            cos, sin = _get_1d_rotary_pos_embed(
                self.axes_dim[i], pos[:, i],
                theta=self.theta, freqs_dtype=freqs_dtype)
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


def _apply_rotary_emb(x: torch.Tensor,
                       freqs_cis: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Apply rotary embeddings to query or key tensor.

    Matches diffusers apply_rotary_emb with use_real=True,
    use_real_unbind_dim=-1, sequence_dim=1.

    Args:
        x: (B, S, H, D) — note: sequence_dim=1 (not 2)
        freqs_cis: (cos, sin) each of shape (S, D)

    Returns:
        Rotated tensor of same shape as x.
    """
    cos, sin = freqs_cis  # [S, D]
    # Expand for batch and head dims: [1, S, 1, D]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    cos, sin = cos.to(x.device), sin.to(x.device)

    # Interleaved rotation: split into real/imag pairs
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)  # [B, S, H, D]

    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    return out


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


class CombinedTimestepGuidanceTextProjEmbeddings(nn.Module):
    """Matches diffusers CombinedTimestepGuidanceTextProjEmbeddings.
    Adds a guidance embedding on top of timestep + text."""

    def __init__(self, embedding_dim: int, pooled_projection_dim: int):
        super().__init__()
        self.time_proj = SinusoidalTimesteps(num_channels=256, flip_sin_to_cos=True,
                                              downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256,
                                                    time_embed_dim=embedding_dim)
        self.guidance_embedder = TimestepEmbedding(in_channels=256,
                                                    time_embed_dim=embedding_dim)
        self.text_embedder = TextProjection(pooled_projection_dim, embedding_dim)

    def forward(self, timestep: torch.Tensor, guidance: torch.Tensor,
                pooled_projection: torch.Tensor) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(
            timesteps_proj.to(dtype=pooled_projection.dtype))
        guidance_proj = self.time_proj(guidance)
        guidance_emb = self.guidance_embedder(
            guidance_proj.to(dtype=pooled_projection.dtype))
        time_guidance_emb = timesteps_emb + guidance_emb
        pooled_projections = self.text_embedder(pooled_projection)
        return time_guidance_emb + pooled_projections


# ============================================================================
# AdaLayerNormZeroSingle (matches diffusers)
# ============================================================================

class AdaLayerNormZeroSingle(nn.Module):
    """Adaptive LayerNorm for single-stream blocks.

    Produces 3 chunks: shift_msa, scale_msa, gate_msa.
    Returns (normalized_x, gate_msa).

    Uses level1 LayerNorm and Linear operators.
    """

    def __init__(self, embedding_dim: int, bias: bool = True):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        self.norm = LayerNorm(embedding_dim, eps=1e-6, elementwise_affine=False)

    def forward(self, x: torch.Tensor, emb: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa


# ============================================================================
# FeedForward (matches diffusers FeedForward with gelu-approximate)
# ============================================================================

class GELUProjection(nn.Module):
    """Linear + GELU(approximate='tanh'). Uses level1 ops."""

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
# FluxAttention (matches diffusers FluxAttention + FluxAttnProcessor)
# ============================================================================

class FluxAttention(nn.Module):
    """
    Attention module for FLUX transformer blocks.

    Supports both dual-stream (with encoder_hidden_states) and single-stream
    (pre_only) modes. Uses RMSNorm on Q/K, rotary position embeddings,
    and ScaledDotProductAttention.

    Uses level1 Linear, RMSNorm, Dropout, and ScaledDotProductAttention ops.
    """

    def __init__(self, query_dim: int, heads: int = 8, dim_head: int = 64,
                 bias: bool = False, added_kv_proj_dim: Optional[int] = None,
                 out_dim: Optional[int] = None,
                 context_pre_only: Optional[bool] = None,
                 pre_only: bool = False, eps: float = 1e-6):
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.head_dim = dim_head
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.added_kv_proj_dim = added_kv_proj_dim

        # Q/K normalization (RMSNorm with learnable weight)
        self.norm_q = RMSNorm(dim_head, eps=eps, learnable_weight=True, dim=-1)
        self.norm_k = RMSNorm(dim_head, eps=eps, learnable_weight=True, dim=-1)

        # Image Q/K/V projections
        self.to_q = Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = Linear(query_dim, self.inner_dim, bias=bias)

        # Output projection (not present in pre_only / single-stream mode)
        if not self.pre_only:
            self.to_out = nn.ModuleList([
                Linear(self.inner_dim, query_dim, bias=True),
                Dropout(p=0.0),
            ])

        # Context (encoder) projections for dual-stream blocks
        if added_kv_proj_dim is not None:
            self.norm_added_q = RMSNorm(dim_head, eps=eps, learnable_weight=True, dim=-1)
            self.norm_added_k = RMSNorm(dim_head, eps=eps, learnable_weight=True, dim=-1)
            self.add_q_proj = Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.add_k_proj = Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.add_v_proj = Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.to_add_out = Linear(self.inner_dim, query_dim, bias=True)

        # Attention kernel
        self.sdpa = ScaledDotProductAttention(mode="sdpa")

    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            hidden_states: (B, S_img, D) or (B, S_img+S_txt, D) for single-stream
            encoder_hidden_states: (B, S_txt, D) for dual-stream blocks
            image_rotary_emb: (cos, sin) for RoPE

        Returns:
            For dual-stream: (hidden_states, encoder_hidden_states)
            For single-stream: hidden_states
        """
        # Image Q/K/V projections
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        # Reshape to multi-head: (B, S, H, D)
        query = query.unflatten(-1, (self.heads, -1))
        key = key.unflatten(-1, (self.heads, -1))
        value = value.unflatten(-1, (self.heads, -1))

        # QK normalization
        query = self.norm_q(query)
        key = self.norm_k(key)

        # Context projections for dual-stream
        if self.added_kv_proj_dim is not None and encoder_hidden_states is not None:
            encoder_query = self.add_q_proj(encoder_hidden_states)
            encoder_key = self.add_k_proj(encoder_hidden_states)
            encoder_value = self.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(-1, (self.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (self.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (self.heads, -1))

            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)

            # Concatenate encoder and image tokens
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        # Apply rotary position embeddings
        if image_rotary_emb is not None:
            query = _apply_rotary_emb(query, image_rotary_emb)
            key = _apply_rotary_emb(key, image_rotary_emb)

        # Transpose to (B, H, S, D) for SDPA
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Scaled dot-product attention
        hidden_out = self.sdpa(query, key, value)

        # Reshape back: (B, H, S, D) -> (B, S, H*D)
        hidden_out = hidden_out.transpose(1, 2).flatten(2, 3)
        hidden_out = hidden_out.to(query.dtype)

        # Split and project outputs
        if encoder_hidden_states is not None:
            encoder_hidden_states_len = encoder_hidden_states.shape[1]
            encoder_out = hidden_out[:, :encoder_hidden_states_len]
            hidden_out = hidden_out[:, encoder_hidden_states_len:]

            # Output projections
            hidden_out = self.to_out[0](hidden_out)
            hidden_out = self.to_out[1](hidden_out)
            encoder_out = self.to_add_out(encoder_out)

            return hidden_out, encoder_out
        else:
            # Single-stream: no split needed
            return hidden_out


# ============================================================================
# FluxTransformerBlock (dual-stream, matches diffusers)
# ============================================================================

class FluxTransformerBlock(nn.Module):
    """Dual-stream transformer block for FLUX.

    Joint attention between image and text tokens, with separate
    FFN paths for each stream. Uses AdaLN-Zero conditioning.
    """

    def __init__(self, dim: int, num_attention_heads: int,
                 attention_head_dim: int, eps: float = 1e-6):
        super().__init__()
        # Image norm (AdaLN-Zero with 6 chunks)
        self.norm1 = AdaLNZero(dim, num_output_chunks=6)

        # Context norm (AdaLN-Zero with 6 chunks)
        self.norm1_context = AdaLNZero(dim, num_output_chunks=6)

        # Joint attention
        self.attn = FluxAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            eps=eps,
        )

        # Image FFN
        self.norm2 = LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.ff = FeedForward(dim=dim, dim_out=dim)

        # Context FFN
        self.norm2_context = LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.ff_context = FeedForward(dim=dim, dim_out=dim)

    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                temb: torch.Tensor,
                image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                joint_attention_kwargs: Optional[Dict[str, Any]] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Norms
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = \
            self.norm1_context(encoder_hidden_states, emb=temb)

        # 2. Joint attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        # 3. Image: attention residual + FFN
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output

        # 4. Context: attention residual + FFN
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        # fp16 safety clamp (matches diffusers)
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


# ============================================================================
# FluxSingleTransformerBlock (single-stream, matches diffusers)
# ============================================================================

class FluxSingleTransformerBlock(nn.Module):
    """Single-stream transformer block for FLUX.

    Concatenates text and image tokens, applies self-attention and
    parallel MLP, then splits back.
    """

    def __init__(self, dim: int, num_attention_heads: int,
                 attention_head_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = Linear(dim, self.mlp_hidden_dim, bias=True)
        self.act_mlp = GELUAct(approximate='tanh')
        self.proj_out = Linear(dim + self.mlp_hidden_dim, dim, bias=True)

        self.attn = FluxAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            pre_only=True,
            eps=1e-6,
        )

    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                temb: torch.Tensor,
                image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                joint_attention_kwargs: Optional[Dict[str, Any]] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)

        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states

        # fp16 safety clamp (matches diffusers)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        encoder_hidden_states, hidden_states = (
            hidden_states[:, :text_seq_len],
            hidden_states[:, text_seq_len:],
        )
        return encoder_hidden_states, hidden_states


# ============================================================================
# Main Model: FluxTransformer2DModel
# ============================================================================

class Flux(nn.Module):
    """
    FluxTransformer2DModel matching black-forest-labs/FLUX.1-dev.

    Rectified flow transformer with:
    - Rotary position embeddings
    - Dual-stream joint attention blocks (image + text)
    - Single-stream self-attention blocks
    - AdaLayerNorm conditioning from timestep + guidance + pooled text
    - RMSNorm on Q/K
    - GELU(tanh) feed-forward networks

    All primitive computations are delegated to level1 operators.
    """

    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        # Store config for pipeline compatibility
        self._config = type("Config", (), {
            "patch_size": patch_size,
            "in_channels": in_channels,
            "out_channels": self.out_channels,
            "num_layers": num_layers,
            "num_single_layers": num_single_layers,
            "attention_head_dim": attention_head_dim,
            "num_attention_heads": num_attention_heads,
            "joint_attention_dim": joint_attention_dim,
            "pooled_projection_dim": pooled_projection_dim,
            "guidance_embeds": guidance_embeds,
            "axes_dims_rope": tuple(axes_dims_rope),
        })()

        # Rotary position embeddings
        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        # Timestep + text + (optional) guidance conditioning
        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings
            if guidance_embeds
            else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim,
            pooled_projection_dim=pooled_projection_dim,
        )

        # Input projections (simple linear, not Conv2d patchify)
        self.context_embedder = Linear(joint_attention_dim, self.inner_dim, bias=True)
        self.x_embedder = Linear(in_channels, self.inner_dim, bias=True)

        # Dual-stream transformer blocks
        self.transformer_blocks = nn.ModuleList([
            FluxTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
            )
            for _ in range(num_layers)
        ])

        # Single-stream transformer blocks
        self.single_transformer_blocks = nn.ModuleList([
            FluxSingleTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
            )
            for _ in range(num_single_layers)
        ])

        # Final output: adaptive norm + linear projection
        self.norm_out = AdaLNContinuous(
            self.inner_dim, self.inner_dim,
            elementwise_affine=False, eps=1e-6)
        self.proj_out = Linear(self.inner_dim,
                               patch_size * patch_size * self.out_channels,
                               bias=True)

    @property
    def config(self):
        """Config object for pipeline compatibility."""
        return self._config

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the model parameters."""
        return next(self.parameters()).dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Tuple]:
        """
        Forward pass matching diffusers FluxTransformer2DModel.

        Args:
            hidden_states: Packed latent sequence (B, S_img, in_channels)
            encoder_hidden_states: T5 text embeddings (B, S_txt, joint_attention_dim)
            pooled_projections: CLIP pooled embeddings (B, pooled_projection_dim)
            timestep: Diffusion timestep (already divided by 1000 by pipeline)
            img_ids: Image position IDs (S_img, 3) for RoPE
            txt_ids: Text position IDs (S_txt, 3) for RoPE
            guidance: Guidance scale embedding (B,) for guidance-distilled models
            return_dict: If True, return dict with 'sample' key

        Returns:
            If return_dict=False: (sample,) tuple
            If return_dict=True: dict with 'sample' key
        """
        # 1. Input projections
        hidden_states = self.x_embedder(hidden_states)

        # 2. Timestep conditioning (multiply by 1000 as diffusers does internally)
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        if guidance is not None:
            temb = self.time_text_embed(timestep, guidance, pooled_projections)
        else:
            temb = self.time_text_embed(timestep, pooled_projections)

        # 3. Context projection
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # 4. Compute rotary position embeddings
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        # 5. Dual-stream transformer blocks
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        # 6. Single-stream transformer blocks
        for block in self.single_transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        # 7. Final norm + projection
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return {"sample": output}


# Backward-compatible alias
Model = Flux


# ============================================================================
# FLUX Pipeline — full text-to-image generation
# ============================================================================
# Implements the same logic as diffusers FluxPipeline.__call__
# but without inheriting from DiffusionPipeline.
#
# External HF components used:
#   - text_encoder   (CLIPTextModel — only pooler_output used)
#   - text_encoder_2  (T5EncoderModel — hidden states used)
#   - tokenizer       (CLIPTokenizer)
#   - tokenizer_2     (T5TokenizerFast)
#   - vae             (AutoencoderKL)
#   - scheduler       (FlowMatchEulerDiscreteScheduler)
#
# The denoiser is our own Flux (FluxTransformer2DModel).
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
    """Minimal output object with ``.images`` attribute."""
    __slots__ = ("images",)

    def __init__(self, images):
        self.images = images


class FluxPipeline:
    """KernelBench FLUX text-to-image pipeline.

    Drop-in replacement for ``diffusers.FluxPipeline`` with the same
    ``__call__`` interface (subset of parameters that matter for generation).
    Uses our :class:`Flux` transformer as the denoiser.
    """

    def __init__(
        self,
        transformer: "Flux",
        scheduler,
        vae,
        text_encoder,
        text_encoder_2,
        tokenizer,
        tokenizer_2,
    ):
        self.transformer = transformer
        self.scheduler = scheduler
        self.vae = vae
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2

        # Derived constants
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if hasattr(self.vae, "config") and hasattr(self.vae.config, "block_out_channels")
            else 8
        )
        self.default_sample_size = 128
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
        return self

    @property
    def device(self):
        return next(self.transformer.parameters()).device

    @property
    def dtype(self):
        return next(self.transformer.parameters()).dtype

    # ------------------------------------------------------------------
    # CLIP text encoding (pooler_output only for FLUX)
    # ------------------------------------------------------------------
    def _get_clip_prompt_embeds(self, prompt, device):
        """Encode prompt with CLIP. Returns pooled output only."""
        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        outputs = self.text_encoder(text_input_ids, output_hidden_states=False)
        pooled = outputs.pooler_output
        pooled = pooled.to(dtype=self.text_encoder.dtype, device=device)
        return pooled

    # ------------------------------------------------------------------
    # T5 text encoding
    # ------------------------------------------------------------------
    def _get_t5_prompt_embeds(self, prompt, max_sequence_length=512,
                               device=None):
        """Encode prompt with T5 text encoder."""
        device = device or self.device
        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        embeds = self.text_encoder_2(text_input_ids)[0]
        dtype = self.text_encoder_2.dtype
        embeds = embeds.to(dtype=dtype, device=device)
        return embeds

    # ------------------------------------------------------------------
    # Full prompt encoding
    # ------------------------------------------------------------------
    def _encode_prompt(self, prompt, device=None, max_sequence_length=512):
        """Encode prompt with CLIP (pooled) and T5 (sequence).

        Returns (prompt_embeds, pooled_prompt_embeds, text_ids).
        """
        device = device or self.device

        pooled_prompt_embeds = self._get_clip_prompt_embeds(prompt, device)
        prompt_embeds = self._get_t5_prompt_embeds(
            prompt, max_sequence_length=max_sequence_length, device=device)

        dtype = (self.text_encoder.dtype
                 if self.text_encoder is not None
                 else self.transformer.dtype)
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(
            device=device, dtype=dtype)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    # ------------------------------------------------------------------
    # Latent preparation (with packing)
    # ------------------------------------------------------------------
    @staticmethod
    def _pack_latents(latents, batch_size, num_channels, height, width):
        """Pack 2x2 patches: (B,C,H,W) -> (B, H/2*W/2, C*4)."""
        latents = latents.view(batch_size, num_channels, height // 2, 2,
                               width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2),
                                  num_channels * 4)
        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        """Unpack: (B, S, C*4) -> (B, C, H, W)."""
        batch_size, num_patches, channels = latents.shape
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))
        latents = latents.view(batch_size, height // 2, width // 2,
                               channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (2 * 2),
                                  height, width)
        return latents

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        """Create position IDs for image latents."""
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height)[:, None])
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width)[None, :])
        latent_image_ids = latent_image_ids.reshape(
            height * width, 3)
        return latent_image_ids.to(device=device, dtype=dtype)

    def _prepare_latents(self, batch_size, num_channels, height, width,
                         dtype, device, generator):
        """Prepare random latents with packing."""
        # Account for VAE compression and 2x2 packing
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels, height, width)
        latents = torch.randn(shape, generator=generator, device=device,
                              dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels,
                                     height, width)
        latent_image_ids = self._prepare_latent_image_ids(
            batch_size, height // 2, width // 2, device, dtype)
        return latents, latent_image_ids

    # ------------------------------------------------------------------
    # VAE decode + image postprocessing
    # ------------------------------------------------------------------
    def _decode_latents(self, latents, height, width):
        """Unpack, scale, and decode latents to images via VAE."""
        latents = self._unpack_latents(latents, height, width,
                                       self.vae_scale_factor)
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
    # __call__ — main generation entry point
    # ------------------------------------------------------------------
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        max_sequence_length: int = 512,
        **kwargs,
    ):
        """Generate images from text prompts.

        Mirrors the core interface of ``diffusers.FluxPipeline.__call__``.

        Note: FLUX uses guidance embedding (not CFG), so negative_prompt is
        ignored and guidance_scale is passed as an embedding input.
        """
        device = self.device

        # 0. Defaults
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        batch_size = 1 if isinstance(prompt, str) else len(prompt)

        # 1. Encode prompt
        prompt_embeds, pooled_prompt_embeds, text_ids = \
            self._encode_prompt(prompt, device, max_sequence_length)

        # 2. Prepare latents (packed)
        num_channels_latents = self.transformer._config.in_channels // 4
        latents, latent_image_ids = self._prepare_latents(
            batch_size, num_channels_latents, height, width,
            prompt_embeds.dtype, device, generator)

        # 3. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None

        image_seq_len = latents.shape[1]
        mu = _calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = _retrieve_timesteps(
            self.scheduler, num_inference_steps, device,
            sigmas=sigmas, mu=mu)

        # 4. Handle guidance embedding
        if self.transformer._config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device,
                                  dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # 5. Denoising loop
        self.scheduler.set_begin_index(0)
        for i, t in enumerate(timesteps):
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            noise_pred = self.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]

            latents_dtype = latents.dtype
            latents = self.scheduler.step(
                noise_pred, t, latents, return_dict=False)[0]
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

        # 6. VAE decode
        if output_type == "latent":
            image = latents
        else:
            image = self._decode_latents(latents, height, width)
            images = self._postprocess(image)

        if output_type == "latent":
            if not return_dict:
                return (image,)
            return _PipelineOutput(images=image)

        if not return_dict:
            return (images,)
        return _PipelineOutput(images=images)
