"""
HunyuanVideo 1.5 — HunyuanVideo15Transformer3DModel (3D Video Diffusion Transformer)

Implements the HunyuanVideo15Transformer3DModel matching
tencent/HunyuanVideo-1.5 from HuggingFace diffusers.

Architecture (HunyuanVideo 1.5 config):
- 65-channel latent input (32 latent + 33 condition channels)
- 32-channel latent output
- inner_dim: 2048 (16 heads × 128 head_dim)
- num_layers: 54 (dual-stream HunyuanVideo15TransformerBlock)
- num_refiner_layers: 2 (token refiner blocks)
- text_embed_dim: 3584 (Qwen2.5VL text encoder)
- text_embed_2_dim: 1472 (ByT5 text encoder)
- image_embed_dim: 1152 (image conditioning)
- rope_axes_dim: (16, 56, 56) for 3D rotary position embeddings
- patch_size: 1, patch_size_t: 1

Key features:
- 3D patch embedding (Conv3d)
- Token refiner for Qwen text embeddings
- ByT5 text projection
- Image embedding projection
- Condition type embeddings (Qwen=0, ByT5=1, Image=2)
- 3D Rotary Position Embeddings (temporal, height, width)
- Dual-stream transformer blocks with AdaLayerNormZero + joint attention
- AdaLayerNormContinuous output normalization

The forward signature matches diffusers:
    forward(hidden_states, timestep, encoder_hidden_states,
            encoder_attention_mask, encoder_hidden_states_2,
            encoder_attention_mask_2, image_embeds, ...)

This model delegates all primitive computations to level1 operators.
"""

import contextlib
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List, Union

# ============================================================================
# Level1 operator imports
# ============================================================================

from KernelBench.level1.normalization._6_LayerNorm import Model as LayerNorm
from KernelBench.level1.normalization._4_RMSNorm import Model as RMSNorm
from KernelBench.level1.matmul._10_Linear import Model as Linear
from KernelBench.level1.diffusion._1_AdaLN import Model as AdaLNContinuous
from KernelBench.level1.diffusion._2_AdaLN_Zero import Model as AdaLNZero
from KernelBench.level1.diffusion._3_TimestepEmbedding import Model as TimestepEmbedding
from KernelBench.level1.diffusion._7_SinusoidalTimesteps import Model as SinusoidalTimesteps
from KernelBench.level1.activations._7_Swish import Model as Swish
from KernelBench.level1.activations._8_GELU import Model as GELUAct
from KernelBench.level1.attention._2_Attention import ScaledDotProductAttention
from KernelBench.level1.regularization._1_Dropout import Model as Dropout
from KernelBench.level1.embeddings._2_Embedding import Model as Embedding
from KernelBench.level1.vision._2_PatchEmbed3D import Model as PatchEmbed3D


# ============================================================================
# 3D Rotary Position Embeddings
# ============================================================================

def _get_1d_rotary_pos_embed(dim: int, pos: torch.Tensor, theta: float = 256.0):
    """Compute 1D rotary positional embedding (cos, sin) for given positions.

    Returns (cos, sin) each of shape [S, dim] with repeat_interleave pattern.
    Matches diffusers get_1d_rotary_pos_embed with use_real=True,
    repeat_interleave_real=True.
    """
    assert dim % 2 == 0
    freqs = (
        1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32,
                                       device=pos.device) / dim))
    )
    freqs = torch.outer(pos, freqs)  # [S, dim/2]
    cos = freqs.cos().repeat_interleave(2, dim=1).float()  # [S, dim]
    sin = freqs.sin().repeat_interleave(2, dim=1).float()  # [S, dim]
    return cos, sin


def _apply_rotary_emb(x: torch.Tensor, freqs_cis: Tuple[torch.Tensor, torch.Tensor],
                      sequence_dim: int = 1) -> torch.Tensor:
    """Apply rotary embedding to tensor x.

    Matches diffusers apply_rotary_emb with use_real=True, use_real_unbind_dim=-1.

    Args:
        x: (B, S, H, D) or (B, H, S, D) depending on sequence_dim
        freqs_cis: (cos, sin) each of shape (S, D)
        sequence_dim: 1 for (B, S, H, D), 2 for (B, H, S, D)
    """
    cos, sin = freqs_cis
    if sequence_dim == 2:
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    elif sequence_dim == 1:
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    else:
        raise ValueError(f"sequence_dim={sequence_dim} not supported")

    cos, sin = cos.to(x.device), sin.to(x.device)

    # Unbind along last dim (flux/hunyuan style)
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)

    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    return out


class HunyuanVideo15RotaryPosEmbed(nn.Module):
    """3D Rotary Position Embeddings for video (temporal, height, width).

    Matches HunyuanVideo15RotaryPosEmbed from diffusers exactly.
    """

    def __init__(self, patch_size: int, patch_size_t: int,
                 rope_dim: List[int], theta: float = 256.0):
        super().__init__()
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.rope_dim = rope_dim
        self.theta = theta

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        rope_sizes = [
            num_frames // self.patch_size_t,
            height // self.patch_size,
            width // self.patch_size,
        ]

        axes_grids = []
        for i in range(len(rope_sizes)):
            grid = torch.arange(0, rope_sizes[i], device=hidden_states.device, dtype=torch.float32)
            axes_grids.append(grid)
        grid = torch.meshgrid(*axes_grids, indexing="ij")
        grid = torch.stack(grid, dim=0)  # [3, T, H, W]

        freqs = []
        for i in range(3):
            freq = _get_1d_rotary_pos_embed(
                self.rope_dim[i], grid[i].reshape(-1), self.theta
            )
            freqs.append(freq)

        freqs_cos = torch.cat([f[0] for f in freqs], dim=1)  # (T*H*W, D/2)
        freqs_sin = torch.cat([f[1] for f in freqs], dim=1)  # (T*H*W, D/2)
        return freqs_cos, freqs_sin


# ============================================================================
# Patch Embedding — uses level1 PatchEmbed3D
# ============================================================================


# ============================================================================
# Image Projection
# ============================================================================

class HunyuanVideo15ImageProjection(nn.Module):
    """Projects image embeddings to transformer hidden size.

    Matches HunyuanVideo15ImageProjection from diffusers.
    """

    def __init__(self, in_channels: int, hidden_size: int):
        super().__init__()
        self.norm_in = LayerNorm(in_channels, eps=1e-5)
        self.linear_1 = Linear(in_channels, in_channels, bias=True)
        self.act_fn = GELUAct()
        self.linear_2 = Linear(in_channels, hidden_size, bias=True)
        self.norm_out = LayerNorm(hidden_size, eps=1e-5)

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm_in(image_embeds)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.norm_out(hidden_states)
        return hidden_states


# ============================================================================
# ByT5 Text Projection
# ============================================================================

class HunyuanVideo15ByT5TextProjection(nn.Module):
    """Projects ByT5 text embeddings to transformer hidden size.

    Matches HunyuanVideo15ByT5TextProjection from diffusers.
    """

    def __init__(self, in_features: int, hidden_size: int, out_features: int):
        super().__init__()
        self.norm = LayerNorm(in_features, eps=1e-5)
        self.linear_1 = Linear(in_features, hidden_size, bias=True)
        self.linear_2 = Linear(hidden_size, hidden_size, bias=True)
        self.linear_3 = Linear(hidden_size, out_features, bias=True)
        self.act_fn = GELUAct()

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(encoder_hidden_states)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_3(hidden_states)
        return hidden_states


# ============================================================================
# Time Embedding
# ============================================================================

class HunyuanVideo15TimeEmbedding(nn.Module):
    """Timestep embedding for HunyuanVideo 1.5.

    Matches HunyuanVideo15TimeEmbedding from diffusers.
    Uses level1 SinusoidalTimesteps + TimestepEmbedding.
    """

    def __init__(self, embedding_dim: int, use_meanflow: bool = False):
        super().__init__()
        self.time_proj = SinusoidalTimesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )

        self.use_meanflow = use_meanflow
        self.time_proj_r = None
        self.timestep_embedder_r = None
        if use_meanflow:
            self.time_proj_r = SinusoidalTimesteps(
                num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
            )
            self.timestep_embedder_r = TimestepEmbedding(
                in_channels=256, time_embed_dim=embedding_dim
            )

    def forward(self, timestep: torch.Tensor,
                timestep_r: Optional[torch.Tensor] = None) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=timestep.dtype))

        if timestep_r is not None and self.timestep_embedder_r is not None:
            timesteps_proj_r = self.time_proj_r(timestep_r)
            timesteps_emb_r = self.timestep_embedder_r(
                timesteps_proj_r.to(dtype=timestep.dtype)
            )
            timesteps_emb = timesteps_emb + timesteps_emb_r

        return timesteps_emb


# ============================================================================
# Token Refiner
# ============================================================================

class HunyuanVideo15AdaNorm(nn.Module):
    """Adaptive normalization for token refiner.

    Matches HunyuanVideo15AdaNorm from diffusers.
    """

    def __init__(self, in_features: int, out_features: Optional[int] = None):
        super().__init__()
        out_features = out_features or 2 * in_features
        self.linear = Linear(in_features, out_features, bias=True)
        self.nonlinearity = Swish()

    def forward(self, temb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        temb = self.linear(self.nonlinearity(temb))
        gate_msa, gate_mlp = temb.chunk(2, dim=1)
        gate_msa, gate_mlp = gate_msa.unsqueeze(1), gate_mlp.unsqueeze(1)
        return gate_msa, gate_mlp


class HunyuanVideo15IndividualTokenRefinerBlock(nn.Module):
    """Individual token refiner block.

    Matches HunyuanVideo15IndividualTokenRefinerBlock from diffusers.
    Uses self-attention + FFN with adaptive gating.
    """

    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ):
        super().__init__()
        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = LayerNorm(hidden_size, eps=1e-6)

        # Self-attention (matching diffusers Attention module layout)
        self.attn = nn.ModuleDict({
            'to_q': Linear(hidden_size, hidden_size, bias=attention_bias),
            'to_k': Linear(hidden_size, hidden_size, bias=attention_bias),
            'to_v': Linear(hidden_size, hidden_size, bias=attention_bias),
            'to_out': nn.ModuleList([
                Linear(hidden_size, hidden_size, bias=True),
                Dropout(p=0.0),
            ]),
        })
        self.attn_heads = num_attention_heads
        self.attn_head_dim = attention_head_dim
        self.sdpa = ScaledDotProductAttention(mode="sdpa")

        self.norm2 = LayerNorm(hidden_size, eps=1e-6)

        # FFN: LinearActivation(silu) -> Dropout -> Linear
        mlp_dim = int(hidden_size * mlp_width_ratio)
        self.ff = nn.ModuleDict({
            'net': nn.ModuleList([
                _LinearSiLU(hidden_size, mlp_dim),
                Dropout(p=mlp_drop_rate),
                Linear(mlp_dim, hidden_size, bias=True),
            ]),
        })

        self.norm_out = HunyuanVideo15AdaNorm(hidden_size, 2 * hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        norm_hidden_states = self.norm1(hidden_states)

        # Self-attention
        q = self.attn['to_q'](norm_hidden_states)
        k = self.attn['to_k'](norm_hidden_states)
        v = self.attn['to_v'](norm_hidden_states)

        bsz, seq_len, _ = q.shape
        q = q.view(bsz, seq_len, self.attn_heads, self.attn_head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.attn_heads, self.attn_head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.attn_heads, self.attn_head_dim).transpose(1, 2)

        attn_output = self.sdpa(q, k, v, attn_mask=attention_mask)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)
        attn_output = self.attn['to_out'][0](attn_output)
        attn_output = self.attn['to_out'][1](attn_output)

        gate_msa, gate_mlp = self.norm_out(temb)
        hidden_states = hidden_states + attn_output * gate_msa

        # FFN
        ff_output = self.norm2(hidden_states)
        for module in self.ff['net']:
            ff_output = module(ff_output)
        hidden_states = hidden_states + ff_output * gate_mlp

        return hidden_states


class HunyuanVideo15IndividualTokenRefiner(nn.Module):
    """Token refiner with multiple blocks.

    Matches HunyuanVideo15IndividualTokenRefiner from diffusers.
    """

    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ):
        super().__init__()
        self.refiner_blocks = nn.ModuleList([
            HunyuanVideo15IndividualTokenRefinerBlock(
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                mlp_width_ratio=mlp_width_ratio,
                mlp_drop_rate=mlp_drop_rate,
                attention_bias=attention_bias,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self_attn_mask = None
        if attention_mask is not None:
            batch_size = attention_mask.shape[0]
            seq_len = attention_mask.shape[1]
            attention_mask = attention_mask.to(hidden_states.device).bool()
            self_attn_mask_1 = attention_mask.view(batch_size, 1, 1, seq_len).repeat(1, 1, seq_len, 1)
            self_attn_mask_2 = self_attn_mask_1.transpose(2, 3)
            self_attn_mask = (self_attn_mask_1 & self_attn_mask_2).bool()

        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, temb, self_attn_mask)

        return hidden_states


class HunyuanVideo15TokenRefiner(nn.Module):
    """Token refiner for Qwen text embeddings.

    Matches HunyuanVideo15TokenRefiner from diffusers.
    Uses CombinedTimestepTextProjEmbeddings pattern.
    """

    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ):
        super().__init__()
        hidden_size = num_attention_heads * attention_head_dim

        # CombinedTimestepTextProjEmbeddings
        self.time_text_embed = nn.ModuleDict({
            'timestep_embedder': TimestepEmbedding(
                in_channels=256, time_embed_dim=hidden_size
            ),
            'text_embedder': nn.ModuleDict({
                'linear_1': Linear(in_channels, hidden_size, bias=True),
                'act_1': Swish(),
                'linear_2': Linear(hidden_size, hidden_size, bias=True),
            }),
        })
        self.time_proj = SinusoidalTimesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
        )

        self.proj_in = Linear(in_channels, hidden_size, bias=True)
        self.token_refiner = HunyuanVideo15IndividualTokenRefiner(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            mlp_width_ratio=mlp_ratio,
            mlp_drop_rate=mlp_drop_rate,
            attention_bias=attention_bias,
        )

    def _compute_temb(self, timestep: torch.Tensor,
                      pooled_projections: torch.Tensor) -> torch.Tensor:
        """Compute combined timestep + text embedding."""
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.time_text_embed['timestep_embedder'](
            timesteps_proj.to(dtype=pooled_projections.dtype)
        )
        text_emb = self.time_text_embed['text_embedder']['linear_1'](pooled_projections)
        text_emb = self.time_text_embed['text_embedder']['act_1'](text_emb)
        text_emb = self.time_text_embed['text_embedder']['linear_2'](text_emb)
        return timesteps_emb + text_emb

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            pooled_projections = hidden_states.mean(dim=1)
        else:
            original_dtype = hidden_states.dtype
            mask_float = attention_mask.float().unsqueeze(-1)
            pooled_projections = (hidden_states * mask_float).sum(dim=1) / mask_float.sum(dim=1)
            pooled_projections = pooled_projections.to(original_dtype)

        temb = self._compute_temb(timestep, pooled_projections)
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.token_refiner(hidden_states, temb, attention_mask)

        return hidden_states


# ============================================================================
# FeedForward (GELU-approximate)
# ============================================================================

class _GELULinear(nn.Module):
    """Linear + GELU activation matching diffusers GELU class layout.

    State-dict: proj.{weight,bias}
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = Linear(dim_in, dim_out, bias=True)
        self.gelu = GELUAct(approximate='tanh')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu(self.proj(x))


class _LinearSiLU(nn.Module):
    """Linear + SiLU activation matching diffusers LinearActivation layout.

    State-dict: proj.{weight,bias}
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = Linear(dim_in, dim_out, bias=True)
        self.activation = Swish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.proj(x))


class FeedForward(nn.Module):
    """Feed-forward network with GELU(approximate='tanh') activation.

    Matches diffusers FeedForward with activation_fn='gelu-approximate'.
    State-dict layout:
        net.0.proj.{weight,bias}  (Linear + GELU)
        net.2.{weight,bias}       (output Linear)
    """

    def __init__(self, hidden_size: int, mult: float = 4.0):
        super().__init__()
        inner_dim = int(hidden_size * mult)
        self.net = nn.ModuleList([
            _GELULinear(hidden_size, inner_dim),
            Dropout(p=0.0),
            Linear(inner_dim, hidden_size, bias=True),
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


# ============================================================================
# Dual-Stream Transformer Block
# ============================================================================

class HunyuanVideo15TransformerBlock(nn.Module):
    """Dual-stream transformer block with joint attention.

    Matches HunyuanVideo15TransformerBlock from diffusers.
    Uses AdaLayerNormZero for both latent and context streams.
    """

    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        qk_norm: str = "rms_norm",
    ):
        super().__init__()
        hidden_size = num_attention_heads * attention_head_dim

        # AdaLayerNormZero for latent and context
        self.norm1 = AdaLNZero(hidden_size, num_output_chunks=6, eps=1e-6)
        self.norm1_context = AdaLNZero(hidden_size, num_output_chunks=6, eps=1e-6)

        # Joint attention (matching diffusers Attention module layout)
        self.attn = nn.ModuleDict()
        self.attn['to_q'] = Linear(hidden_size, hidden_size, bias=True)
        self.attn['to_k'] = Linear(hidden_size, hidden_size, bias=True)
        self.attn['to_v'] = Linear(hidden_size, hidden_size, bias=True)
        self.attn['to_out'] = nn.ModuleList([
            Linear(hidden_size, hidden_size, bias=True),
            Dropout(p=0.0),
        ])
        # Context KV projections
        self.attn['add_q_proj'] = Linear(hidden_size, hidden_size, bias=True)
        self.attn['add_k_proj'] = Linear(hidden_size, hidden_size, bias=True)
        self.attn['add_v_proj'] = Linear(hidden_size, hidden_size, bias=True)
        self.attn['to_add_out'] = Linear(hidden_size, hidden_size, bias=True)

        # QK normalization
        if qk_norm == "rms_norm":
            self.attn['norm_q'] = RMSNorm(attention_head_dim, eps=1e-6, learnable_weight=True, dim=-1)
            self.attn['norm_k'] = RMSNorm(attention_head_dim, eps=1e-6, learnable_weight=True, dim=-1)
            self.attn['norm_added_q'] = RMSNorm(attention_head_dim, eps=1e-6, learnable_weight=True, dim=-1)
            self.attn['norm_added_k'] = RMSNorm(attention_head_dim, eps=1e-6, learnable_weight=True, dim=-1)

        self.attn_heads = num_attention_heads
        self.attn_head_dim = attention_head_dim
        self.sdpa = ScaledDotProductAttention(mode="sdpa")

        # FFN for latent and context
        self.norm2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(hidden_size, mult=mlp_ratio)

        self.norm2_context = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(hidden_size, mult=mlp_ratio)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Input normalization (AdaLN-Zero)
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        # 2. Joint attention
        # Latent QKV
        query = self.attn['to_q'](norm_hidden_states)
        key = self.attn['to_k'](norm_hidden_states)
        value = self.attn['to_v'](norm_hidden_states)

        bsz = query.shape[0]
        query = query.unflatten(2, (self.attn_heads, -1))
        key = key.unflatten(2, (self.attn_heads, -1))
        value = value.unflatten(2, (self.attn_heads, -1))

        # QK normalization
        query = self.attn['norm_q'](query)
        key = self.attn['norm_k'](key)

        # Apply RoPE to latent stream
        if freqs_cis is not None:
            query = _apply_rotary_emb(query, freqs_cis, sequence_dim=1)
            key = _apply_rotary_emb(key, freqs_cis, sequence_dim=1)

        # Context QKV
        encoder_query = self.attn['add_q_proj'](norm_encoder_hidden_states)
        encoder_key = self.attn['add_k_proj'](norm_encoder_hidden_states)
        encoder_value = self.attn['add_v_proj'](norm_encoder_hidden_states)

        encoder_query = encoder_query.unflatten(2, (self.attn_heads, -1))
        encoder_key = encoder_key.unflatten(2, (self.attn_heads, -1))
        encoder_value = encoder_value.unflatten(2, (self.attn_heads, -1))

        encoder_query = self.attn['norm_added_q'](encoder_query)
        encoder_key = self.attn['norm_added_k'](encoder_key)

        # Concatenate latent and context
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)

        # Build attention mask for HunyuanVideo15
        seq_len = query.shape[1]
        if attention_mask is not None:
            attention_mask_padded = F.pad(
                attention_mask, (seq_len - attention_mask.shape[1], 0), value=True
            )
            attention_mask_padded = attention_mask_padded.bool()
            self_attn_mask_1 = attention_mask_padded.view(bsz, 1, 1, seq_len).repeat(1, 1, seq_len, 1)
            self_attn_mask_2 = self_attn_mask_1.transpose(2, 3)
            attention_mask_2d = (self_attn_mask_1 & self_attn_mask_2).bool()
        else:
            attention_mask_2d = None

        # Attention (B, S, H, D) -> need to transpose to (B, H, S, D)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attn_output = self.sdpa(
            query, key, value, attn_mask=attention_mask_2d
        )
        attn_output = attn_output.transpose(1, 2).flatten(2, 3)
        attn_output = attn_output.to(query.dtype)

        # Split latent and context outputs
        enc_seq_len = encoder_hidden_states.shape[1]
        attn_output, context_attn_output = (
            attn_output[:, :-enc_seq_len],
            attn_output[:, -enc_seq_len:],
        )

        # Output projections
        attn_output = self.attn['to_out'][0](attn_output)
        attn_output = self.attn['to_out'][1](attn_output)
        context_attn_output = self.attn['to_add_out'](context_attn_output)

        # 3. Modulation and residual connection
        hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + context_attn_output * c_gate_msa.unsqueeze(1)

        norm_hidden_states = self.norm2(hidden_states)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)

        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        # 4. Feed-forward
        ff_output = self.ff(norm_hidden_states)
        context_ff_output = self.ff_context(norm_encoder_hidden_states)

        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return hidden_states, encoder_hidden_states


# ============================================================================
# Main Transformer Model
# ============================================================================

class HunyuanVideo15Transformer(nn.Module):
    """HunyuanVideo 1.5 3D Video Diffusion Transformer.

    Matches HunyuanVideo15Transformer3DModel from diffusers.

    Args:
        in_channels: Input latent channels (default 65: 32 latent + 33 condition)
        out_channels: Output latent channels (default 32)
        num_attention_heads: Number of attention heads
        attention_head_dim: Dimension per attention head
        num_layers: Number of dual-stream transformer blocks
        num_refiner_layers: Number of token refiner layers
        mlp_ratio: MLP expansion ratio
        patch_size: Spatial patch size
        patch_size_t: Temporal patch size
        qk_norm: QK normalization type
        text_embed_dim: Qwen text embedding dimension
        text_embed_2_dim: ByT5 text embedding dimension
        image_embed_dim: Image embedding dimension
        rope_theta: RoPE base frequency
        rope_axes_dim: RoPE dimensions for (temporal, height, width)
        use_meanflow: Whether to use meanflow timestep embedding
    """

    def __init__(
        self,
        in_channels: int = 65,
        out_channels: int = 32,
        num_attention_heads: int = 16,
        attention_head_dim: int = 128,
        num_layers: int = 54,
        num_refiner_layers: int = 2,
        mlp_ratio: float = 4.0,
        patch_size: int = 1,
        patch_size_t: int = 1,
        qk_norm: str = "rms_norm",
        text_embed_dim: int = 3584,
        text_embed_2_dim: int = 1472,
        image_embed_dim: int = 1152,
        rope_theta: float = 256.0,
        rope_axes_dim: Tuple[int, ...] = (16, 56, 56),
        use_meanflow: bool = False,
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # Store config (mirrors diffusers FrozenDict for pipeline compatibility)
        self.config = type('Config', (), {
            'patch_size': patch_size,
            'patch_size_t': patch_size_t,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'num_attention_heads': num_attention_heads,
            'attention_head_dim': attention_head_dim,
            'num_layers': num_layers,
            'num_refiner_layers': num_refiner_layers,
            'mlp_ratio': mlp_ratio,
            'qk_norm': qk_norm,
            'text_embed_dim': text_embed_dim,
            'text_embed_2_dim': text_embed_2_dim,
            'image_embed_dim': image_embed_dim,
            'rope_theta': rope_theta,
            'rope_axes_dim': list(rope_axes_dim),
            'target_size': 640,
            'use_meanflow': use_meanflow,
        })()
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t

        # 1. Latent and condition embedders
        self.x_embedder = PatchEmbed3D(
            patch_size=patch_size, temporal_patch_size=patch_size_t,
            in_channels=in_channels, embed_dim=inner_dim,
        )
        self.image_embedder = HunyuanVideo15ImageProjection(image_embed_dim, inner_dim)

        self.context_embedder = HunyuanVideo15TokenRefiner(
            text_embed_dim, num_attention_heads, attention_head_dim,
            num_layers=num_refiner_layers,
        )
        self.context_embedder_2 = HunyuanVideo15ByT5TextProjection(
            text_embed_2_dim, 2048, inner_dim
        )

        self.time_embed = HunyuanVideo15TimeEmbedding(inner_dim, use_meanflow=use_meanflow)

        self.cond_type_embed = Embedding(3, inner_dim)

        # 2. RoPE
        self.rope = HunyuanVideo15RotaryPosEmbed(
            patch_size, patch_size_t, list(rope_axes_dim), rope_theta
        )

        # 3. Dual stream transformer blocks
        self.transformer_blocks = nn.ModuleList([
            HunyuanVideo15TransformerBlock(
                num_attention_heads, attention_head_dim,
                mlp_ratio=mlp_ratio, qk_norm=qk_norm,
            )
            for _ in range(num_layers)
        ])

        # 5. Output projection
        self.norm_out = AdaLNContinuous(inner_dim, inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = Linear(inner_dim, patch_size_t * patch_size * patch_size * out_channels, bias=True)

    @property
    def dtype(self) -> torch.dtype:
        return self.proj_out.weight.dtype

    @contextlib.contextmanager
    def cache_context(self, context_name: str = ""):
        """No-op context manager for pipeline compatibility with CacheMixin."""
        yield

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        timestep_r: Optional[torch.LongTensor] = None,
        encoder_hidden_states_2: Optional[torch.Tensor] = None,
        encoder_attention_mask_2: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor], Any]:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size_t, self.patch_size, self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # 1. RoPE
        image_rotary_emb = self.rope(hidden_states)

        # 2. Conditional embeddings
        temb = self.time_embed(timestep, timestep_r=timestep_r)

        hidden_states = self.x_embedder(hidden_states)

        # Qwen text embedding
        encoder_hidden_states = self.context_embedder(
            encoder_hidden_states, timestep, encoder_attention_mask
        )

        encoder_hidden_states_cond_emb = self.cond_type_embed(
            torch.zeros_like(encoder_hidden_states[:, :, 0], dtype=torch.long)
        )
        encoder_hidden_states = encoder_hidden_states + encoder_hidden_states_cond_emb

        # ByT5 text embedding
        encoder_hidden_states_2 = self.context_embedder_2(encoder_hidden_states_2)

        encoder_hidden_states_2_cond_emb = self.cond_type_embed(
            torch.ones_like(encoder_hidden_states_2[:, :, 0], dtype=torch.long)
        )
        encoder_hidden_states_2 = encoder_hidden_states_2 + encoder_hidden_states_2_cond_emb

        # Image embedding
        encoder_hidden_states_3 = self.image_embedder(image_embeds)
        is_t2v = torch.all(image_embeds == 0)
        if is_t2v:
            encoder_hidden_states_3 = encoder_hidden_states_3 * 0.0
            encoder_attention_mask_3 = torch.zeros(
                (batch_size, encoder_hidden_states_3.shape[1]),
                dtype=encoder_attention_mask.dtype,
                device=encoder_attention_mask.device,
            )
        else:
            encoder_attention_mask_3 = torch.ones(
                (batch_size, encoder_hidden_states_3.shape[1]),
                dtype=encoder_attention_mask.dtype,
                device=encoder_attention_mask.device,
            )
        encoder_hidden_states_3_cond_emb = self.cond_type_embed(
            2 * torch.ones_like(
                encoder_hidden_states_3[:, :, 0], dtype=torch.long,
            )
        )
        encoder_hidden_states_3 = encoder_hidden_states_3 + encoder_hidden_states_3_cond_emb

        # Reorder and combine text tokens: valid tokens first, then padding
        encoder_attention_mask = encoder_attention_mask.bool()
        encoder_attention_mask_2 = encoder_attention_mask_2.bool()
        encoder_attention_mask_3 = encoder_attention_mask_3.bool()
        new_encoder_hidden_states = []
        new_encoder_attention_mask = []

        for text, text_mask, text_2, text_mask_2, image, image_mask in zip(
            encoder_hidden_states,
            encoder_attention_mask,
            encoder_hidden_states_2,
            encoder_attention_mask_2,
            encoder_hidden_states_3,
            encoder_attention_mask_3,
        ):
            new_encoder_hidden_states.append(
                torch.cat([
                    image[image_mask],
                    text_2[text_mask_2],
                    text[text_mask],
                    image[~image_mask],
                    torch.zeros_like(text_2[~text_mask_2]),
                    torch.zeros_like(text[~text_mask]),
                ], dim=0)
            )
            new_encoder_attention_mask.append(
                torch.cat([
                    image_mask[image_mask],
                    text_mask_2[text_mask_2],
                    text_mask[text_mask],
                    image_mask[~image_mask],
                    text_mask_2[~text_mask_2],
                    text_mask[~text_mask],
                ], dim=0)
            )

        encoder_hidden_states = torch.stack(new_encoder_hidden_states)
        encoder_attention_mask = torch.stack(new_encoder_attention_mask)

        # 4. Transformer blocks
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                encoder_attention_mask,
                image_rotary_emb,
            )

        # 5. Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width,
            -1, p_t, p_h, p_w
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (hidden_states,)

        # Return as a simple namespace with .sample attribute
        return type('Output', (), {'sample': hidden_states})()


# Alias for registration
HunyuanVideo15 = HunyuanVideo15Transformer


# ============================================================================
# Pipeline
# ============================================================================

class HunyuanVideo15Pipeline:
    """Minimal pipeline for HunyuanVideo 1.5 end-to-end testing.

    Wraps the transformer, VAE, text encoders, and scheduler
    for text-to-video generation.
    """

    def __init__(
        self,
        transformer,
        scheduler,
        vae,
        text_encoder,
        tokenizer,
        text_encoder_2,
        tokenizer_2,
    ):
        self.transformer = transformer
        self.scheduler = scheduler
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.text_encoder_2 = text_encoder_2
        self.tokenizer_2 = tokenizer_2

    def to(self, device):
        self.transformer = self.transformer.to(device)
        return self

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 121,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        output_type: str = "np",
        return_dict: bool = True,
        **kwargs,
    ):
        raise NotImplementedError(
            "Full pipeline generation not yet implemented for HunyuanVideo 1.5. "
            "Use model-level alignment tests instead."
        )
