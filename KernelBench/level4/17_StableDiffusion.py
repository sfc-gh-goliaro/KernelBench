"""
Stable Diffusion XL UNet2DConditionModel

Implements the UNet2DConditionModel architecture matching
stabilityai/stable-diffusion-xl-base-1.0 from HuggingFace diffusers.

Architecture (SDXL config):
- 4-channel latent input/output
- block_out_channels: (320, 640, 1280)
- layers_per_block: 2
- transformer_layers_per_block: [1, 2, 10]
- cross_attention_dim: 2048
- attention_head_dim: [5, 10, 20] (i.e. num_attention_heads)
- down_block_types: CrossAttnDownBlock2D, CrossAttnDownBlock2D, DownBlock2D
- up_block_types: UpBlock2D, CrossAttnUpBlock2D, CrossAttnUpBlock2D
- mid_block_type: UNetMidBlock2DCrossAttn
- addition_embed_type: "text_time" (SDXL pooled text + time_ids conditioning)
- use_linear_projection: True

The forward signature matches diffusers:
    forward(sample, timestep, encoder_hidden_states, added_cond_kwargs, ...)

This model delegates all primitive computations to level1 operators:
- level1/normalization/_3_GroupNorm       → GroupNorm
- level1/normalization/_6_LayerNorm       → LayerNorm
- level1/activations/_7_Swish            → SiLU activation
- level1/activations/_8_GELU             → GELU activation
- level1/activations/_17_GeluAndMul      → GEGLU gated activation
- level1/activations/_18_Mish            → Mish activation
- level1/matmul/_10_Linear               → Linear layers
- level1/convolutions/_8_Conv2d_Square   → 2D convolutions
- level1/attention/_2_Attention          → Scaled dot-product attention
- level1/upsampling/_2_Interpolate       → Nearest-neighbor upsampling
- level1/regularization/_1_Dropout       → Dropout regularization
- level1/diffusion/_3_TimestepEmbedding  → Timestep embedding MLP
- level1/diffusion/_7_SinusoidalTimesteps → Sinusoidal timestep encoding

Level4 code is purely wiring — no raw computation happens here.
"""

import inspect

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import Optional, Dict, Any, Tuple, List, Union

# ============================================================================
# Level1 operator imports
# ============================================================================

# Parameter-owning operators (weights stored inside wrapper)
from KernelBench.level1.normalization._3_GroupNorm import Model as GroupNorm
from KernelBench.level1.normalization._6_LayerNorm import Model as LayerNorm
from KernelBench.level1.matmul._10_Linear import Model as Linear
from KernelBench.level1.convolutions._8_Conv2d_Square import Model as Conv2d

# Diffusion-specific operators
from KernelBench.level1.diffusion._3_TimestepEmbedding import Model as TimestepEmbedding
from KernelBench.level1.diffusion._7_SinusoidalTimesteps import Model as SinusoidalTimesteps

# Parameter-free operators (no weights, pure computation)
from KernelBench.level1.activations._7_Swish import Model as Swish
from KernelBench.level1.activations._8_GELU import Model as GELU
from KernelBench.level1.activations._17_GeluAndMul import Model as GeluAndMul
from KernelBench.level1.activations._18_Mish import Model as Mish
from KernelBench.level1.attention._2_Attention import ScaledDotProductAttention
from KernelBench.level1.upsampling._2_Interpolate import Model as Interpolate
from KernelBench.level1.regularization._1_Dropout import Model as Dropout


# ============================================================================
# GEGLU Feed-Forward (matches diffusers FeedForward with GEGLU)
# ============================================================================

class GEGLU(nn.Module):
    """Matches diffusers GEGLU activation for feed-forward.

    Uses Linear level1 op for projection, GeluAndMul level1 op for gated activation.

    Note: diffusers GEGLU splits into (value, gate) and computes value * gelu(gate),
    while GeluAndMul splits into (gate, value) and computes gelu(gate) * value.
    We swap the two halves before passing to GeluAndMul to match diffusers behavior.
    """
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = Linear(dim_in, dim_out * 2, bias=True)
        self.geglu_act = GeluAndMul(approximate='none')

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        # Swap halves: diffusers GEGLU = value * gelu(gate) where value=first, gate=second
        # GeluAndMul = gelu(first) * second, so we swap to make first=gate, second=value
        value, gate = hidden_states.chunk(2, dim=-1)
        hidden_states = torch.cat([gate, value], dim=-1)
        hidden_states = self.geglu_act(hidden_states)
        return hidden_states


class FeedForward(nn.Module):
    """Matches diffusers FeedForward network inside BasicTransformerBlock."""
    def __init__(self, dim: int, dim_out: Optional[int] = None, mult: float = 4.0,
                 activation_fn: str = "geglu", dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim

        if activation_fn == "geglu":
            self.net = nn.ModuleList([
                GEGLU(dim, inner_dim),
                Dropout(p=dropout),
                Linear(inner_dim, dim_out, bias=True),
            ])
        else:
            self.net = nn.ModuleList([
                Linear(dim, inner_dim, bias=True),
                GELU(),
                Dropout(p=dropout),
                Linear(inner_dim, dim_out, bias=True),
            ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


# ============================================================================
# Attention (matches diffusers Attention processor)
# ============================================================================

class CrossAttention(nn.Module):
    """
    Attention module matching diffusers Attention class.

    Uses Linear level1 op for projections, ScaledDotProductAttention level1 op
    for the attention kernel.

    Supports both self-attention (encoder_hidden_states=None)
    and cross-attention (encoder_hidden_states provided).
    """
    def __init__(self, query_dim: int, cross_attention_dim: Optional[int] = None,
                 heads: int = 8, dim_head: int = 64, dropout: float = 0.0,
                 bias: bool = False, out_bias: bool = True, upcast_attention: bool = False):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = inner_dim
        self.scale = dim_head ** -0.5
        self.upcast_attention = upcast_attention

        self.to_q = Linear(query_dim, inner_dim, bias=bias)
        self.to_k = Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_out = nn.ModuleList([
            Linear(inner_dim, query_dim, bias=out_bias),
            Dropout(p=dropout),
        ])

        # ScaledDotProductAttention level1 op (parameter-free)
        self.sdpa = ScaledDotProductAttention(mode="sdpa")

    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        q = self.to_q(hidden_states)
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

        head_dim = self.dim_head
        q = q.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        # Delegate attention computation to level1 op
        hidden_states = self.sdpa(q, k, v, scale=self.scale)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.inner_dim)

        # to_out
        for module in self.to_out:
            hidden_states = module(hidden_states)

        return hidden_states


# ============================================================================
# BasicTransformerBlock (matches diffusers BasicTransformerBlock)
# ============================================================================

class BasicTransformerBlock(nn.Module):
    """
    Matches diffusers BasicTransformerBlock with layer_norm norm_type.
    Contains: self-attention, cross-attention, feed-forward.
    Uses LayerNorm level1 op for normalization.
    """
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int,
                 dropout: float = 0.0, cross_attention_dim: Optional[int] = None,
                 activation_fn: str = "geglu", attention_bias: bool = False,
                 only_cross_attention: bool = False, upcast_attention: bool = False,
                 norm_eps: float = 1e-5):
        super().__init__()
        self.only_cross_attention = only_cross_attention

        # 1. Self-Attention
        self.norm1 = LayerNorm(dim, eps=norm_eps)
        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )

        # 2. Cross-Attention
        if cross_attention_dim is not None:
            self.norm2 = LayerNorm(dim, eps=norm_eps)
            self.attn2 = CrossAttention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        else:
            self.norm2 = None
            self.attn2 = None

        # 3. Feed-Forward
        self.norm3 = LayerNorm(dim, eps=norm_eps)
        self.ff = FeedForward(dim, activation_fn=activation_fn, dropout=dropout)

    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. Self-Attention
        norm_hidden_states = self.norm1(hidden_states)
        if self.only_cross_attention:
            attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
        else:
            attn_output = self.attn1(norm_hidden_states)
        hidden_states = attn_output + hidden_states

        # 2. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = self.norm2(hidden_states)
            attn_output = self.attn2(norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
            hidden_states = attn_output + hidden_states

        # 3. Feed-Forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states

        return hidden_states


# ============================================================================
# Transformer2DModel (matches diffusers Transformer2DModel, continuous input)
# ============================================================================

class Transformer2DModel(nn.Module):
    """
    Matches diffusers Transformer2DModel with continuous input and use_linear_projection=True.
    Wraps BasicTransformerBlock with GroupNorm + linear proj_in/proj_out.
    Uses GroupNorm and Linear level1 ops.
    """
    def __init__(self, num_attention_heads: int, attention_head_dim: int,
                 in_channels: int, num_layers: int = 1,
                 cross_attention_dim: Optional[int] = None,
                 norm_num_groups: int = 32,
                 use_linear_projection: bool = False,
                 only_cross_attention: bool = False,
                 upcast_attention: bool = False,
                 activation_fn: str = "geglu",
                 norm_eps: float = 1e-5):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim
        self.in_channels = in_channels
        self.use_linear_projection = use_linear_projection

        self.norm = GroupNorm(num_features=in_channels, num_groups=norm_num_groups, eps=1e-6)

        if use_linear_projection:
            self.proj_in = Linear(in_channels, inner_dim, bias=True)
        else:
            self.proj_in = Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                num_attention_heads,
                attention_head_dim,
                cross_attention_dim=cross_attention_dim,
                activation_fn=activation_fn,
                only_cross_attention=only_cross_attention,
                upcast_attention=upcast_attention,
                norm_eps=norm_eps,
            )
            for _ in range(num_layers)
        ])

        if use_linear_projection:
            self.proj_out = Linear(inner_dim, in_channels, bias=True)
        else:
            self.proj_out = Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, channel, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)

        if self.use_linear_projection:
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, channel)
            hidden_states = self.proj_in(hidden_states)
        else:
            hidden_states = self.proj_in(hidden_states)
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, self.inner_dim)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states)

        if self.use_linear_projection:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(batch, height, width, channel).permute(0, 3, 1, 2)
        else:
            hidden_states = hidden_states.reshape(batch, height, width, self.inner_dim).permute(0, 3, 1, 2)
            hidden_states = self.proj_out(hidden_states)

        output = hidden_states + residual
        return output


# ============================================================================
# ResnetBlock2D (matches diffusers ResnetBlock2D with time_embedding_norm="default")
# ============================================================================

class ResnetBlock2D(nn.Module):
    """Matches diffusers ResnetBlock2D with default time embedding norm.
    Uses GroupNorm, Conv2d, Linear, Swish level1 ops."""
    def __init__(self, in_channels: int, out_channels: Optional[int] = None,
                 temb_channels: int = 512, groups: int = 32, groups_out: Optional[int] = None,
                 eps: float = 1e-6, non_linearity: str = "swish",
                 output_scale_factor: float = 1.0, dropout: float = 0.0, **kwargs):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_scale_factor = output_scale_factor
        groups_out = groups_out or groups

        self.norm1 = GroupNorm(num_features=in_channels, num_groups=groups, eps=eps)
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

        if temb_channels is not None:
            self.time_emb_proj = Linear(temb_channels, out_channels, bias=True)
        else:
            self.time_emb_proj = None

        self.norm2 = GroupNorm(num_features=out_channels, num_groups=groups_out, eps=eps)
        self.dropout = Dropout(p=dropout)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.nonlinearity = Swish()

        self.use_in_shortcut = in_channels != out_channels
        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        return output_tensor


# ============================================================================
# Downsample2D / Upsample2D (matches diffusers)
# ============================================================================

class Downsample2D(nn.Module):
    """Matches diffusers Downsample2D with use_conv=True. Uses Conv2d level1 op."""
    def __init__(self, channels: int, out_channels: Optional[int] = None,
                 padding: int = 1):
        super().__init__()
        out_channels = out_channels or channels
        self.conv = Conv2d(channels, out_channels, kernel_size=3, stride=2, padding=padding, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.conv(hidden_states)


class Upsample2D(nn.Module):
    """Matches diffusers Upsample2D with use_conv=True, name='conv'.
    Uses Conv2d and Interpolate level1 ops."""
    def __init__(self, channels: int, out_channels: Optional[int] = None):
        super().__init__()
        out_channels = out_channels or channels
        self.conv = Conv2d(channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.interpolate = Interpolate(scale_factor=2.0, mode='nearest')

    def forward(self, hidden_states: torch.Tensor, output_size: Optional[int] = None) -> torch.Tensor:
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        if output_size is not None:
            hidden_states = self.interpolate(hidden_states, size=output_size)
        else:
            hidden_states = self.interpolate(hidden_states)

        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        hidden_states = self.conv(hidden_states)
        return hidden_states


# ============================================================================
# UNet Blocks (matches diffusers CrossAttnDownBlock2D, DownBlock2D,
#              CrossAttnUpBlock2D, UpBlock2D, UNetMidBlock2DCrossAttn)
# ============================================================================

class CrossAttnDownBlock2D(nn.Module):
    """Matches diffusers CrossAttnDownBlock2D."""
    def __init__(self, in_channels: int, out_channels: int, temb_channels: int,
                 num_layers: int = 2, transformer_layers_per_block: int = 1,
                 num_attention_heads: int = 1, cross_attention_dim: int = 1280,
                 resnet_groups: int = 32, resnet_eps: float = 1e-5,
                 add_downsample: bool = True, downsample_padding: int = 1,
                 use_linear_projection: bool = False, only_cross_attention: bool = False,
                 upcast_attention: bool = False):
        super().__init__()
        self.has_cross_attention = True
        resnets = []
        attentions = []

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(
                in_channels=in_ch, out_channels=out_channels,
                temb_channels=temb_channels, groups=resnet_groups, eps=resnet_eps,
            ))
            attentions.append(Transformer2DModel(
                num_attention_heads,
                out_channels // num_attention_heads,
                in_channels=out_channels,
                num_layers=transformer_layers_per_block[i],
                cross_attention_dim=cross_attention_dim,
                norm_num_groups=resnet_groups,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention,
                upcast_attention=upcast_attention,
            ))

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

        if add_downsample:
            self.downsamplers = nn.ModuleList([
                Downsample2D(out_channels, out_channels=out_channels, padding=downsample_padding)
            ])
        else:
            self.downsamplers = None

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                **kwargs) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        output_states = ()
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)
            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class DownBlock2D(nn.Module):
    """Matches diffusers DownBlock2D (no attention)."""
    def __init__(self, in_channels: int, out_channels: int, temb_channels: int,
                 num_layers: int = 2, resnet_groups: int = 32, resnet_eps: float = 1e-5,
                 add_downsample: bool = True, downsample_padding: int = 1):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(
                in_channels=in_ch, out_channels=out_channels,
                temb_channels=temb_channels, groups=resnet_groups, eps=resnet_eps,
            ))
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList([
                Downsample2D(out_channels, out_channels=out_channels, padding=downsample_padding)
            ])
        else:
            self.downsamplers = None

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor,
                **kwargs) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        output_states = ()
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class UNetMidBlock2DCrossAttn(nn.Module):
    """Matches diffusers UNetMidBlock2DCrossAttn."""
    def __init__(self, in_channels: int, temb_channels: int,
                 num_layers: int = 1, transformer_layers_per_block: int = 1,
                 num_attention_heads: int = 1, cross_attention_dim: int = 1280,
                 resnet_groups: int = 32, resnet_eps: float = 1e-5,
                 output_scale_factor: float = 1.0,
                 use_linear_projection: bool = False,
                 upcast_attention: bool = False):
        super().__init__()
        self.has_cross_attention = True

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        # First resnet
        resnets = [ResnetBlock2D(
            in_channels=in_channels, out_channels=in_channels,
            temb_channels=temb_channels, groups=resnet_groups, eps=resnet_eps,
            output_scale_factor=output_scale_factor,
        )]
        attentions = []

        for i in range(num_layers):
            attentions.append(Transformer2DModel(
                num_attention_heads,
                in_channels // num_attention_heads,
                in_channels=in_channels,
                num_layers=transformer_layers_per_block[i],
                cross_attention_dim=cross_attention_dim,
                norm_num_groups=resnet_groups,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
            ))
            resnets.append(ResnetBlock2D(
                in_channels=in_channels, out_channels=in_channels,
                temb_channels=temb_channels, groups=resnet_groups, eps=resnet_eps,
                output_scale_factor=output_scale_factor,
            ))

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)
            hidden_states = resnet(hidden_states, temb)
        return hidden_states


class CrossAttnUpBlock2D(nn.Module):
    """Matches diffusers CrossAttnUpBlock2D."""
    def __init__(self, in_channels: int, out_channels: int, prev_output_channel: int,
                 temb_channels: int, num_layers: int = 2,
                 transformer_layers_per_block: int = 1,
                 num_attention_heads: int = 1, cross_attention_dim: int = 1280,
                 resnet_groups: int = 32, resnet_eps: float = 1e-5,
                 add_upsample: bool = True, resolution_idx: Optional[int] = None,
                 use_linear_projection: bool = False,
                 only_cross_attention: bool = False,
                 upcast_attention: bool = False):
        super().__init__()
        self.has_cross_attention = True
        resnets = []
        attentions = []

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(ResnetBlock2D(
                in_channels=resnet_in_channels + res_skip_channels,
                out_channels=out_channels,
                temb_channels=temb_channels, groups=resnet_groups, eps=resnet_eps,
            ))
            attentions.append(Transformer2DModel(
                num_attention_heads,
                out_channels // num_attention_heads,
                in_channels=out_channels,
                num_layers=transformer_layers_per_block[i],
                cross_attention_dim=cross_attention_dim,
                norm_num_groups=resnet_groups,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention,
                upcast_attention=upcast_attention,
            ))

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.resolution_idx = resolution_idx

    def forward(self, hidden_states: torch.Tensor,
                res_hidden_states_tuple: Tuple[torch.Tensor, ...],
                temb: torch.Tensor,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                upsample_size: Optional[int] = None,
                **kwargs) -> torch.Tensor:
        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


class UpBlock2D(nn.Module):
    """Matches diffusers UpBlock2D (no attention)."""
    def __init__(self, in_channels: int, out_channels: int, prev_output_channel: int,
                 temb_channels: int, num_layers: int = 2,
                 resnet_groups: int = 32, resnet_eps: float = 1e-5,
                 add_upsample: bool = True, resolution_idx: Optional[int] = None):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(ResnetBlock2D(
                in_channels=resnet_in_channels + res_skip_channels,
                out_channels=out_channels,
                temb_channels=temb_channels, groups=resnet_groups, eps=resnet_eps,
            ))

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.resolution_idx = resolution_idx

    def forward(self, hidden_states: torch.Tensor,
                res_hidden_states_tuple: Tuple[torch.Tensor, ...],
                temb: torch.Tensor,
                upsample_size: Optional[int] = None,
                **kwargs) -> torch.Tensor:
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


# ============================================================================
# Main Model Class: UNet2DConditionModel for SDXL
# ============================================================================

class StableDiffusionXL(nn.Module):
    """
    UNet2DConditionModel matching stabilityai/stable-diffusion-xl-base-1.0.

    The forward signature matches the diffusers UNet2DConditionModel:
        forward(sample, timestep, encoder_hidden_states, added_cond_kwargs=None, return_dict=False)

    SDXL-specific features:
    - addition_embed_type="text_time": uses pooled text embeddings + time_ids
    - use_linear_projection=True in transformer blocks
    - Asymmetric transformer layers: [1, 2, 10] for down blocks, [10, 2, 1] reversed for up blocks

    All primitive computations are delegated to level1 operators.
    """

    def __init__(
        self,
        sample_size: Optional[int] = 128,
        in_channels: int = 4,
        out_channels: int = 4,
        down_block_types: Tuple[str, ...] = (
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
        ),
        up_block_types: Tuple[str, ...] = (
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "UpBlock2D",
        ),
        block_out_channels: Tuple[int, ...] = (320, 640, 1280),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 2048,
        transformer_layers_per_block: Union[int, List[int]] = [1, 2, 10],
        attention_head_dim: Union[int, List[int]] = [5, 10, 20],
        use_linear_projection: bool = True,
        addition_embed_type: str = "text_time",
        addition_time_embed_dim: int = 256,
        projection_class_embeddings_input_dim: int = 2816,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        act_fn: str = "silu",
        mid_block_scale_factor: float = 1,
        **kwargs,
    ):
        super().__init__()
        self.sample_size = sample_size

        # Store a config object so the model can be used as a drop-in
        # replacement for diffusers UNet2DConditionModel in pipelines that
        # access unet.config.sample_size, unet.config.in_channels, etc.
        self._config = type("Config", (), {
            "sample_size": sample_size,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "time_cond_proj_dim": None,
            "addition_time_embed_dim": addition_time_embed_dim,
        })()

        # Normalize attention_head_dim to be num_attention_heads
        # In diffusers SDXL config, attention_head_dim is actually [5, 10, 20] = num_attention_heads
        if isinstance(attention_head_dim, int):
            num_attention_heads = (attention_head_dim,) * len(down_block_types)
        else:
            num_attention_heads = tuple(attention_head_dim)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block_list = [layers_per_block] * len(down_block_types)
        else:
            layers_per_block_list = list(layers_per_block)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim_list = [cross_attention_dim] * len(down_block_types)
        else:
            cross_attention_dim_list = list(cross_attention_dim)

        # Time embedding
        time_embed_dim = block_out_channels[0] * 4

        self.time_proj = SinusoidalTimesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(block_out_channels[0], time_embed_dim, act_fn=act_fn)

        # SDXL addition embedding (text_time)
        self.addition_embed_type = addition_embed_type
        if addition_embed_type == "text_time":
            self.add_time_proj = SinusoidalTimesteps(addition_time_embed_dim, flip_sin_to_cos, freq_shift)
            self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)

        # Input convolution
        self.conv_in = Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1, bias=True)

        # Down blocks
        self.down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            if down_block_type == "CrossAttnDownBlock2D":
                self.down_blocks.append(CrossAttnDownBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    num_layers=layers_per_block_list[i],
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    num_attention_heads=num_attention_heads[i],
                    cross_attention_dim=cross_attention_dim_list[i],
                    resnet_groups=norm_num_groups,
                    resnet_eps=norm_eps,
                    add_downsample=not is_final_block,
                    use_linear_projection=use_linear_projection,
                ))
            elif down_block_type == "DownBlock2D":
                self.down_blocks.append(DownBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    num_layers=layers_per_block_list[i],
                    resnet_groups=norm_num_groups,
                    resnet_eps=norm_eps,
                    add_downsample=not is_final_block,
                ))

        # Mid block
        self.mid_block = UNetMidBlock2DCrossAttn(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            transformer_layers_per_block=transformer_layers_per_block[-1],
            num_attention_heads=num_attention_heads[-1],
            cross_attention_dim=cross_attention_dim_list[-1],
            resnet_groups=norm_num_groups,
            resnet_eps=norm_eps,
            output_scale_factor=mid_block_scale_factor,
            use_linear_projection=use_linear_projection,
        )

        # Up blocks
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block_list))
        reversed_transformer_layers_per_block = list(reversed(transformer_layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim_list))

        self.num_upsamplers = 0
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            if up_block_type == "CrossAttnUpBlock2D":
                self.up_blocks.append(CrossAttnUpBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=time_embed_dim,
                    num_layers=reversed_layers_per_block[i] + 1,
                    transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                    num_attention_heads=reversed_num_attention_heads[i],
                    cross_attention_dim=reversed_cross_attention_dim[i],
                    resnet_groups=norm_num_groups,
                    resnet_eps=norm_eps,
                    add_upsample=add_upsample,
                    resolution_idx=i,
                    use_linear_projection=use_linear_projection,
                ))
            elif up_block_type == "UpBlock2D":
                self.up_blocks.append(UpBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=time_embed_dim,
                    num_layers=reversed_layers_per_block[i] + 1,
                    resnet_groups=norm_num_groups,
                    resnet_eps=norm_eps,
                    add_upsample=add_upsample,
                    resolution_idx=i,
                ))

        # Output
        self.conv_norm_out = GroupNorm(num_features=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        self.conv_act = Swish()
        self.conv_out = Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1, bias=True)

    @property
    def config(self):
        """Config object for pipeline compatibility (mimics diffusers FrozenDict)."""
        return self._config

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the model parameters (for pipeline compatibility)."""
        return next(self.parameters()).dtype

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Tuple]:
        """
        Forward pass matching diffusers UNet2DConditionModel.

        The signature matches the diffusers UNet so this model can be used as a
        drop-in replacement inside StableDiffusionXLPipeline. Extra arguments
        beyond what this implementation uses (e.g. timestep_cond,
        cross_attention_kwargs) are accepted for compatibility but ignored.

        Args:
            sample: Noisy latent (B, C, H, W)
            timestep: Timestep scalar or tensor
            encoder_hidden_states: Text encoder hidden states (B, seq_len, dim)
            added_cond_kwargs: Dict with 'text_embeds' and 'time_ids' for SDXL
            return_dict: If True, return dict with 'sample' key

        Returns:
            If return_dict=False: (sample,) tuple
            If return_dict=True: dict with 'sample' key
        """
        # Ensure timestep is tensor
        if not torch.is_tensor(timestep):
            dtype = torch.int64
            timestep = torch.tensor([timestep], dtype=dtype, device=sample.device)
        elif len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)

        timestep = timestep.expand(sample.shape[0])

        # 1. Time embedding
        t_emb = self.time_proj(timestep)
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb)

        # SDXL addition embedding (text_time)
        if self.addition_embed_type == "text_time" and added_cond_kwargs is not None:
            text_embeds = added_cond_kwargs["text_embeds"]
            time_ids = added_cond_kwargs["time_ids"]
            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(emb.dtype)
            aug_emb = self.add_embedding(add_embeds)
            emb = emb + aug_emb

        # 2. Pre-process
        sample = self.conv_in(sample)

        # 3. Down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, 'has_cross_attention') and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        # 4. Mid
        if hasattr(self.mid_block, 'has_cross_attention') and self.mid_block.has_cross_attention:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
            )
        else:
            sample = self.mid_block(sample, emb)

        # 5. Up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]

            if hasattr(upsample_block, 'has_cross_attention') and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                )

        # 6. Post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if return_dict:
            return {"sample": sample}

        return (sample,)


# ============================================================================
# SDXL Pipeline — full text-to-image generation
# ============================================================================
# Implements the same logic as diffusers StableDiffusionXLPipeline.__call__
# but without inheriting from DiffusionPipeline.  All heavy computation
# (text encoding, denoising, VAE decode) runs on GPU.
#
# External HF components used:
#   - text_encoder / text_encoder_2  (CLIPTextModel / CLIPTextModelWithProjection)
#   - tokenizer / tokenizer_2       (CLIPTokenizer)
#   - vae                           (AutoencoderKL)
#   - scheduler                     (EulerDiscreteScheduler or compatible)
#
# The denoiser is our own StableDiffusionXL (UNet2DConditionModel).
# ============================================================================


def _retrieve_timesteps(scheduler, num_inference_steps, device, timesteps=None,
                        sigmas=None, **kwargs):
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


class StableDiffusionXLPipeline:
    """KernelBench SDXL text-to-image pipeline.

    Drop-in replacement for ``diffusers.StableDiffusionXLPipeline`` with the
    same ``__call__`` interface (subset of parameters that matter for
    generation).  Uses our :class:`StableDiffusionXL` UNet as the denoiser.
    """

    def __init__(
        self,
        unet: "StableDiffusionXL",
        scheduler,
        vae,
        text_encoder,
        text_encoder_2,
        tokenizer,
        tokenizer_2,
    ):
        self.unet = unet
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
        self.default_sample_size = (
            self.unet._config.sample_size
            if hasattr(self.unet, "_config") and hasattr(self.unet._config, "sample_size")
            else 128
        )

    # ------------------------------------------------------------------
    # Device helpers
    # ------------------------------------------------------------------
    def to(self, device):
        self.unet = self.unet.to(device)
        self.vae = self.vae.to(device)
        self.text_encoder = self.text_encoder.to(device)
        self.text_encoder_2 = self.text_encoder_2.to(device)
        return self

    @property
    def device(self):
        return next(self.unet.parameters()).device

    @property
    def dtype(self):
        return next(self.unet.parameters()).dtype

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------
    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        device=None,
        clip_skip: Optional[int] = None,
    ):
        """Encode prompt with both CLIP text encoders.

        Returns (prompt_embeds, negative_prompt_embeds,
                 pooled_prompt_embeds, negative_pooled_prompt_embeds).
        """
        device = device or self.device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        tokenizers = [self.tokenizer, self.tokenizer_2]
        text_encoders = [self.text_encoder, self.text_encoder_2]

        prompt_embeds_list = []
        pooled_prompt_embeds = None
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(device)
            outputs = text_encoder(text_input_ids, output_hidden_states=True)

            # Pooled output from the last text encoder
            if pooled_prompt_embeds is None and outputs[0].ndim == 2:
                pooled_prompt_embeds = outputs[0]

            if clip_skip is None:
                embeds = outputs.hidden_states[-2]
            else:
                embeds = outputs.hidden_states[-(clip_skip + 2)]
            prompt_embeds_list.append(embeds)

        prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)

        # Negative prompt
        do_cfg = negative_prompt is not None or True  # always do CFG
        negative_prompt = negative_prompt or ""
        negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
        if len(negative_prompt) == 1 and batch_size > 1:
            negative_prompt = negative_prompt * batch_size

        neg_embeds_list = []
        negative_pooled_prompt_embeds = None
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            uncond_input = tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            neg_outputs = text_encoder(
                uncond_input.input_ids.to(device), output_hidden_states=True
            )
            if negative_pooled_prompt_embeds is None and neg_outputs[0].ndim == 2:
                negative_pooled_prompt_embeds = neg_outputs[0]
            neg_embeds_list.append(neg_outputs.hidden_states[-2])

        negative_prompt_embeds = torch.cat(neg_embeds_list, dim=-1)

        # Cast to text_encoder_2 dtype
        te2_dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=te2_dtype, device=device)
        negative_prompt_embeds = negative_prompt_embeds.to(dtype=te2_dtype, device=device)

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
        latents = torch.randn(shape, generator=generator, device=device,
                              dtype=dtype)
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    # ------------------------------------------------------------------
    # Time IDs for SDXL micro-conditioning
    # ------------------------------------------------------------------
    @staticmethod
    def _get_add_time_ids(original_size, crops_coords_top_left, target_size,
                          dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        return torch.tensor([add_time_ids], dtype=dtype)

    # ------------------------------------------------------------------
    # VAE decode + image postprocessing
    # ------------------------------------------------------------------
    def _decode_latents(self, latents):
        """Decode latents to images via VAE, handling fp16 upcasting."""
        force_upcast = getattr(self.vae.config, "force_upcast", False)
        needs_upcasting = (self.vae.dtype == torch.float16 and force_upcast)
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)
            latents = latents.to(torch.float32)
        elif latents.dtype != self.vae.dtype:
            latents = latents.to(self.vae.dtype)

        # Un-scale latents
        has_mean = (hasattr(self.vae.config, "latents_mean")
                    and self.vae.config.latents_mean is not None)
        has_std = (hasattr(self.vae.config, "latents_std")
                   and self.vae.config.latents_std is not None)
        if has_mean and has_std:
            mu = torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(
                latents.device, latents.dtype)
            sigma = torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(
                latents.device, latents.dtype)
            latents = latents * sigma / self.vae.config.scaling_factor + mu
        else:
            latents = latents / self.vae.config.scaling_factor

        image = self.vae.decode(latents, return_dict=False)[0]

        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

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
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
    ):
        """Generate images from text prompts.

        Mirrors the core interface of
        ``diffusers.StableDiffusionXLPipeline.__call__``.
        """
        device = self.device

        # 0. Defaults
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        do_cfg = guidance_scale > 1.0

        # 1. Encode prompt
        (prompt_embeds, negative_prompt_embeds,
         pooled_prompt_embeds, negative_pooled_prompt_embeds
         ) = self._encode_prompt(prompt, negative_prompt, device)

        # 2. Prepare timesteps
        timesteps, num_inference_steps = _retrieve_timesteps(
            self.scheduler, num_inference_steps, device)

        # 3. Prepare latents
        latents = self._prepare_latents(
            batch_size, self.unet._config.in_channels, height, width,
            prompt_embeds.dtype, device, generator)

        # 4. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        text_encoder_projection_dim = self.text_encoder_2.config.projection_dim
        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size,
            dtype=prompt_embeds.dtype)
        negative_add_time_ids = add_time_ids  # same for negative

        if do_cfg:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            add_text_embeds = torch.cat(
                [negative_pooled_prompt_embeds, add_text_embeds])
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids])

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size, 1)

        # 5. Prepare extra step kwargs
        extra_step_kwargs = {}
        if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = 0.0

        # 6. Denoising loop
        for i, t in enumerate(timesteps):
            latent_model_input = (torch.cat([latents] * 2)
                                  if do_cfg else latents)
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)

            added_cond_kwargs = {
                "text_embeds": add_text_embeds,
                "time_ids": add_time_ids,
            }
            noise_pred = self.unet(
                latent_model_input, t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = (noise_pred_uncond
                              + guidance_scale * (noise_pred_text - noise_pred_uncond))

            latents_dtype = latents.dtype
            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs,
                return_dict=False)[0]
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

        # 7. VAE decode
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
