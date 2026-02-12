"""
3D Causal VAE Decoder for HunyuanVideo 1.5

Implements the decode path of AutoencoderKLHunyuanVideo15 from HuggingFace
diffusers. This is a 3D causal VAE decoder that converts video latents
back to pixel space.

Architecture:
- CausalConv3d input projection (with residual shortcut)
- MidBlock: ResBlock + CausalAttention + ResBlock
- UpBlocks: ResBlocks + optional spatial/temporal upsampling
- RMS normalization + SiLU + CausalConv3d output

Key features:
- Causal temporal padding (pad only on the left in temporal dimension)
- 3D ResNet blocks with custom RMS normalization
- Causal attention in the mid block
- Pixel-shuffle-style upsampling with temporal awareness

HuggingFace weight structure (AutoencoderKLHunyuanVideo15):
    decoder.conv_in.conv.{weight,bias}
    decoder.mid_block.resnets.{i}.norm{1,2}.{gamma}
    decoder.mid_block.resnets.{i}.conv{1,2}.conv.{weight,bias}
    decoder.mid_block.attentions.{i}.norm.{gamma}
    decoder.mid_block.attentions.{i}.to_{q,k,v}.{weight,bias}
    decoder.mid_block.attentions.{i}.proj_out.{weight,bias}
    decoder.up_blocks.{i}.resnets.{j}...
    decoder.up_blocks.{i}.upsamplers.0.conv.conv.{weight,bias}
    decoder.norm_out.{gamma}
    decoder.conv_out.conv.{weight,bias}

Level1 operators used:
- Swish (SiLU) from level1/activations/_7_Swish
- ScaledDotProductAttention from level1/attention/_2_Attention
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

from KernelBench.level1.activations._7_Swish import Model as Swish
from KernelBench.level1.attention._2_Attention import ScaledDotProductAttention


# ============================================================================
# CausalConv3d
# ============================================================================

class CausalConv3d(nn.Module):
    """3D convolution with causal temporal padding.

    Matches HunyuanVideo15CausalConv3d exactly.
    Pads temporally on the left only (causal), and symmetrically on spatial dims.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        bias: bool = True,
        pad_mode: str = "replicate",
    ):
        super().__init__()
        kernel_size = (kernel_size,) * 3 if isinstance(kernel_size, int) else kernel_size

        self.pad_mode = pad_mode
        # Padding order for F.pad: (W_left, W_right, H_left, H_right, D_left, D_right)
        # D = temporal dimension, causal: pad left only
        self.time_causal_padding = (
            kernel_size[0] // 2,   # W left
            kernel_size[0] // 2,   # W right
            kernel_size[1] // 2,   # H left
            kernel_size[1] // 2,   # H right
            kernel_size[2] - 1,    # D (temporal) left - causal
            0,                     # D (temporal) right
        )

        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            bias=bias,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.pad(hidden_states, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(hidden_states)


# ============================================================================
# Custom RMS Normalization for video
# ============================================================================

class VideoRMSNorm(nn.Module):
    """RMS normalization for video tensors.

    Matches HunyuanVideo15RMS_norm exactly.
    Normalizes along channel dimension using F.normalize, then scales.

    Args:
        dim: Number of channels
        channel_first: If True, input is (B, C, ...), normalize along dim=1
        images: If True, broadcastable dims are (1, 1); else (1, 1, 1)
        bias: Whether to include learnable bias
    """

    def __init__(self, dim: int, channel_first: bool = True,
                 images: bool = True, bias: bool = False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            F.normalize(x, dim=(1 if self.channel_first else -1))
            * self.scale * self.gamma + self.bias
        )


# ============================================================================
# 3D ResNet Block
# ============================================================================

class ResnetBlock3D(nn.Module):
    """3D residual block with CausalConv3d and VideoRMSNorm.

    Matches HunyuanVideo15ResnetBlock exactly.
    """

    def __init__(self, in_channels: int, out_channels: Optional[int] = None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.swish = Swish()

        self.norm1 = VideoRMSNorm(in_channels, images=False)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3)

        self.norm2 = VideoRMSNorm(out_channels, images=False)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3)

        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv3d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.swish(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.swish(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        return hidden_states + residual


# ============================================================================
# 3D Causal Attention Block
# ============================================================================

class AttnBlock3D(nn.Module):
    """3D causal attention block for the VAE mid block.

    Matches HunyuanVideo15AttnBlock exactly.
    Uses 1x1x1 Conv3d for Q/K/V projections and causal temporal attention mask.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = VideoRMSNorm(in_channels, images=False)

        self.to_q = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.to_k = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.to_v = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, kernel_size=1)

        self.sdpa = ScaledDotProductAttention(mode="eager")

    @staticmethod
    def _prepare_causal_attention_mask(
        n_frame: int, n_hw: int, dtype, device, batch_size: int = None
    ) -> torch.Tensor:
        """Prepare a causal attention mask for 3D videos.

        Each token can attend to all tokens in the same and previous frames.
        """
        seq_len = n_frame * n_hw
        mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
        for i in range(seq_len):
            i_frame = i // n_hw
            mask[i, : (i_frame + 1) * n_hw] = 0
        if batch_size is not None:
            mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.norm(x)

        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        batch_size, channels, frames, height, width = query.shape

        # Reshape to (B, 1, F*H*W, C) for single-head attention
        query = query.reshape(batch_size, channels, frames * height * width).permute(0, 2, 1).unsqueeze(1).contiguous()
        key = key.reshape(batch_size, channels, frames * height * width).permute(0, 2, 1).unsqueeze(1).contiguous()
        value = value.reshape(batch_size, channels, frames * height * width).permute(0, 2, 1).unsqueeze(1).contiguous()

        attention_mask = self._prepare_causal_attention_mask(
            frames, height * width, query.dtype, query.device, batch_size=batch_size
        )

        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)

        # Reshape back to (B, C, F, H, W)
        x = x.squeeze(1).reshape(batch_size, frames, height, width, channels).permute(0, 4, 1, 2, 3)
        x = self.proj_out(x)

        return x + identity


# ============================================================================
# Upsample Block
# ============================================================================

class Upsample3D(nn.Module):
    """Pixel-shuffle-style upsampling with temporal awareness.

    Matches HunyuanVideo15Upsample exactly.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 add_temporal_upsample: bool = True):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_upsample else 1 * 2 * 2
        self.conv = CausalConv3d(in_channels, out_channels * factor, kernel_size=3)

        self.add_temporal_upsample = add_temporal_upsample
        self.repeats = factor * out_channels // in_channels

    @staticmethod
    def _rearrange(tensor: torch.Tensor, r1: int = 1, r2: int = 2, r3: int = 2) -> torch.Tensor:
        """Convert (b, r1*r2*r3*c, f, h, w) -> (b, c, r1*f, r2*h, r3*w)."""
        b, packed_c, f, h, w = tensor.shape
        factor = r1 * r2 * r3
        c = packed_c // factor

        tensor = tensor.view(b, r1, r2, r3, c, f, h, w)
        tensor = tensor.permute(0, 4, 5, 1, 6, 2, 7, 3)
        return tensor.reshape(b, c, f * r1, h * r2, w * r3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r1 = 2 if self.add_temporal_upsample else 1
        h = self.conv(x)

        if self.add_temporal_upsample:
            # First frame: spatial-only upsample (no temporal duplication)
            h_first = h[:, :, :1, :, :]
            h_first = self._rearrange(h_first, r1=1, r2=2, r3=2)
            h_first = h_first[:, : h_first.shape[1] // 2]
            # Remaining frames: full spatio-temporal upsample
            h_next = h[:, :, 1:, :, :]
            h_next = self._rearrange(h_next, r1=r1, r2=2, r3=2)
            h = torch.cat([h_first, h_next], dim=2)

            # Shortcut computation
            x_first = x[:, :, :1, :, :]
            x_first = self._rearrange(x_first, r1=1, r2=2, r3=2)
            x_first = x_first.repeat_interleave(repeats=self.repeats // 2, dim=1)

            x_next = x[:, :, 1:, :, :]
            x_next = self._rearrange(x_next, r1=r1, r2=2, r3=2)
            x_next = x_next.repeat_interleave(repeats=self.repeats, dim=1)
            shortcut = torch.cat([x_first, x_next], dim=2)
        else:
            h = self._rearrange(h, r1=r1, r2=2, r3=2)
            shortcut = x.repeat_interleave(repeats=self.repeats, dim=1)
            shortcut = self._rearrange(shortcut, r1=r1, r2=2, r3=2)

        return h + shortcut


# ============================================================================
# Mid Block
# ============================================================================

class MidBlock3D(nn.Module):
    """Mid block with ResBlocks and optional attention.

    Matches HunyuanVideo15MidBlock exactly.
    """

    def __init__(self, in_channels: int, num_layers: int = 1,
                 add_attention: bool = True):
        super().__init__()
        self.add_attention = add_attention

        resnets = [ResnetBlock3D(in_channels, in_channels)]
        attentions = []

        for _ in range(num_layers):
            if add_attention:
                attentions.append(AttnBlock3D(in_channels))
            else:
                attentions.append(None)
            resnets.append(ResnetBlock3D(in_channels, in_channels))

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states)

        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = attn(hidden_states)
            hidden_states = resnet(hidden_states)

        return hidden_states


# ============================================================================
# Up Block
# ============================================================================

class UpBlock3D(nn.Module):
    """Up block with ResBlocks and optional upsampling.

    Matches HunyuanVideo15UpBlock3D exactly.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        upsample_out_channels: Optional[int] = None,
        add_temporal_upsample: bool = True,
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock3D(input_channels, out_channels))

        self.resnets = nn.ModuleList(resnets)

        if upsample_out_channels is not None:
            self.upsamplers = nn.ModuleList([
                Upsample3D(
                    out_channels, out_channels=upsample_out_channels,
                    add_temporal_upsample=add_temporal_upsample,
                )
            ])
        else:
            self.upsamplers = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


# ============================================================================
# VAE 3D Decoder (main class)
# ============================================================================

class VAEDecoder3D(nn.Module):
    """3D Causal VAE Decoder matching HunyuanVideo15Decoder3D.

    Decodes video latents (B, C_latent, F, H, W) to pixel space
    (B, C_out, F_out, H_out, W_out) using causal 3D convolutions
    and pixel-shuffle upsampling.

    Args:
        in_channels: Latent channels (default 32)
        out_channels: Output pixel channels (default 3)
        block_out_channels: Channel sizes for each block (reversed from encoder)
        layers_per_block: Number of ResBlocks per up block
        spatial_compression_ratio: Total spatial downsampling factor
        temporal_compression_ratio: Total temporal downsampling factor
        upsample_match_channel: Whether upsample output matches next block's channels
        scaling_factor: Latent scaling factor
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (1024, 1024, 512, 256, 128),
        layers_per_block: int = 2,
        spatial_compression_ratio: int = 16,
        temporal_compression_ratio: int = 4,
        upsample_match_channel: bool = True,
        scaling_factor: float = 1.03682,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scaling_factor = scaling_factor
        self.repeat = block_out_channels[0] // self.in_channels

        self.conv_in = CausalConv3d(self.in_channels, block_out_channels[0], kernel_size=3)
        self.up_blocks = nn.ModuleList([])

        # Mid block
        self.mid_block = MidBlock3D(in_channels=block_out_channels[0])

        # Up blocks
        input_channel = block_out_channels[0]
        for i in range(len(block_out_channels)):
            output_channel = block_out_channels[i]

            add_spatial_upsample = i < np.log2(spatial_compression_ratio)
            add_temporal_upsample = i < np.log2(temporal_compression_ratio)
            if add_spatial_upsample or add_temporal_upsample:
                upsample_out_channels = (
                    block_out_channels[i + 1] if upsample_match_channel else output_channel
                )
                up_block = UpBlock3D(
                    num_layers=self.layers_per_block + 1,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    upsample_out_channels=upsample_out_channels,
                    add_temporal_upsample=add_temporal_upsample,
                )
                input_channel = upsample_out_channels
            else:
                up_block = UpBlock3D(
                    num_layers=self.layers_per_block + 1,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    upsample_out_channels=None,
                    add_temporal_upsample=False,
                )
                input_channel = output_channel

            self.up_blocks.append(up_block)

        # Output
        self.norm_out = VideoRMSNorm(block_out_channels[-1], images=False)
        self.conv_act = nn.SiLU()
        self.conv_out = CausalConv3d(block_out_channels[-1], out_channels, kernel_size=3)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Decode video latents to pixel space.

        Args:
            hidden_states: (B, C_latent, F, H, W) video latents

        Returns:
            (B, C_out, F_out, H_out, W_out) decoded video
        """
        hidden_states = (
            self.conv_in(hidden_states)
            + hidden_states.repeat_interleave(repeats=self.repeat, dim=1)
        )

        hidden_states = self.mid_block(hidden_states)

        for up_block in self.up_blocks:
            hidden_states = up_block(hidden_states)

        # Post-process
        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Convenience decode method (matches HF API)."""
        return self.forward(z)
