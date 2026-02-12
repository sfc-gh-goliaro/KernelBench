"""
VAE Decoder (AutoencoderKL — decode path only)

Implements the decoder portion of the Variational Autoencoder used in
Stable Diffusion models (SDXL and SD3.5).  Aligned with HuggingFace's
``diffusers.AutoencoderKL`` decoder.

Architecture:
  - post_quant_conv (optional 1x1 Conv2d)
  - conv_in (3x3 Conv2d)
  - mid_block: ResnetBlock2D -> SelfAttention -> ResnetBlock2D
  - up_blocks: N x UpDecoderBlock2D (ResnetBlock2D * (layers_per_block+1) + Upsample2D)
  - conv_norm_out (GroupNorm) + SiLU + conv_out (3x3 Conv2d)

HuggingFace weight structure (decoder path of AutoencoderKL):
  post_quant_conv.{weight,bias}
  decoder.conv_in.{weight,bias}
  decoder.mid_block.resnets.{0,1}.{norm1,conv1,norm2,conv2}.{weight,bias}
  decoder.mid_block.attentions.0.{group_norm,to_q,to_k,to_v,to_out.0}.{weight,bias}
  decoder.up_blocks.{i}.resnets.{j}.{norm1,conv1,norm2,conv2}.{weight,bias}
  decoder.up_blocks.{i}.resnets.{j}.conv_shortcut.{weight,bias}  (when in/out differ)
  decoder.up_blocks.{i}.upsamplers.0.conv.{weight,bias}
  decoder.conv_norm_out.{weight,bias}
  decoder.conv_out.{weight,bias}

Level1 operators used:
  - GroupNorm                  from level1/normalization/_3_GroupNorm
  - Conv2d                     from level1/convolutions/_8_Conv2d_Square
  - Swish (SiLU)               from level1/activations/_7_Swish
  - Interpolate                from level1/upsampling/_2_Interpolate
  - Dropout                    from level1/regularization/_1_Dropout
  - Linear                     from level1/matmul/_10_Linear
  - ScaledDotProductAttention  from level1/attention/_2_Attention
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Union

# Level1 operator imports
from KernelBench.level1.normalization._3_GroupNorm import Model as GroupNorm
from KernelBench.level1.convolutions._8_Conv2d_Square import Model as Conv2d
from KernelBench.level1.activations._7_Swish import Model as Swish
from KernelBench.level1.upsampling._2_Interpolate import Model as Interpolate
from KernelBench.level1.regularization._1_Dropout import Model as Dropout
from KernelBench.level1.matmul._10_Linear import Model as Linear
from KernelBench.level1.attention._2_Attention import ScaledDotProductAttention


# ============================================================================
# VAE ResnetBlock2D (no time embedding — VAE doesn't use timesteps)
# ============================================================================

class VAEResnetBlock2D(nn.Module):
    """Residual block for the VAE decoder.

    Matches diffusers ResnetBlock2D with temb=None (no time conditioning).
    """

    def __init__(self, in_channels: int, out_channels: int,
                 norm_num_groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = GroupNorm(num_features=in_channels,
                               num_groups=norm_num_groups, eps=eps)
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3,
                            padding=1, bias=True)
        self.norm2 = GroupNorm(num_features=out_channels,
                               num_groups=norm_num_groups, eps=eps)
        self.dropout = Dropout(p=0.0)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3,
                            padding=1, bias=True)
        self.nonlinearity = Swish()

        if in_channels != out_channels:
            self.conv_shortcut = Conv2d(in_channels, out_channels,
                                        kernel_size=1, bias=True)
        else:
            self.conv_shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        h = self.norm1(x)
        h = self.nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual.contiguous())

        return residual + h


# ============================================================================
# VAE Self-Attention (spatial self-attention in mid block)
# ============================================================================

class VAESelfAttention(nn.Module):
    """Self-attention for the VAE mid block.

    Matches diffusers Attention with group_norm, operating on 4D spatial
    tensors (B, C, H, W).  The VAE uses single-head attention (heads=1).
    """

    def __init__(self, channels: int, num_heads: int = 1,
                 norm_num_groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // self.num_heads

        self.group_norm = GroupNorm(num_features=channels,
                                    num_groups=norm_num_groups, eps=eps)
        self.to_q = Linear(channels, channels, bias=True)
        self.to_k = Linear(channels, channels, bias=True)
        self.to_v = Linear(channels, channels, bias=True)
        self.to_out = nn.ModuleList([
            Linear(channels, channels, bias=True),
            Dropout(p=0.0),
        ])
        self.attn = ScaledDotProductAttention()

    def forward(self, hidden_states: torch.Tensor,
                temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, channel, height, width = hidden_states.shape

        # Reshape to sequence
        residual = hidden_states
        hidden_states = hidden_states.view(batch, channel, height * width)
        hidden_states = hidden_states.transpose(1, 2)  # (B, H*W, C)

        hidden_states = self.group_norm(
            hidden_states.transpose(1, 2)).transpose(1, 2)

        q = self.to_q(hidden_states)
        k = self.to_k(hidden_states)
        v = self.to_v(hidden_states)

        # Reshape to multi-head
        q = q.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)

        hidden_states = self.attn(q, k, v)

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch, -1, self.num_heads * self.head_dim)
        hidden_states = hidden_states.to(q.dtype)

        # Output projection + dropout
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        # Reshape back to spatial
        hidden_states = hidden_states.transpose(-1, -2).reshape(
            batch, channel, height, width)

        return hidden_states + residual


# ============================================================================
# VAE Mid Block
# ============================================================================

class VAEMidBlock(nn.Module):
    """Mid block: ResnetBlock -> SelfAttention -> ResnetBlock."""

    def __init__(self, channels: int, norm_num_groups: int = 32,
                 eps: float = 1e-6):
        super().__init__()
        self.attentions = nn.ModuleList([
            VAESelfAttention(channels, num_heads=1,
                             norm_num_groups=norm_num_groups, eps=eps),
        ])
        self.resnets = nn.ModuleList([
            VAEResnetBlock2D(channels, channels,
                             norm_num_groups=norm_num_groups, eps=eps),
            VAEResnetBlock2D(channels, channels,
                             norm_num_groups=norm_num_groups, eps=eps),
        ])

    def forward(self, hidden_states: torch.Tensor,
                temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states)
            hidden_states = resnet(hidden_states)
        return hidden_states


# ============================================================================
# Upsample2D
# ============================================================================

class Upsample2D(nn.Module):
    """2x nearest-neighbor upsample followed by 3x3 convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.conv = Conv2d(channels, channels, kernel_size=3, padding=1,
                           bias=True)
        self._interpolate = Interpolate(scale_factor=2.0, mode='nearest')

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()
        hidden_states = self._interpolate(hidden_states)
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)
        hidden_states = self.conv(hidden_states)
        return hidden_states


# ============================================================================
# UpDecoderBlock2D
# ============================================================================

class UpDecoderBlock2D(nn.Module):
    """Decoder up-block: (layers_per_block + 1) ResnetBlocks + optional Upsample."""

    def __init__(self, in_channels: int, out_channels: int,
                 num_layers: int = 3, add_upsample: bool = True,
                 norm_num_groups: int = 32, eps: float = 1e-6):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            res_in = in_channels if i == 0 else out_channels
            resnets.append(VAEResnetBlock2D(
                res_in, out_channels,
                norm_num_groups=norm_num_groups, eps=eps))
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels)])
        else:
            self.upsamplers = None

    def forward(self, hidden_states: torch.Tensor,
                temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        return hidden_states


# ============================================================================
# VAE Decoder
# ============================================================================

class _DecoderNetwork(nn.Module):
    """Internal decoder network (matches HF's ``Decoder`` sub-module)."""

    def __init__(self, latent_channels, out_channels, block_out_channels,
                 layers_per_block, norm_num_groups, norm_eps):
        super().__init__()
        reversed_block_out = list(reversed(block_out_channels))

        self.conv_in = Conv2d(latent_channels, reversed_block_out[0],
                              kernel_size=3, padding=1, bias=True)

        self.mid_block = VAEMidBlock(
            reversed_block_out[0],
            norm_num_groups=norm_num_groups, eps=norm_eps)

        # Up blocks: each block takes prev_out_ch -> reversed_block_out[i]
        up_blocks = []
        for i in range(len(block_out_channels)):
            is_last = (i == len(block_out_channels) - 1)
            in_ch = reversed_block_out[i]
            out_ch = reversed_block_out[i]
            # The previous block's output channels determine this block's
            # input for the first resnet. For block 0, input comes from
            # mid_block (reversed_block_out[0]). For subsequent blocks,
            # input comes from the previous block's output.
            prev_ch = reversed_block_out[i - 1] if i > 0 else reversed_block_out[0]
            up_blocks.append(UpDecoderBlock2D(
                in_channels=prev_ch,
                out_channels=out_ch,
                num_layers=layers_per_block + 1,
                add_upsample=not is_last,
                norm_num_groups=norm_num_groups,
                eps=norm_eps,
            ))
        self.up_blocks = nn.ModuleList(up_blocks)

        final_ch = reversed_block_out[-1]
        self.conv_norm_out = GroupNorm(
            num_features=final_ch, num_groups=norm_num_groups, eps=norm_eps)
        self.conv_act = Swish()
        self.conv_out = Conv2d(final_ch, out_channels, kernel_size=3,
                               padding=1, bias=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid_block(h)
        for up_block in self.up_blocks:
            h = up_block(h)
        h = self.conv_norm_out(h)
        h = self.conv_act(h)
        h = self.conv_out(h)
        return h


class VAEDecoder(nn.Module):
    """VAE Decoder matching diffusers AutoencoderKL decode path.

    Includes the optional post_quant_conv and the full decoder network.
    Call ``decode(latents)`` to get images in [-1, 1].
    """

    def __init__(
        self,
        latent_channels: int = 4,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        scaling_factor: float = 0.13025,
        shift_factor: Optional[float] = None,
        force_upcast: bool = True,
        use_post_quant_conv: bool = True,
    ):
        super().__init__()

        # Store config as a simple namespace for pipeline compatibility
        self._config = type("Config", (), {
            "block_out_channels": block_out_channels,
            "scaling_factor": scaling_factor,
            "shift_factor": shift_factor,
            "force_upcast": force_upcast,
            "latents_mean": None,
            "latents_std": None,
        })()

        # Post-quantization conv (SDXL has it, SD3.5 doesn't)
        if use_post_quant_conv:
            self.post_quant_conv = Conv2d(latent_channels, latent_channels,
                                          kernel_size=1, bias=True)
        else:
            self.post_quant_conv = None

        # Decoder network (nested to match HF key structure: decoder.*)
        self.decoder = _DecoderNetwork(
            latent_channels, out_channels, block_out_channels,
            layers_per_block, norm_num_groups, norm_eps)

    @property
    def config(self):
        return self._config

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def decode(self, z: torch.Tensor,
               return_dict: bool = True) -> Union[dict, Tuple[torch.Tensor]]:
        """Decode latents to images.

        Args:
            z: Latent tensor (B, latent_channels, H_lat, W_lat)
            return_dict: If True, return dict with 'sample' key.

        Returns:
            Decoded image tensor (B, out_channels, H, W) in [-1, 1].
        """
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)

        h = self.decoder(z)

        if not return_dict:
            return (h,)
        return {"sample": h}
