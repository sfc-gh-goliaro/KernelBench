"""
FLUX Diffusion Transformer

A modern diffusion model implementing FLUX.1 architecture:
- DiT-based architecture with MMDiTBlock
- AdaLN-Zero conditioning on timestep
- RoPE for spatial positions
- Flow matching for generation
- Multimodal attention for text-image fusion

Reference: FLUX.1-dev/schnell
- DiT architecture with text conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """RMS Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryEmbedding2D(nn.Module):
    """2D Rotary Position Embedding for images."""
    def __init__(self, dim: int, max_res: int = 64):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim // 2, 2).float() / (dim // 2)))
        self.register_buffer("inv_freq", inv_freq)
        self.max_res = max_res

    def forward(self, h: int, w: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        y = torch.arange(h, device=device).float()
        x = torch.arange(w, device=device).float()
        y_freqs = torch.outer(y, self.inv_freq)
        x_freqs = torch.outer(x, self.inv_freq)

        # Create 2D position encoding
        y_cos = y_freqs.cos().unsqueeze(1).expand(-1, w, -1)
        y_sin = y_freqs.sin().unsqueeze(1).expand(-1, w, -1)
        x_cos = x_freqs.cos().unsqueeze(0).expand(h, -1, -1)
        x_sin = x_freqs.sin().unsqueeze(0).expand(h, -1, -1)

        cos = torch.cat([y_cos, x_cos], dim=-1).reshape(h * w, -1)
        sin = torch.cat([y_sin, x_sin], dim=-1).reshape(h * w, -1)

        return cos, sin


def apply_rope_2d(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply 2D RoPE to queries and keys."""
    def rotate(x, cos, sin):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return rotate(q, cos, sin), rotate(k, cos, sin)


class TimestepEmbedding(nn.Module):
    """Timestep embedding with sinusoidal + MLP."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        half_dim = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half_dim, device=t.device) / half_dim)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(embedding)


class AdaLNZero(nn.Module):
    """Adaptive Layer Normalization Zero for conditioning."""
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, dim * 6)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        params = self.proj(cond)[:, None, :]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = params.chunk(6, dim=-1)
        return self.norm(x) * (1 + scale_msa) + shift_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp


class Attention(nn.Module):
    """Multi-head attention with optional RoPE."""
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if cos is not None and sin is not None:
            q, k = apply_rope_2d(q, k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        return self.proj(x)


class MLP(nn.Module):
    """MLP with GELU activation."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class MMDiTBlock(nn.Module):
    """Multimodal DiT Block for text-image joint attention."""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1_img = RMSNorm(dim)
        self.norm1_txt = RMSNorm(dim)

        self.attn = Attention(dim, num_heads)

        self.norm2_img = RMSNorm(dim)
        self.norm2_txt = RMSNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp_img = MLP(dim, mlp_hidden)
        self.mlp_txt = MLP(dim, mlp_hidden)

        # AdaLN modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 12),
        )

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        cond: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Modulation parameters
        mod = self.adaLN_modulation(cond)[:, None, :]
        (shift_img_attn, scale_img_attn, gate_img_attn,
         shift_img_mlp, scale_img_mlp, gate_img_mlp,
         shift_txt_attn, scale_txt_attn, gate_txt_attn,
         shift_txt_mlp, scale_txt_mlp, gate_txt_mlp) = mod.chunk(12, dim=-1)

        # Norm and modulate
        img_norm = self.norm1_img(img) * (1 + scale_img_attn) + shift_img_attn
        txt_norm = self.norm1_txt(txt) * (1 + scale_txt_attn) + shift_txt_attn

        # Joint attention
        x = torch.cat([img_norm, txt_norm], dim=1)
        attn_out = self.attn(x, cos, sin)
        img_attn, txt_attn = attn_out.split([img.shape[1], txt.shape[1]], dim=1)

        # Residual with gate
        img = img + gate_img_attn * img_attn
        txt = txt + gate_txt_attn * txt_attn

        # MLP
        img_norm = self.norm2_img(img) * (1 + scale_img_mlp) + shift_img_mlp
        txt_norm = self.norm2_txt(txt) * (1 + scale_txt_mlp) + shift_txt_mlp

        img = img + gate_img_mlp * self.mlp_img(img_norm)
        txt = txt + gate_txt_mlp * self.mlp_txt(txt_norm)

        return img, txt


class SingleDiTBlock(nn.Module):
    """Single-stream DiT Block for later layers."""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mod = self.adaLN_modulation(cond)[:, None, :]
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)

        x_norm = self.norm1(x) * (1 + scale_attn) + shift_attn
        x = x + gate_attn * self.attn(x_norm, cos, sin)

        x_norm = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(x_norm)

        return x


class PatchEmbed(nn.Module):
    """Patch embedding for latent images."""
    def __init__(self, in_channels: int = 16, hidden_size: int = 1152, patch_size: int = 2):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        h, w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return x, h, w


class FinalLayer(nn.Module):
    """Final layer to unpatchify."""
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.proj = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 2),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        mod = self.adaLN_modulation(cond)[:, None, :]
        shift, scale = mod.chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return self.proj(x)


class Model(nn.Module):
    """FLUX-style Diffusion Transformer."""
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        hidden_size: int = 1152,
        num_heads: int = 16,
        depth_double: int = 12,  # Double-stream blocks
        depth_single: int = 12,  # Single-stream blocks
        mlp_ratio: float = 4.0,
        patch_size: int = 2,
        text_hidden_size: int = 768,
        max_text_len: int = 77,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.out_channels = out_channels

        # Patch embedding
        self.patch_embed = PatchEmbed(in_channels, hidden_size, patch_size)

        # Text projection
        self.text_proj = nn.Linear(text_hidden_size, hidden_size)

        # Timestep embedding
        self.time_embed = TimestepEmbedding(hidden_size, hidden_size)

        # RoPE
        self.rope = RotaryEmbedding2D(hidden_size // num_heads)

        # Double-stream (MMDiT) blocks
        self.double_blocks = nn.ModuleList([
            MMDiTBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(depth_double)
        ])

        # Single-stream blocks
        self.single_blocks = nn.ModuleList([
            SingleDiTBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(depth_single)
        ])

        # Final layer
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels)

    def unpatchify(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Convert patch tokens back to image."""
        c = self.out_channels
        p = self.patch_size
        x = x.reshape(-1, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(-1, c, h * p, w * p)
        return x

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        # Patch embed
        img, h, w = self.patch_embed(x)

        # Text projection
        txt = self.text_proj(text_embeds)

        # Time conditioning
        cond = self.time_embed(t, self.hidden_size)

        # RoPE
        cos, sin = self.rope(h, w, x.device)

        # Double-stream blocks (joint attention)
        for block in self.double_blocks:
            img, txt = block(img, txt, cond, cos, sin)

        # Concatenate for single-stream
        x = torch.cat([img, txt], dim=1)

        # Single-stream blocks
        for block in self.single_blocks:
            x = block(x, cond)

        # Take only image tokens
        x = x[:, :img.shape[1]]

        # Final layer
        x = self.final_layer(x, cond)

        # Unpatchify
        x = self.unpatchify(x, h, w)

        return x


# Configuration (reduced for benchmarking)
batch_size = 2
latent_size = 32  # 32x32 latent
in_channels = 16
out_channels = 16

hidden_size = 768
num_heads = 12
depth_double = 6
depth_single = 6
patch_size = 2

text_hidden_size = 768
max_text_len = 77


def get_inputs():
    x = torch.randn(batch_size, in_channels, latent_size, latent_size)
    t = torch.rand(batch_size)
    text_embeds = torch.randn(batch_size, max_text_len, text_hidden_size)
    return [x, t, text_embeds]


def get_init_inputs():
    return [{
        'in_channels': in_channels,
        'out_channels': out_channels,
        'hidden_size': hidden_size,
        'num_heads': num_heads,
        'depth_double': depth_double,
        'depth_single': depth_single,
        'patch_size': patch_size,
        'text_hidden_size': text_hidden_size,
        'max_text_len': max_text_len,
    }]

