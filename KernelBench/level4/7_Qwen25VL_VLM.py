"""
Qwen2.5-VL Vision-Language Model

A modern vision-language model implementing Qwen2.5-VL features:
- Conv3D patch embedding for video/image
- M-RoPE (Multi-resolution Rotary Position Embedding)
- 3D-RoPE for temporal understanding
- Vision encoder with QuickGELU
- Vision-Language fusion via cross-attention

Reference: Qwen2-VL-7B
- Hidden: 3584, Heads: 28, KV Heads: 4, FFN: 18944, Layers: 28
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """RMS Normalization."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class QuickGELU(nn.Module):
    """Fast GELU approximation."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class Conv3dPatchEmbed(nn.Module):
    """3D Patch Embedding for video/image using Conv3d."""
    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (2, 14, 14),
        in_channels: int = 3,
        embed_dim: int = 1280,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W) for video or (B, C, 1, H, W) for image
        x = self.proj(x)  # (B, embed_dim, T', H', W')
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, T'*H'*W', embed_dim)
        return x, (T, H, W)


class MultiResolutionRoPE(nn.Module):
    """Multi-resolution Rotary Position Embedding for Qwen-VL."""
    def __init__(self, dim: int, max_seq_len: int = 8192):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(
        self,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # position_ids: (B, seq_len, 3) for (t, h, w) coordinates
        # or (B, seq_len) for 1D positions
        if position_ids.dim() == 2:
            freqs = torch.outer(position_ids[0].float(), self.inv_freq)
        else:
            # 3D position encoding
            t, h, w = position_ids[..., 0], position_ids[..., 1], position_ids[..., 2]
            t_freqs = torch.outer(t[0].float(), self.inv_freq[:self.dim//6])
            h_freqs = torch.outer(h[0].float(), self.inv_freq[self.dim//6:2*self.dim//6])
            w_freqs = torch.outer(w[0].float(), self.inv_freq[2*self.dim//6:])
            freqs = torch.cat([t_freqs, h_freqs, w_freqs], dim=-1)

        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class VisionAttention(nn.Module):
    """Vision encoder attention with M-RoPE."""
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.rope = MultiResolutionRoPE(self.head_dim)

    def forward(self, x: torch.Tensor, grid_thw: Optional[Tuple[int, int, int]] = None) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Apply RoPE if grid dimensions provided
        if grid_thw is not None:
            T, H, W = grid_thw
            position_ids = torch.arange(N, device=x.device)
            cos, sin = self.rope(position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        return self.proj(x)


class VisionMLP(nn.Module):
    """Vision encoder MLP with QuickGELU."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = QuickGELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class VisionBlock(nn.Module):
    """Vision encoder transformer block."""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = VisionAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = VisionMLP(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, grid_thw: Optional[Tuple[int, int, int]] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), grid_thw)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    """Qwen-VL Vision Encoder."""
    def __init__(
        self,
        hidden_size: int = 1280,
        num_layers: int = 32,
        num_heads: int = 16,
        patch_size: Tuple[int, int, int] = (2, 14, 14),
        in_channels: int = 3,
    ):
        super().__init__()
        self.patch_embed = Conv3dPatchEmbed(patch_size, in_channels, hidden_size)
        self.blocks = nn.ModuleList([
            VisionBlock(hidden_size, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        # x: (B, C, T, H, W)
        x, grid_thw = self.patch_embed(x)

        for block in self.blocks:
            x = block(x, grid_thw)

        x = self.norm(x)
        return x, grid_thw


class VisionLanguageProjector(nn.Module):
    """MLP projector from vision to language space."""
    def __init__(self, vision_dim: int, language_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, language_dim),
            nn.GELU(),
            nn.Linear(language_dim, language_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class LanguageGQA(nn.Module):
    """Grouped-Query Attention for language model."""
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.num_kv_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE
        t = torch.arange(N, device=x.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().unsqueeze(0).unsqueeze(0), emb.sin().unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos.squeeze(0), sin.squeeze(0))

        # Expand KV
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        causal_mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(causal_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        return self.o_proj(out)


class LanguageMLP(nn.Module):
    """SwiGLU MLP for language model."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LanguageBlock(nn.Module):
    """Language model decoder block."""
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, intermediate_size: int):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = LanguageGQA(hidden_size, num_heads, num_kv_heads)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = LanguageMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Model(nn.Module):
    """Qwen2.5-VL Vision-Language Model."""
    def __init__(
        self,
        # Vision encoder
        vision_hidden_size: int = 1280,
        vision_num_layers: int = 32,
        vision_num_heads: int = 16,
        # Language model
        vocab_size: int = 152064,
        hidden_size: int = 3584,
        num_layers: int = 28,
        num_heads: int = 28,
        num_kv_heads: int = 4,
        intermediate_size: int = 18944,
    ):
        super().__init__()

        # Vision encoder
        self.vision_encoder = VisionEncoder(
            vision_hidden_size, vision_num_layers, vision_num_heads
        )

        # Vision-language projector
        self.projector = VisionLanguageProjector(vision_hidden_size, hidden_size)

        # Language model
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            LanguageBlock(hidden_size, num_heads, num_kv_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Encode vision
        vision_embeds, _ = self.vision_encoder(pixel_values)
        vision_embeds = self.projector(vision_embeds)

        # Embed text
        text_embeds = self.embed_tokens(input_ids)

        # Concatenate vision and text (simplified - actual implementation is more complex)
        hidden_states = torch.cat([vision_embeds, text_embeds], dim=1)

        # Language model
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        # Return only text logits
        return logits[:, vision_embeds.shape[1]:]


# Configuration (reduced for benchmarking)
batch_size = 2
img_size = 224
num_frames = 1
sequence_length = 128

# Vision
vision_hidden_size = 1024
vision_num_layers = 8
vision_num_heads = 16

# Language
vocab_size = 152064
hidden_size = 2048
num_layers = 6
num_heads = 16
num_kv_heads = 4
intermediate_size = 5504


def get_inputs():
    pixel_values = torch.randn(batch_size, 3, num_frames, img_size, img_size)
    input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
    return [pixel_values, input_ids]


def get_init_inputs():
    return [{
        'vision_hidden_size': vision_hidden_size,
        'vision_num_layers': vision_num_layers,
        'vision_num_heads': vision_num_heads,
        'vocab_size': vocab_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'num_kv_heads': num_kv_heads,
        'intermediate_size': intermediate_size,
    }]

