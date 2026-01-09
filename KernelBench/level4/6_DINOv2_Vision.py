"""
DINOv2 Vision Transformer

A self-supervised vision transformer implementing DINOv2 features:
- Patch embedding via Conv2d
- Learnable CLS token
- Learnable position embeddings with interpolation support
- LayerScale for training stability
- Register tokens (optional)

Reference: DINOv2-giant
- Hidden: 1536, Heads: 24, FFN: 6144, Layers: 40
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PatchEmbed(nn.Module):
    """Image to Patch Embedding using Conv2d."""
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 1536,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""
    def __init__(
        self,
        dim: int,
        num_heads: int = 24,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    """Per-channel learnable scaling (DINOv2/CaiT feature)."""
    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class MLP(nn.Module):
    """MLP with GELU activation."""
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN (used in some DINOv2 variants)."""
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(hidden_features, out_features, bias=False)
        self.w3 = nn.Linear(in_features, hidden_features, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w2(F.silu(self.w1(x)) * self.w3(x))
        x = self.drop(x)
        return x


class Block(nn.Module):
    """DINOv2 Transformer block with LayerScale."""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = 1e-5,
        use_swiglu: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if use_swiglu:
            self.mlp = SwiGLUFFN(dim, mlp_hidden_dim, dim, drop)
        else:
            self.mlp = MLP(dim, mlp_hidden_dim, dim, drop)

        # LayerScale
        self.ls1 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.ls2 = LayerScale(dim, init_values) if init_values else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class Model(nn.Module):
    """DINOv2 Vision Transformer."""
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 1536,
        depth: int = 40,
        num_heads: int = 24,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 4,
        init_values: float = 1e-5,
        use_swiglu: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_register_tokens = num_register_tokens

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS and register tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
        else:
            self.register_tokens = None

        # Position embedding (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                embed_dim, num_heads, mlp_ratio,
                init_values=init_values, use_swiglu=use_swiglu
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # Head (for feature extraction, identity; for classification, add Linear)
        self.head = nn.Identity()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.register_tokens is not None:
            nn.init.trunc_normal_(self.register_tokens, std=0.02)

    def interpolate_pos_embed(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Interpolate position embeddings for different resolutions."""
        num_patches = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1

        if num_patches == N:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]

        dim = x.shape[-1]
        h0 = int(math.sqrt(N))
        w0 = h0

        patch_pos_embed = patch_pos_embed.reshape(1, h0, w0, dim).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(patch_pos_embed, size=(h, w), mode='bicubic', align_corners=False)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)

        return torch.cat([class_pos_embed.unsqueeze(1), patch_pos_embed], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Patch embedding
        x = self.patch_embed(x)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embedding
        h, w = H // self.patch_embed.patch_size, W // self.patch_embed.patch_size
        pos_embed = self.interpolate_pos_embed(x, h, w)
        x = x + pos_embed

        # Add register tokens (after position embedding)
        if self.register_tokens is not None:
            reg_tokens = self.register_tokens.expand(B, -1, -1)
            x = torch.cat([x[:, :1], reg_tokens, x[:, 1:]], dim=1)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Return CLS token (excluding register tokens)
        return self.head(x[:, 0])


# Configuration (reduced for benchmarking)
batch_size = 4
img_size = 224
patch_size = 14
in_channels = 3
embed_dim = 1024
depth = 12
num_heads = 16
mlp_ratio = 4.0


def get_inputs():
    return [torch.randn(batch_size, in_channels, img_size, img_size)]


def get_init_inputs():
    return [{
        'img_size': img_size,
        'patch_size': patch_size,
        'in_channels': in_channels,
        'embed_dim': embed_dim,
        'depth': depth,
        'num_heads': num_heads,
        'mlp_ratio': mlp_ratio,
        'num_register_tokens': 4,
        'init_values': 1e-5,
        'use_swiglu': True,
    }]

