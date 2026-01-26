"""
Swin Transformer V2 Vision Model

Implements Swin Transformer V2 architecture:
- Shifted window attention
- Hierarchical feature maps
- Relative position bias

Variants from Table 5:
- SwinV2-T: tiny, embed_dim=96
- SwinV2-S: small, embed_dim=96
- SwinV2-B: base, embed_dim=128
- SwinV2-L: large, embed_dim=192

This model uses level1 operators from KernelBench.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, Tuple, List

# Import level1 operators (used directly - no wrapping needed)
from ..level1.normalization._6_LayerNorm import Model as LayerNorm
from ..level1.activations._8_GELU import Model as GELU
from ..level1.activations._5_Softmax import Model as Softmax
from ..level1.matmul._1_MatMul import Model as MatMul


# ============================================================================
# Model Variants - configs loaded from HuggingFace
# ============================================================================

VARIANTS: Dict[str, str] = {
    "T": "microsoft/swinv2-tiny-patch4-window8-256",
    "S": "microsoft/swinv2-small-patch4-window8-256",
    "B": "microsoft/swinv2-base-patch4-window12-192-22k",
    "L": "microsoft/swinv2-large-patch4-window12-192-22k",
}


# ============================================================================
# Component Modules (using level1 operators)
# ============================================================================

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition into non-overlapping windows."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reverse window partition."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention using level1 operators."""
    def __init__(self, dim: int, window_size: int, num_heads: int, pretrained_window_size: int = 0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        
        # Cosine attention with learnable logit scale
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        
        # Continuous relative position bias MLP
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False)
        )
        
        # Get relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        
        # Normalize coordinates
        relative_coords[:, :, 0] = relative_coords[:, :, 0] / (window_size - 1)
        relative_coords[:, :, 1] = relative_coords[:, :, 1] / (window_size - 1)
        relative_coords = relative_coords * 8
        relative_coords = torch.sign(relative_coords) * torch.log2(torch.abs(relative_coords) + 1) / 3
        
        self.register_buffer("relative_coords_table", relative_coords.reshape(-1, 2))
        
        self.matmul = MatMul()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Cosine attention
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01)).exp()
        attn = attn * logit_scale
        
        # Relative position bias
        relative_position_bias = self.cpb_mlp(self.relative_coords_table).view(
            self.window_size ** 2, self.window_size ** 2, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + 16 * torch.sigmoid(relative_position_bias).unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = F.softmax(attn, dim=-1)
        x = self.matmul(attn, v).transpose(1, 2).reshape(B_, N, C)
        
        return self.proj(x)


class MLP(nn.Module):
    """MLP block using level1 operators."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.gelu = GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.gelu(self.fc1(x)))


class SwinTransformerBlock(nn.Module):
    """Swin Transformer block using level1 operators."""
    def __init__(self, dim: int, num_heads: int, window_size: int, shift_size: int = 0,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.norm1 = LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))
        
        self.attn_mask = None

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Pad if needed
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        
        # Cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        
        # Partition into windows
        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # Window attention
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        
        # Merge windows
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        
        # Remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        
        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        
        return x


class PatchMerging(nn.Module):
    """Patch merging layer using level1 operators."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        
        x = self.norm(x)
        x = self.reduction(x)
        
        return x, H // 2, W // 2


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    Swin Transformer V2 for image classification.
    
    Uses level1 operators from KernelBench:
    - LayerNorm from level1/normalization/6_LayerNorm
    - GELU from level1/activations/8_GELU
    - MatMul from level1/matmul/1_MatMul
    
    Supports variants: T, S, B, L (configs loaded from HuggingFace)
    """
    
    VARIANTS = VARIANTS
    
    @classmethod
    def from_pretrained(cls, variant: str = "T", operator_level: Optional[OperatorLevel] = None, **kwargs):
        """Create model with config loaded from HuggingFace."""
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}. Available: {list(VARIANTS.keys())}")
        hf_config = load_hf_config(VARIANTS[variant])
        hf_config.update(kwargs)
        return cls(operator_level=operator_level, **hf_config)
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        operator_level: Optional[OperatorLevel] = None,
        **kwargs
    ):
        embed_dim = kwargs.get('embed_dim', kwargs.get('hidden_size', 96))
        num_classes = kwargs.get('num_classes', 1000)
        
        if config is None:
            config = ModelConfig(
                hidden_size=embed_dim,
                vocab_size=num_classes,
            )
        
        super().__init__()
        
        image_size = kwargs.get('image_size', 256)
        patch_size = kwargs.get('patch_size', 4)
        depths = kwargs.get('depths', [2, 2, 6, 2])
        num_heads = kwargs.get('num_heads', [3, 6, 12, 24])
        window_size = kwargs.get('window_size', 8)
        
        self.num_stages = len(depths)
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_norm = LayerNorm(embed_dim)
        
        H = W = image_size // patch_size
        
        self.stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        for i in range(self.num_stages):
            dim = embed_dim * (2 ** i)
            blocks = nn.ModuleList([
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads[i],
                    window_size=window_size,
                    shift_size=0 if j % 2 == 0 else window_size // 2,
                )
                for j in range(depths[i])
            ])
            self.stages.append(blocks)
            
            if i < self.num_stages - 1:
                self.downsample_layers.append(PatchMerging(dim))
        
        final_dim = embed_dim * (2 ** (self.num_stages - 1))
        self.norm = LayerNorm(final_dim)
        self.head = nn.Linear(final_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        x = self.patch_embed(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.patch_norm(x)
        
        for i, stage in enumerate(self.stages):
            for block in stage:
                x = block(x, H, W)
            
            if i < self.num_stages - 1:
                x, H, W = self.downsample_layers[i](x, H, W)
        
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
image_size = 256
num_classes = 1000


def get_inputs():
    return [torch.randn(batch_size, 3, image_size, image_size)]


def get_init_inputs():
    return [{
        'image_size': image_size,
        'embed_dim': 96,
        'depths': [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 24],
        'window_size': 8,
        'num_classes': num_classes,
    }]
