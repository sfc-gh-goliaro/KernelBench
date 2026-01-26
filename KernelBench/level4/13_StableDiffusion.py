"""
Stable Diffusion / FLUX Diffusion Model

Implements latent diffusion model architecture:
- UNet backbone with cross-attention
- Time step embedding
- Text conditioning via cross-attention

Variants from Table 5:
- SD-1.5: 4-channel latent, 768 CLIP embedding
- SD-XL: 4-channel latent, dual text encoders
- FLUX.1-schnell: 16-channel latent, flow matching

This model uses level1 operators from KernelBench.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, Tuple, List

# Import level1 operators (used directly - no wrapping needed)
from ..level1.normalization._3_GroupNorm import Model as GroupNorm
from ..level1.normalization._6_LayerNorm import Model as LayerNorm
from ..level1.activations._7_Swish import Model as Swish
from ..level1.activations._8_GELU import Model as GELU
from ..level1.matmul._1_MatMul import Model as MatMul


# ============================================================================
# Model Variants - configs loaded from HuggingFace
# ============================================================================

VARIANTS: Dict[str, str] = {
    "SD-1.5": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "SD-XL": "stabilityai/stable-diffusion-xl-base-1.0",
    "FLUX-schnell": "black-forest-labs/FLUX.1-schnell",
}


# ============================================================================
# Component Modules (using level1 operators)
# ============================================================================

class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""
    def __init__(self, channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(channels, time_embed_dim)
        self.act = Swish()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    @staticmethod
    def get_sinusoidal_embeddings(timesteps: torch.Tensor, channels: int) -> torch.Tensor:
        half = channels // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / half)
        args = timesteps[:, None] * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = self.get_sinusoidal_embeddings(timesteps, self.linear_1.in_features)
        emb = self.linear_1(emb)
        emb = self.act(emb)
        emb = self.linear_2(emb)
        return emb


class ResBlock(nn.Module):
    """Residual block with time conditioning using level1 operators."""
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.time_emb_proj = nn.Linear(time_embed_dim, out_channels)
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.skip_connection = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        h = h + self.time_emb_proj(time_emb)[:, :, None, None]
        h = self.out_layers(h)
        return h + self.skip_connection(x)


class CrossAttention(nn.Module):
    """Cross-attention for text conditioning using level1 operators."""
    def __init__(self, query_dim: int, context_dim: int, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.head_dim = query_dim // heads
        
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim)
        
        self.matmul = MatMul()

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        
        q = self.to_q(x).view(B, L, self.heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(B, -1, self.heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(B, -1, self.heads, self.head_dim).transpose(1, 2)
        
        attn = self.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        out = self.matmul(attn, v).transpose(1, 2).reshape(B, L, C)
        return self.to_out(out)


class SpatialTransformer(nn.Module):
    """Spatial transformer block using level1 operators."""
    def __init__(self, channels: int, context_dim: int, num_heads: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.proj_in = nn.Conv2d(channels, channels, 1)
        
        self.attn = CrossAttention(channels, context_dim, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 4),
            GELU(),
            nn.Linear(channels * 4, channels),
        )
        
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x
        
        x = self.norm(x)
        x = self.proj_in(x)
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        x = x + self.attn(x, context)
        x = x + self.ff(x)
        
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = self.proj_out(x)
        
        return x + residual


class DownBlock(nn.Module):
    """Downsampling block using level1 operators."""
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int, 
                 context_dim: int, num_res_blocks: int = 2, use_attention: bool = True):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        
        for i in range(num_res_blocks):
            ch = in_channels if i == 0 else out_channels
            self.res_blocks.append(ResBlock(ch, out_channels, time_embed_dim))
            if use_attention:
                self.attn_blocks.append(SpatialTransformer(out_channels, context_dim))
            else:
                self.attn_blocks.append(nn.Identity())
        
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        hiddens = []
        for res, attn in zip(self.res_blocks, self.attn_blocks):
            x = res(x, time_emb)
            if not isinstance(attn, nn.Identity):
                x = attn(x, context)
            hiddens.append(x)
        
        x = self.downsample(x)
        hiddens.append(x)
        return x, hiddens


class UpBlock(nn.Module):
    """Upsampling block using level1 operators."""
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int,
                 context_dim: int, num_res_blocks: int = 2, use_attention: bool = True):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        
        for i in range(num_res_blocks):
            ch = in_channels if i == 0 else out_channels
            self.res_blocks.append(ResBlock(ch * 2, out_channels, time_embed_dim))
            if use_attention:
                self.attn_blocks.append(SpatialTransformer(out_channels, context_dim))
            else:
                self.attn_blocks.append(nn.Identity())
        
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, skip_connections: List[torch.Tensor],
                time_emb: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        for i, (res, attn) in enumerate(zip(self.res_blocks, self.attn_blocks)):
            skip = skip_connections.pop() if skip_connections else torch.zeros_like(x)
            x = torch.cat([x, skip], dim=1)
            x = res(x, time_emb)
            if not isinstance(attn, nn.Identity):
                x = attn(x, context)
        
        x = self.upsample(x)
        return x


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    Latent Diffusion / Stable Diffusion UNet model.
    
    Uses level1 operators from KernelBench:
    - GroupNorm from level1/normalization/3_GroupNorm
    - LayerNorm from level1/normalization/6_LayerNorm
    - Swish/SiLU from level1/activations/7_Swish
    - GELU from level1/activations/8_GELU
    - MatMul from level1/matmul/1_MatMul
    
    Supports variants: SD-1.5, SD-XL, FLUX-schnell (configs loaded from HuggingFace)
    """
    
    VARIANTS = VARIANTS
    
    @classmethod
    def from_pretrained(cls, variant: str = "SD-XL", operator_level: Optional[OperatorLevel] = None, **kwargs):
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
        in_channels = kwargs.get('in_channels', 4)
        out_channels = kwargs.get('out_channels', 4)
        model_channels = kwargs.get('model_channels', kwargs.get('hidden_size', 320))
        num_res_blocks = kwargs.get('num_res_blocks', 2)
        context_dim = kwargs.get('context_dim', 768)
        channel_mult = kwargs.get('channel_mult', [1, 2, 4, 4])
        
        if config is None:
            config = ModelConfig(
                hidden_size=model_channels,
            )
        
        super().__init__()
        
        time_embed_dim = model_channels * 4
        self.time_embed = TimestepEmbedding(model_channels, time_embed_dim)
        
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, model_channels, 3, padding=1)
        ])
        
        ch = model_channels
        self.down_blocks = nn.ModuleList()
        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            use_attn = i in [1, 2]  # Use attention at certain resolutions
            self.down_blocks.append(DownBlock(ch, out_ch, time_embed_dim, context_dim, num_res_blocks, use_attn))
            ch = out_ch
        
        self.middle_block = nn.Sequential(
            ResBlock(ch, ch, time_embed_dim),
            SpatialTransformer(ch, context_dim),
            ResBlock(ch, ch, time_embed_dim),
        )
        
        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            use_attn = i in [1, 2]
            self.up_blocks.append(UpBlock(ch, out_ch, time_embed_dim, context_dim, num_res_blocks, use_attn))
            ch = out_ch
        
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            Swish(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        time_emb = self.time_embed(timesteps)
        
        h = self.input_blocks[0](x)
        
        skip_connections = [h]
        for down_block in self.down_blocks:
            h, hiddens = down_block(h, time_emb, context)
            skip_connections.extend(hiddens)
        
        h = self.middle_block[0](h, time_emb)
        h = self.middle_block[1](h, context)
        h = self.middle_block[2](h, time_emb)
        
        for up_block in self.up_blocks:
            h = up_block(h, skip_connections, time_emb, context)
        
        return self.out(h)


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 2
height = 64
width = 64
in_channels = 4
context_length = 77
context_dim = 768


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    timesteps = torch.randint(0, 1000, (batch_size,))
    context = torch.randn(batch_size, context_length, context_dim)
    return [x, timesteps, context]


def get_init_inputs():
    return [{
        'in_channels': in_channels,
        'out_channels': in_channels,
        'model_channels': 320,
        'num_res_blocks': 2,
        'context_dim': context_dim,
        'channel_mult': [1, 2, 4],
    }]
