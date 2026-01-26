"""
CosyVoice Text-to-Speech Model

Implements CosyVoice TTS architecture:
- Flow-based acoustic model
- Conditional flow matching
- Neural codec vocoder integration

Variants from Table 5:
- CosyVoice-300M: 300M parameters

This model uses level1 operators from KernelBench.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, Tuple

# Import level1 operators (used directly - no wrapping needed)
from ..level1.normalization._6_LayerNorm import Model as LayerNorm
from ..level1.activations._7_Swish import Model as Swish
from ..level1.activations._8_GELU import Model as GELU
from ..level1.embeddings._3_SinusoidalPosEmbed import Model as SinusoidalPosEmbed
from ..level1.matmul._1_MatMul import Model as MatMul


# ============================================================================
# Model Variants - configs loaded from HuggingFace
# ============================================================================

VARIANTS: Dict[str, str] = {
    "300M": "FunAudioLLM/CosyVoice-300M",
}


# ============================================================================
# Component Modules (using level1 operators)
# ============================================================================

class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for timesteps."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ConditionalLayerNorm(nn.Module):
    """Conditional layer norm with style/time embedding."""
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.scale = nn.Linear(cond_dim, dim)
        self.shift = nn.Linear(cond_dim, dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        scale = self.scale(cond).unsqueeze(1)
        shift = self.shift(cond).unsqueeze(1)
        return x * (1 + scale) + shift


class FeedForward(nn.Module):
    """Feed-forward block using level1 operators."""
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden)
        self.gelu = GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.gelu(self.fc1(x)))


class CrossAttention(nn.Module):
    """Cross-attention using level1 operators."""
    def __init__(self, dim: int, context_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(context_dim, dim)
        self.v = nn.Linear(context_dim, dim)
        self.out = nn.Linear(dim, dim)
        
        self.matmul = MatMul()

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        ctx_len = context.shape[1]
        
        q = self.q(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(context).view(B, ctx_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(context).view(B, ctx_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = self.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = self.matmul(attn, v).transpose(1, 2).reshape(B, L, -1)
        
        return self.out(out)


class FlowMatchingBlock(nn.Module):
    """Flow matching transformer block using level1 operators."""
    def __init__(self, dim: int, context_dim: int, num_heads: int, cond_dim: int):
        super().__init__()
        self.norm1 = ConditionalLayerNorm(dim, cond_dim)
        self.self_attn = CrossAttention(dim, dim, num_heads)
        
        self.norm2 = ConditionalLayerNorm(dim, cond_dim)
        self.cross_attn = CrossAttention(dim, context_dim, num_heads)
        
        self.norm3 = ConditionalLayerNorm(dim, cond_dim)
        self.ff = FeedForward(dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x, cond), self.norm1(x, cond))
        x = x + self.cross_attn(self.norm2(x, cond), context)
        x = x + self.ff(self.norm3(x, cond))
        return x


class TextEncoder(nn.Module):
    """Text encoder using level1 operators."""
    def __init__(self, vocab_size: int, dim: int, num_heads: int, num_layers: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, dim))
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, num_heads, dim * 4, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norm = LayerNorm(dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        x = x + self.pos_embed[:, :x.shape[1]]
        
        for layer in self.layers:
            x = layer(x)
        
        return self.norm(x)


class FlowModel(nn.Module):
    """Conditional flow matching model using level1 operators."""
    def __init__(self, mel_dim: int, dim: int, context_dim: int, num_heads: int, num_layers: int):
        super().__init__()
        cond_dim = dim
        
        self.input_proj = nn.Linear(mel_dim, dim)
        self.time_embed = SinusoidalPositionalEmbedding(dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, cond_dim),
            Swish(),
            nn.Linear(cond_dim, cond_dim),
        )
        
        self.layers = nn.ModuleList([
            FlowMatchingBlock(dim, context_dim, num_heads, cond_dim)
            for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(dim, mel_dim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        # Time embedding
        t_emb = self.time_embed(t)
        cond = self.time_mlp(t_emb)
        
        x = self.input_proj(x)
        
        for layer in self.layers:
            x = layer(x, context, cond)
        
        return self.output_proj(x)


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    CosyVoice text-to-speech model.
    
    Uses level1 operators from KernelBench:
    - LayerNorm from level1/normalization/6_LayerNorm
    - Swish/SiLU from level1/activations/7_Swish
    - GELU from level1/activations/8_GELU
    - MatMul from level1/matmul/1_MatMul
    
    Supports variants: 300M (configs loaded from HuggingFace)
    """
    
    VARIANTS = VARIANTS
    
    @classmethod
    def from_pretrained(cls, variant: str = "300M", operator_level: Optional[OperatorLevel] = None, **kwargs):
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
        hidden_size = kwargs.get('hidden_size', kwargs.get('llm_hidden', 512))
        num_heads = kwargs.get('num_heads', kwargs.get('llm_heads', 8))
        num_layers = kwargs.get('num_layers', kwargs.get('llm_layers', 12))
        flow_layers = kwargs.get('flow_layers', 6)
        vocab_size = kwargs.get('vocab_size', kwargs.get('speech_token_size', 8192))
        num_mel_bins = kwargs.get('num_mel_bins', kwargs.get('mel_bins', 80))
        
        if config is None:
            config = ModelConfig(
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=num_heads,
                vocab_size=vocab_size,
            )
        
        super().__init__()
        
        self.text_encoder = TextEncoder(vocab_size, hidden_size, num_heads, num_layers)
        self.flow = FlowModel(num_mel_bins, hidden_size, hidden_size, num_heads, flow_layers)

    def forward(
        self,
        input_ids: torch.Tensor,
        mel_spectrogram: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        context = self.text_encoder(input_ids)
        velocity = self.flow(mel_spectrogram, timesteps, context)
        return velocity


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 4
text_len = 64
mel_len = 256
num_mel_bins = 80
vocab_size = 8192


def get_inputs():
    input_ids = torch.randint(0, vocab_size, (batch_size, text_len))
    mel = torch.randn(batch_size, mel_len, num_mel_bins)
    timesteps = torch.rand(batch_size)
    return [input_ids, mel, timesteps]


def get_init_inputs():
    return [{
        'hidden_size': 512,
        'num_heads': 8,
        'num_layers': 4,
        'flow_layers': 4,
        'vocab_size': vocab_size,
        'num_mel_bins': num_mel_bins,
    }]
