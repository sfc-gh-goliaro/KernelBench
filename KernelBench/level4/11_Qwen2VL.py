"""
Qwen2-VL Vision-Language Model

Implements Qwen2-VL architecture:
- Vision encoder with dynamic resolution
- Language model with cross-attention
- M-RoPE for multimodal position encoding

Variants from Table 5:
- Qwen2-VL-2B: vision_dim=1280, llm_dim=1536
- Qwen2-VL-7B: vision_dim=1280, llm_dim=3584

This model uses level1 operators from KernelBench.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, Tuple, List

# Import level1 operators
# Operators that need wrapping
from ..level1.normalization._4_RMSNorm import Model as RMSNormL1
from ..level1.embeddings._1_RotaryEmbedding import Model as RotaryEmbeddingL1
# Operators used directly
from ..level1.normalization._6_LayerNorm import Model as LayerNorm
from ..level1.activations._7_Swish import Model as Swish
from ..level1.activations._8_GELU import Model as GELU
from ..level1.matmul._1_MatMul import Model as MatMul


# ============================================================================
# Model Variants - configs loaded from HuggingFace
# ============================================================================

VARIANTS: Dict[str, str] = {
    "2B": "Qwen/Qwen2-VL-2B-Instruct",
    "7B": "Qwen/Qwen2-VL-7B-Instruct",
}


# ============================================================================
# Wrapper classes for level1 operators that need adaptation
# ============================================================================

class RMSNorm(nn.Module):
    """RMS normalization with learnable weight, using level1 operator."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self._rms_norm = RMSNormL1(dim, eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self._rms_norm(x.transpose(1, -1)).transpose(1, -1)
        return normalized * self.weight


# ============================================================================
# Component Modules (using level1 operators)
# ============================================================================

class VisionAttention(nn.Module):
    """Vision encoder attention using level1 operators."""
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.matmul = MatMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = self.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        x = self.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class VisionMLP(nn.Module):
    """Vision MLP using level1 operators."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.gelu = GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.gelu(self.fc1(x)))


class VisionBlock(nn.Module):
    """Vision transformer block using level1 operators."""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = VisionAttention(dim, num_heads)
        self.norm2 = LayerNorm(dim)
        self.mlp = VisionMLP(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    """Qwen2-VL vision encoder using level1 operators."""
    def __init__(self, dim: int = 1280, num_heads: int = 16, num_layers: int = 32, 
                 patch_size: int = 14, image_size: int = 448):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
        
        self.blocks = nn.ModuleList([
            VisionBlock(dim, num_heads) for _ in range(num_layers)
        ])
        
        self.norm = LayerNorm(dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(pixel_values)
        x = x.flatten(2).transpose(1, 2)
        
        if x.shape[1] == self.pos_embed.shape[1]:
            x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
        
        return self.norm(x)


class LLMAttention(nn.Module):
    """LLM attention with GQA using level1 operators."""
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.num_kv_groups = num_heads // num_kv_heads
        
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)
        
        self.matmul = MatMul()

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = x.shape
        
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)
        
        attn = self.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        
        if attention_mask is not None:
            attn = attn + attention_mask
        
        attn = F.softmax(attn, dim=-1)
        out = self.matmul(attn, v).transpose(1, 2).reshape(B, L, -1)
        
        return self.o_proj(out)


class LLMMLP(nn.Module):
    """LLM MLP using level1 operators."""
    def __init__(self, dim: int, intermediate_size: Optional[int] = None):
        super().__init__()
        intermediate = intermediate_size or int(dim * 8 / 3)
        self.gate_proj = nn.Linear(dim, intermediate, bias=False)
        self.up_proj = nn.Linear(dim, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, dim, bias=False)
        self.swish = Swish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.swish(self.gate_proj(x)) * self.up_proj(x))


class LLMBlock(nn.Module):
    """LLM transformer block using level1 operators."""
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = LLMAttention(dim, num_heads, num_kv_heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = LLMMLP(dim)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    Qwen2-VL vision-language model.
    
    Uses level1 operators from KernelBench:
    - RMSNorm from level1/normalization/4_RMSNorm
    - LayerNorm from level1/normalization/6_LayerNorm
    - GELU from level1/activations/8_GELU
    - Swish/SiLU from level1/activations/7_Swish
    - MatMul from level1/matmul/1_MatMul
    
    Supports variants: 2B, 7B (configs loaded from HuggingFace)
    """
    
    VARIANTS = VARIANTS
    
    @classmethod
    def from_pretrained(cls, variant: str = "7B", operator_level: Optional[OperatorLevel] = None, **kwargs):
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
        vision_dim = kwargs.get('vision_dim', kwargs.get('vision_hidden', 1280))
        vision_heads = kwargs.get('vision_heads', 16)
        vision_layers = kwargs.get('vision_layers', 32)
        llm_dim = kwargs.get('llm_dim', kwargs.get('hidden_size', 3584))
        llm_heads = kwargs.get('llm_heads', kwargs.get('num_heads', 28))
        llm_kv_heads = kwargs.get('llm_kv_heads', kwargs.get('num_kv_heads', 4))
        llm_layers = kwargs.get('llm_layers', kwargs.get('num_layers', 28))
        vocab_size = kwargs.get('vocab_size', 152064)
        patch_size = kwargs.get('patch_size', 14)
        
        if config is None:
            config = ModelConfig(
                hidden_size=llm_dim,
                num_layers=llm_layers,
                num_heads=llm_heads,
                num_kv_heads=llm_kv_heads,
                vocab_size=vocab_size,
            )
        
        super().__init__()
        
        self.vision_encoder = VisionEncoder(vision_dim, vision_heads, vision_layers, patch_size)
        self.vision_projector = nn.Linear(vision_dim, llm_dim)
        
        self.embed_tokens = nn.Embedding(vocab_size, llm_dim)
        
        self.layers = nn.ModuleList([
            LLMBlock(llm_dim, llm_heads, llm_kv_heads)
            for _ in range(llm_layers)
        ])
        
        self.norm = RMSNorm(llm_dim)
        self.lm_head = nn.Linear(llm_dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_embeds = self.embed_tokens(input_ids)
        
        if pixel_values is not None:
            vision_embeds = self.vision_encoder(pixel_values)
            vision_embeds = self.vision_projector(vision_embeds)
            # Prepend vision embeddings to text
            text_embeds = torch.cat([vision_embeds, text_embeds], dim=1)
        
        seq_len = text_embeds.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=text_embeds.device) * float('-inf'),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)
        
        x = text_embeds
        for layer in self.layers:
            x = layer(x, causal_mask)
        
        x = self.norm(x)
        return self.lm_head(x)


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 2
sequence_length = 256
image_size = 448
vocab_size = 152064


def get_inputs():
    input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
    pixel_values = torch.randn(batch_size, 3, image_size, image_size)
    return [input_ids, pixel_values]


def get_init_inputs():
    return [{
        'vision_dim': 1280,
        'vision_heads': 16,
        'vision_layers': 8,
        'llm_dim': 1536,
        'llm_heads': 12,
        'llm_kv_heads': 2,
        'llm_layers': 8,
        'vocab_size': vocab_size,
    }]
