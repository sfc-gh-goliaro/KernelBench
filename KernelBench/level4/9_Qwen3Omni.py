"""
Qwen3-Omni Multimodal Model

A multimodal model supporting text, image, and audio:
- Vision encoder (Qwen-VL style with Conv3D)
- Audio encoder (Whisper-style with Conv1D)
- Language model decoder with multimodal fusion
- Talker component for speech generation

Reference: Qwen3-Omni
- Vision encoder, Audio encoder, Text decoder with multimodal fusion
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


class LayerNorm(nn.Module):
    """Layer Normalization."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


# ============ Vision Encoder ============

class VisionPatchEmbed(nn.Module):
    """Vision patch embedding."""
    def __init__(self, img_size: int = 224, patch_size: int = 14, in_channels: int = 3, embed_dim: int = 1024):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class VisionAttention(nn.Module):
    """Vision self-attention."""
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class VisionMLP(nn.Module):
    """Vision MLP with GELU."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class VisionBlock(nn.Module):
    """Vision transformer block."""
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
    """Vision encoder component."""
    def __init__(self, hidden_size: int = 1024, num_layers: int = 24, num_heads: int = 16):
        super().__init__()
        self.patch_embed = VisionPatchEmbed(embed_dim=hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, hidden_size))
        self.blocks = nn.ModuleList([VisionBlock(hidden_size, num_heads) for _ in range(num_layers)])
        self.norm = LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


# ============ Audio Encoder ============

class AudioConvEncoder(nn.Module):
    """Conv1D encoder for audio features."""
    def __init__(self, n_mels: int = 128, hidden_size: int = 1024):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        return x.transpose(1, 2)


class AudioAttention(nn.Module):
    """Audio self-attention."""
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(x)


class AudioBlock(nn.Module):
    """Audio transformer block."""
    def __init__(self, dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = AudioAttention(dim, num_heads)
        self.norm2 = LayerNorm(dim)
        self.fc1 = nn.Linear(dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        residual = x
        x = self.norm2(x)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return residual + x


class AudioEncoder(nn.Module):
    """Whisper-style audio encoder."""
    def __init__(self, n_mels: int = 128, hidden_size: int = 1024, num_layers: int = 24, num_heads: int = 16):
        super().__init__()
        self.conv = AudioConvEncoder(n_mels, hidden_size)
        self.blocks = nn.ModuleList([AudioBlock(hidden_size, num_heads, hidden_size * 4) for _ in range(num_layers)])
        self.norm = LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


# ============ Multimodal Projectors ============

class ModalityProjector(nn.Module):
    """Project modality embeddings to language model space."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ============ Language Decoder ============

class LanguageAttention(nn.Module):
    """GQA for language model."""
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.num_kv_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)

        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(causal_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.o_proj(out)


class LanguageMLP(nn.Module):
    """SwiGLU MLP."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LanguageBlock(nn.Module):
    """Language decoder block."""
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, intermediate_size: int):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = LanguageAttention(hidden_size, num_heads, num_kv_heads)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = LanguageMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class LanguageDecoder(nn.Module):
    """Language model decoder (Thinker)."""
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            LanguageBlock(hidden_size, num_heads, num_kv_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)


# ============ Talker (Speech Generation) ============

class Talker(nn.Module):
    """Speech generation component (simplified Token2Wav)."""
    def __init__(self, hidden_size: int, audio_vocab_size: int = 8192):
        super().__init__()
        self.audio_head = nn.Linear(hidden_size, audio_vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.audio_head(hidden_states)


# ============ Main Model ============

class Model(nn.Module):
    """Qwen3-Omni Multimodal Model."""
    def __init__(
        self,
        # Vision
        vision_hidden_size: int = 1024,
        vision_num_layers: int = 24,
        vision_num_heads: int = 16,
        # Audio
        n_mels: int = 128,
        audio_hidden_size: int = 1024,
        audio_num_layers: int = 24,
        audio_num_heads: int = 16,
        # Language
        vocab_size: int = 152064,
        hidden_size: int = 3584,
        num_layers: int = 28,
        num_heads: int = 28,
        num_kv_heads: int = 4,
        intermediate_size: int = 18944,
        # Talker
        audio_vocab_size: int = 8192,
    ):
        super().__init__()

        # Encoders
        self.vision_encoder = VisionEncoder(vision_hidden_size, vision_num_layers, vision_num_heads)
        self.audio_encoder = AudioEncoder(n_mels, audio_hidden_size, audio_num_layers, audio_num_heads)

        # Projectors
        self.vision_projector = ModalityProjector(vision_hidden_size, hidden_size)
        self.audio_projector = ModalityProjector(audio_hidden_size, hidden_size)

        # Language decoder (Thinker)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.decoder = LanguageDecoder(
            vocab_size, hidden_size, num_layers, num_heads, num_kv_heads, intermediate_size
        )

        # Talker
        self.talker = Talker(hidden_size, audio_vocab_size)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        audio_values: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings_list = []

        # Vision encoding
        if pixel_values is not None:
            vision_embeds = self.vision_encoder(pixel_values)
            vision_embeds = self.vision_projector(vision_embeds)
            embeddings_list.append(vision_embeds)

        # Audio encoding
        if audio_values is not None:
            audio_embeds = self.audio_encoder(audio_values)
            audio_embeds = self.audio_projector(audio_embeds)
            embeddings_list.append(audio_embeds)

        # Text embedding
        if input_ids is not None:
            text_embeds = self.embed_tokens(input_ids)
            embeddings_list.append(text_embeds)

        # Concatenate all modalities
        hidden_states = torch.cat(embeddings_list, dim=1)

        # Decode
        text_logits = self.decoder(hidden_states)

        # Talker output
        audio_logits = self.talker(hidden_states)

        return text_logits, audio_logits


# Configuration (reduced for benchmarking)
batch_size = 2
img_size = 224
audio_length = 1500
sequence_length = 64

# Vision
vision_hidden_size = 768
vision_num_layers = 6
vision_num_heads = 12

# Audio
n_mels = 128
audio_hidden_size = 768
audio_num_layers = 6
audio_num_heads = 12

# Language
vocab_size = 152064
hidden_size = 1536
num_layers = 6
num_heads = 12
num_kv_heads = 4
intermediate_size = 4096


def get_inputs():
    input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
    pixel_values = torch.randn(batch_size, 3, img_size, img_size)
    audio_values = torch.randn(batch_size, n_mels, audio_length)
    return [input_ids, pixel_values, audio_values]


def get_init_inputs():
    return [{
        'vision_hidden_size': vision_hidden_size,
        'vision_num_layers': vision_num_layers,
        'vision_num_heads': vision_num_heads,
        'n_mels': n_mels,
        'audio_hidden_size': audio_hidden_size,
        'audio_num_layers': audio_num_layers,
        'audio_num_heads': audio_num_heads,
        'vocab_size': vocab_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'num_kv_heads': num_kv_heads,
        'intermediate_size': intermediate_size,
    }]

