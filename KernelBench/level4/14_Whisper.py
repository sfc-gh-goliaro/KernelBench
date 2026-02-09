"""
Whisper Speech Recognition Model

Implements OpenAI Whisper architecture:
- Audio encoder (CNN + Transformer)
- Text decoder (causal Transformer)
- Cross-attention for audio-text fusion

Variants from Table 5:
- Whisper-Base: encoder_dim=512, decoder_dim=512
- Whisper-Large-v3: encoder_dim=1280, decoder_dim=1280

This model uses level1 operators from KernelBench.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, Tuple

# Import level1 operators (used directly - no wrapping needed)
from ..level1.normalization._6_LayerNorm import Model as LayerNorm
from ..level1.activations._8_GELU import Model as GELU
from ..level1.embeddings._3_SinusoidalPosEmbed import Model as SinusoidalPosEmbed
from ..level1.matmul._1_MatMul import Model as MatMul


# ============================================================================
# Model Variants - configs loaded from HuggingFace
# ============================================================================

VARIANTS: Dict[str, str] = {
    "Base": "openai/whisper-base",
    "Large-v3": "openai/whisper-large-v3",
}


# ============================================================================
# Component Modules (using level1 operators)
# ============================================================================

class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding."""
    def __init__(self, dim: int, max_len: int = 3000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.shape[1]]


class MultiHeadAttention(nn.Module):
    """Multi-head attention using level1 operators."""
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.matmul = MatMul()

    def forward(
        self,
        x: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        is_cross_attention = key_value_states is not None
        kv = key_value_states if is_cross_attention else x
        kv_len = kv.shape[1]
        
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(B, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = self.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        
        if attention_mask is not None:
            attn = attn + attention_mask
        
        attn = F.softmax(attn, dim=-1)
        out = self.matmul(attn, v).transpose(1, 2).reshape(B, L, -1)
        
        return self.out_proj(out)


class WhisperMLP(nn.Module):
    """Whisper MLP using level1 operators."""
    def __init__(self, dim: int, intermediate_size: Optional[int] = None):
        super().__init__()
        intermediate = intermediate_size or dim * 4
        self.fc1 = nn.Linear(dim, intermediate)
        self.gelu = GELU()
        self.fc2 = nn.Linear(intermediate, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.gelu(self.fc1(x)))


class WhisperEncoderLayer(nn.Module):
    """Whisper encoder layer using level1 operators."""
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.self_attn_layer_norm = LayerNorm(dim)
        self.self_attn = MultiHeadAttention(dim, num_heads)
        self.final_layer_norm = LayerNorm(dim)
        self.fc = WhisperMLP(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.self_attn_layer_norm(x)
        x = residual + self.self_attn(x)
        
        residual = x
        x = self.final_layer_norm(x)
        x = residual + self.fc(x)
        
        return x


class WhisperDecoderLayer(nn.Module):
    """Whisper decoder layer with cross-attention using level1 operators."""
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.self_attn_layer_norm = LayerNorm(dim)
        self.self_attn = MultiHeadAttention(dim, num_heads)
        
        self.encoder_attn_layer_norm = LayerNorm(dim)
        self.encoder_attn = MultiHeadAttention(dim, num_heads)
        
        self.final_layer_norm = LayerNorm(dim)
        self.fc = WhisperMLP(dim)

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self attention
        residual = x
        x = self.self_attn_layer_norm(x)
        x = residual + self.self_attn(x, attention_mask=attention_mask)
        
        # Cross attention
        residual = x
        x = self.encoder_attn_layer_norm(x)
        x = residual + self.encoder_attn(x, encoder_hidden_states)
        
        # FFN
        residual = x
        x = self.final_layer_norm(x)
        x = residual + self.fc(x)
        
        return x


class WhisperEncoder(nn.Module):
    """Whisper audio encoder using level1 operators."""
    def __init__(self, n_mels: int, dim: int, num_heads: int, num_layers: int):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)
        self.gelu = GELU()
        
        self.pos_embed = SinusoidalPositionalEmbedding(dim)
        
        self.layers = nn.ModuleList([
            WhisperEncoderLayer(dim, num_heads) for _ in range(num_layers)
        ])
        
        self.layer_norm = LayerNorm(dim)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        # input_features: (batch, n_mels, time)
        x = self.gelu(self.conv1(input_features))
        x = self.gelu(self.conv2(x))
        
        x = x.permute(0, 2, 1)  # (batch, time, dim)
        x = self.pos_embed(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return self.layer_norm(x)


class WhisperDecoder(nn.Module):
    """Whisper text decoder using level1 operators."""
    def __init__(self, dim: int, num_heads: int, num_layers: int, vocab_size: int, max_len: int = 448):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_len, dim)
        
        self.layers = nn.ModuleList([
            WhisperDecoderLayer(dim, num_heads) for _ in range(num_layers)
        ])
        
        self.layer_norm = LayerNorm(dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=input_ids.device)
        
        x = self.embed_tokens(input_ids) + self.pos_embed(positions)
        
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device) * float('-inf'),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)
        
        for layer in self.layers:
            x = layer(x, encoder_hidden_states, causal_mask)
        
        return self.layer_norm(x)


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    Whisper speech recognition model.
    
    Uses level1 operators from KernelBench:
    - LayerNorm from level1/normalization/6_LayerNorm
    - GELU from level1/activations/8_GELU
    - SinusoidalPosEmbed from level1/embeddings/3_SinusoidalPosEmbed
    - MatMul from level1/matmul/1_MatMul
    
    Supports variants: Base, Large-v3 (configs loaded from HuggingFace)
    """
    
    VARIANTS = VARIANTS
    
    @classmethod
    def from_pretrained(cls, variant: str = "Base", operator_level: Optional[OperatorLevel] = None, **kwargs):
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
        encoder_dim = kwargs.get('encoder_dim', kwargs.get('hidden_size', 512))
        encoder_heads = kwargs.get('encoder_heads', kwargs.get('num_heads', 8))
        encoder_layers = kwargs.get('encoder_layers', kwargs.get('num_encoder_layers', 6))
        decoder_dim = kwargs.get('decoder_dim', kwargs.get('hidden_size', 512))
        decoder_heads = kwargs.get('decoder_heads', kwargs.get('num_heads', 8))
        decoder_layers = kwargs.get('decoder_layers', kwargs.get('num_decoder_layers', 6))
        vocab_size = kwargs.get('vocab_size', 51865)
        n_mels = kwargs.get('n_mels', 80)
        
        if config is None:
            config = ModelConfig(
                hidden_size=decoder_dim,
                num_layers=decoder_layers,
                num_heads=decoder_heads,
                vocab_size=vocab_size,
            )
        
        super().__init__()
        
        self.encoder = WhisperEncoder(n_mels, encoder_dim, encoder_heads, encoder_layers)
        self.decoder = WhisperDecoder(decoder_dim, decoder_heads, decoder_layers, vocab_size)
        
        if encoder_dim != decoder_dim:
            self.encoder_proj = nn.Linear(encoder_dim, decoder_dim)
        else:
            self.encoder_proj = nn.Identity()
        
        self.proj_out = nn.Linear(decoder_dim, vocab_size, bias=False)

    def forward(
        self,
        input_features: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        encoder_hidden_states = self.encoder(input_features)
        encoder_hidden_states = self.encoder_proj(encoder_hidden_states)
        
        decoder_output = self.decoder(decoder_input_ids, encoder_hidden_states)
        logits = self.proj_out(decoder_output)
        
        return logits


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 4
audio_len = 3000
n_mels = 80
decoder_seq_len = 128
vocab_size = 51865


def get_inputs():
    input_features = torch.randn(batch_size, n_mels, audio_len)
    decoder_input_ids = torch.randint(0, vocab_size, (batch_size, decoder_seq_len))
    return [input_features, decoder_input_ids]


def get_init_inputs():
    return [{
        'encoder_dim': 512,
        'encoder_heads': 8,
        'encoder_layers': 4,
        'decoder_dim': 512,
        'decoder_heads': 8,
        'decoder_layers': 4,
        'vocab_size': vocab_size,
        'n_mels': n_mels,
    }]
