"""
Whisper Audio-Language Model

A speech recognition model implementing Whisper architecture:
- Conv1D audio encoder for mel spectrogram features
- Sinusoidal position embeddings
- Encoder-decoder with cross-attention
- GELU activation

Reference: Whisper-large-v3
- Hidden: 1280, Heads: 20, FFN: 5120, Encoder Layers: 32, Decoder Layers: 32
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, :x.size(1)]


class LayerNorm(nn.Module):
    """Layer Normalization."""
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class AudioConvEncoder(nn.Module):
    """Conv1D encoder for mel spectrogram features."""
    def __init__(self, n_mels: int = 128, hidden_size: int = 1280):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_mels, time)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        return x.transpose(1, 2)  # (B, time/2, hidden_size)


class MultiHeadAttention(nn.Module):
    """Multi-head attention."""
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        B, tgt_len, _ = query.shape
        src_len = key.shape[1]

        q = self.q_proj(query).view(B, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if is_causal:
            causal_mask = torch.triu(torch.ones(tgt_len, src_len, device=query.device), diagonal=1).bool()
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, tgt_len, -1)
        return self.out_proj(attn_output)


class EncoderLayer(nn.Module):
    """Whisper encoder layer."""
    def __init__(self, hidden_size: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden_size, num_heads)
        self.self_attn_layer_norm = LayerNorm(hidden_size)

        self.fc1 = nn.Linear(hidden_size, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_size)
        self.final_layer_norm = LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, x, x)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x


class DecoderLayer(nn.Module):
    """Whisper decoder layer with cross-attention."""
    def __init__(self, hidden_size: int, num_heads: int, ffn_dim: int):
        super().__init__()
        # Self-attention
        self.self_attn = MultiHeadAttention(hidden_size, num_heads)
        self.self_attn_layer_norm = LayerNorm(hidden_size)

        # Cross-attention
        self.encoder_attn = MultiHeadAttention(hidden_size, num_heads)
        self.encoder_attn_layer_norm = LayerNorm(hidden_size)

        # FFN
        self.fc1 = nn.Linear(hidden_size, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_size)
        self.final_layer_norm = LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Self-attention (causal)
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, x, x, is_causal=True)
        x = residual + x

        # Cross-attention
        residual = x
        x = self.encoder_attn_layer_norm(x)
        x = self.encoder_attn(x, encoder_hidden_states, encoder_hidden_states)
        x = residual + x

        # FFN
        residual = x
        x = self.final_layer_norm(x)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x


class WhisperEncoder(nn.Module):
    """Whisper audio encoder."""
    def __init__(
        self,
        n_mels: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        max_source_positions: int = 1500,
    ):
        super().__init__()
        self.conv = AudioConvEncoder(n_mels, hidden_size)
        self.embed_positions = SinusoidalPositionalEmbedding(hidden_size, max_source_positions)

        self.layers = nn.ModuleList([
            EncoderLayer(hidden_size, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])
        self.layer_norm = LayerNorm(hidden_size)

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        # mel_spectrogram: (B, n_mels, time)
        x = self.conv(mel_spectrogram)
        x = x + self.embed_positions(x)

        for layer in self.layers:
            x = layer(x)

        x = self.layer_norm(x)
        return x


class WhisperDecoder(nn.Module):
    """Whisper text decoder."""
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        max_target_positions: int = 448,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.embed_positions = SinusoidalPositionalEmbedding(hidden_size, max_target_positions)

        self.layers = nn.ModuleList([
            DecoderLayer(hidden_size, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])
        self.layer_norm = LayerNorm(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        x = x + self.embed_positions(x)

        for layer in self.layers:
            x = layer(x, encoder_hidden_states)

        x = self.layer_norm(x)
        return x


class Model(nn.Module):
    """Whisper Audio-Language Model."""
    def __init__(
        self,
        vocab_size: int = 51865,
        n_mels: int = 128,
        hidden_size: int = 1280,
        encoder_layers: int = 32,
        decoder_layers: int = 32,
        num_heads: int = 20,
        ffn_dim: int = 5120,
    ):
        super().__init__()
        self.encoder = WhisperEncoder(
            n_mels, hidden_size, encoder_layers, num_heads, ffn_dim
        )
        self.decoder = WhisperDecoder(
            vocab_size, hidden_size, decoder_layers, num_heads, ffn_dim
        )
        self.proj_out = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        mel_spectrogram: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        encoder_hidden_states = self.encoder(mel_spectrogram)
        decoder_hidden_states = self.decoder(decoder_input_ids, encoder_hidden_states)
        logits = self.proj_out(decoder_hidden_states)
        return logits


# Configuration (reduced for benchmarking)
batch_size = 4
n_mels = 128
audio_length = 3000  # ~30 seconds of audio
decoder_seq_length = 64

vocab_size = 51865
hidden_size = 768
encoder_layers = 6
decoder_layers = 6
num_heads = 12
ffn_dim = 3072


def get_inputs():
    mel_spectrogram = torch.randn(batch_size, n_mels, audio_length)
    decoder_input_ids = torch.randint(0, vocab_size, (batch_size, decoder_seq_length))
    return [mel_spectrogram, decoder_input_ids]


def get_init_inputs():
    return [{
        'vocab_size': vocab_size,
        'n_mels': n_mels,
        'hidden_size': hidden_size,
        'encoder_layers': encoder_layers,
        'decoder_layers': decoder_layers,
        'num_heads': num_heads,
        'ffn_dim': ffn_dim,
    }]

