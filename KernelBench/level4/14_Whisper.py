"""
Whisper Speech Recognition Model

Implements OpenAI Whisper architecture aligned with HuggingFace
WhisperForConditionalGeneration:
- Audio encoder: Conv1d front-end + sinusoidal positional embeddings + Transformer
- Text decoder: learned positional embeddings + causal Transformer with cross-attention
- Shared vocab projection (proj_out tied to decoder.embed_tokens)

HuggingFace weight structure (WhisperForConditionalGeneration):
  model.encoder.conv1.{weight,bias}
  model.encoder.conv2.{weight,bias}
  model.encoder.embed_positions.weight                          (frozen sinusoidal)
  model.encoder.layers.{i}.self_attn.{q_proj,k_proj,v_proj,out_proj}.{weight,bias}
  model.encoder.layers.{i}.self_attn_layer_norm.{weight,bias}
  model.encoder.layers.{i}.fc1.{weight,bias}
  model.encoder.layers.{i}.fc2.{weight,bias}
  model.encoder.layers.{i}.final_layer_norm.{weight,bias}
  model.encoder.layer_norm.{weight,bias}
  model.decoder.embed_tokens.weight
  model.decoder.embed_positions.weight                          (learned)
  model.decoder.layers.{i}.self_attn.{q_proj,k_proj,v_proj,out_proj}.{weight,bias}
  model.decoder.layers.{i}.self_attn_layer_norm.{weight,bias}
  model.decoder.layers.{i}.encoder_attn.{q_proj,k_proj,v_proj,out_proj}.{weight,bias}
  model.decoder.layers.{i}.encoder_attn_layer_norm.{weight,bias}
  model.decoder.layers.{i}.fc1.{weight,bias}
  model.decoder.layers.{i}.fc2.{weight,bias}
  model.decoder.layers.{i}.final_layer_norm.{weight,bias}
  model.decoder.layer_norm.{weight,bias}
  proj_out.weight                                               (tied to embed_tokens)

Tested against: openai/whisper-large-v2

This model uses level1 operators from KernelBench:
- ScaledDotProductAttention from level1/attention/_2_Attention
- GELU from level1/activations/_8_GELU
- MelSpectrogram from level1/audio/_1_MelSpectrogram (in WhisperFeatureExtractor)
- Linear from level1/matmul/_10_Linear
- LayerNorm from level1/normalization/_6_LayerNorm
- Conv1d from level1/convolutions/_10_Conv1d_Standard
- Embedding from level1/embeddings/_2_Embedding

Note: Using level1 wrappers changes the state-dict key names (e.g.
LayerNorm adds ".ln.", Embedding adds ".embedding.", Conv1d adds ".conv1d.").
The weight-copying logic in test_hf_alignment.py unwraps these prefixes
when mapping KB keys to HF keys.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Dict, Any

# Import level1 operators
from ..level1.attention._2_Attention import ScaledDotProductAttention
from ..level1.activations._8_GELU import Model as GELU
from ..level1.audio._1_MelSpectrogram import Model as MelSpectrogram
from ..level1.audio._1_MelSpectrogram import create_mel_filterbank_slaney
from ..level1.matmul._10_Linear import Model as Linear
from ..level1.normalization._6_LayerNorm import Model as LayerNorm
from ..level1.convolutions._10_Conv1d_Standard import Model as Conv1d
from ..level1.embeddings._2_Embedding import Model as Embedding


# ============================================================================
# Feature Extraction (preprocessing)
# ============================================================================

# Whisper audio hyper-parameters
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30  # seconds
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk


class WhisperFeatureExtractor(nn.Module):
    """
    Whisper-compatible audio feature extractor.

    Converts raw audio waveforms to log-mel spectrogram features that match
    the HuggingFace WhisperFeatureExtractor / OpenAI Whisper `log_mel_spectrogram`.

    Uses level1 operators from KernelBench:
    - MelSpectrogram from level1/audio/_1_MelSpectrogram (with Slaney mel filters)

    Pipeline:
        1. Pad or trim waveform to 30 seconds (480000 samples)
        2. Compute STFT -> power spectrogram -> mel filterbank (via MelSpectrogram)
        3. Truncate last STFT frame (Whisper convention)
        4. log10 -> clamp(max - 8.0) -> (x + 4.0) / 4.0

    Shapes:
        Input:  (batch, samples) raw audio waveform at 16 kHz
        Output: (batch, n_mels, 3000) log-mel spectrogram features
    """

    def __init__(self, n_mels: int = 80, n_fft: int = N_FFT,
                 hop_length: int = HOP_LENGTH, sample_rate: int = SAMPLE_RATE,
                 chunk_length: int = CHUNK_LENGTH):
        super().__init__()
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.chunk_length = chunk_length
        self.n_samples = chunk_length * sample_rate

        # Build Slaney-normalized mel filterbank matching HuggingFace/librosa
        num_freq_bins = 1 + n_fft // 2  # 201
        mel_filters_np = create_mel_filterbank_slaney(
            num_frequency_bins=num_freq_bins,
            num_mel_filters=n_mels,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=sample_rate,
        )
        # mel_filters_np: (num_freq_bins, n_mels) — will be transposed inside MelSpectrogram

        # Use level1 MelSpectrogram with pre-computed Slaney filters and log10 mode
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=0.0,
            f_max=8000.0,
            mel_scale="slaney",
            log_mel="log10",
            mel_filters_np=mel_filters_np,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract log-mel spectrogram features from raw audio.

        Args:
            waveform: (batch, samples) raw audio at 16 kHz, float32

        Returns:
            (batch, n_mels, 3000) log-mel spectrogram features
        """
        # Feature extraction must be done in float32 for numerical accuracy
        # (matches HuggingFace / OpenAI Whisper behavior).
        # MelSpectrogram uses authoritative float32 numpy copies of its buffers
        # so precision is preserved even when the parent model is cast to bf16.
        orig_dtype = waveform.dtype
        waveform = waveform.float()

        # 1. Pad or trim to exactly n_samples (30s = 480000 samples)
        if waveform.shape[-1] > self.n_samples:
            waveform = waveform[..., :self.n_samples]
        elif waveform.shape[-1] < self.n_samples:
            pad_len = self.n_samples - waveform.shape[-1]
            waveform = F.pad(waveform, (0, pad_len))

        # 2. Compute log10 mel spectrogram via level1 MelSpectrogram
        #    Output: (batch, n_mels, time_frames) where time_frames = n_samples/hop_length + 1
        log_mel = self.mel_spectrogram(waveform)

        # 3. Truncate last STFT frame (Whisper convention: stft[..., :-1])
        log_mel = log_mel[..., :-1]

        # 4. Whisper normalization: clamp to max - 8.0, then scale
        if waveform.dim() == 2:
            # Batched: per-sample max
            max_val = log_mel.amax(dim=(-2, -1), keepdim=True)
        else:
            max_val = log_mel.max()
        log_mel = torch.maximum(log_mel, max_val - 8.0)
        log_mel = (log_mel + 4.0) / 4.0

        return log_mel


# ============================================================================
# Model Variants - configs loaded from HuggingFace
# ============================================================================

VARIANTS: Dict[str, str] = {
    "Base": "openai/whisper-base",
    "Large-v2": "openai/whisper-large-v2",
    "Large-v3": "openai/whisper-large-v3",
}


# ============================================================================
# Attention Modules
# ============================================================================

class WhisperAttention(nn.Module):
    """
    Multi-headed attention matching HuggingFace WhisperAttention.

    Key Whisper-specific detail: Q is pre-scaled by head_dim**-0.5 before
    computing attention scores, and attention is called with scaling=1.0.
    This ordering matters for floating-point numerical alignment.

    k_proj has NO bias; q_proj, v_proj, out_proj have bias.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = Linear(embed_dim, embed_dim, bias=True)

        self.sdpa = ScaledDotProductAttention()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, embed_dim)
            key_value_states: If provided, used for cross-attention K/V
            attention_mask: Optional float mask (batch, 1, tgt_len, src_len)

        Returns:
            (batch, seq_len, embed_dim)
        """
        bsz, tgt_len, _ = hidden_states.shape
        is_cross_attention = key_value_states is not None
        kv_input = key_value_states if is_cross_attention else hidden_states
        kv_len = kv_input.shape[1]

        # Pre-scale Q (Whisper-specific: scale Q, not the attention scores)
        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self.k_proj(kv_input)
        value_states = self.v_proj(kv_input)

        # Reshape to (batch, num_heads, seq, head_dim)
        query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Use level1 SDPA with scale=1.0 (Q already scaled)
        attn_output = self.sdpa(
            query_states, key_states, value_states,
            attn_mask=attention_mask,
            scale=1.0,
        )

        # Reshape back to (batch, seq, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, tgt_len, self.embed_dim)
        return self.out_proj(attn_output)


# ============================================================================
# Encoder / Decoder Layers
# ============================================================================

class WhisperEncoderLayer(nn.Module):
    """Encoder layer: pre-norm self-attention + pre-norm FFN."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.self_attn = WhisperAttention(embed_dim, num_heads)
        self.self_attn_layer_norm = LayerNorm(embed_dim)
        self.fc1 = Linear(embed_dim, ffn_dim, bias=True)
        self.fc2 = Linear(ffn_dim, embed_dim, bias=True)
        self.final_layer_norm = LayerNorm(embed_dim)
        self.activation_fn = GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class WhisperDecoderLayer(nn.Module):
    """Decoder layer: pre-norm self-attention + pre-norm cross-attention + pre-norm FFN."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        # Self-attention
        self.self_attn = WhisperAttention(embed_dim, num_heads)
        self.self_attn_layer_norm = LayerNorm(embed_dim)

        # Cross-attention
        self.encoder_attn = WhisperAttention(embed_dim, num_heads)
        self.encoder_attn_layer_norm = LayerNorm(embed_dim)

        # FFN
        self.fc1 = Linear(embed_dim, ffn_dim, bias=True)
        self.fc2 = Linear(ffn_dim, embed_dim, bias=True)
        self.final_layer_norm = LayerNorm(embed_dim)
        self.activation_fn = GELU()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention (causal)
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        # Cross-attention
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states = self.encoder_attn(hidden_states, key_value_states=encoder_hidden_states)
        hidden_states = residual + hidden_states

        # FFN
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ============================================================================
# Encoder / Decoder
# ============================================================================

class WhisperEncoder(nn.Module):
    """
    Whisper audio encoder.

    - Conv1d front-end (two layers) with GELU activation
    - Sinusoidal positional embeddings (frozen nn.Embedding, initialized from HF)
    - Transformer encoder layers
    - Final LayerNorm
    """

    def __init__(
        self,
        num_mel_bins: int,
        d_model: int,
        num_heads: int,
        encoder_layers: int,
        encoder_ffn_dim: int,
        max_source_positions: int = 1500,
    ):
        super().__init__()
        self.conv1 = Conv1d(num_mel_bins, d_model, kernel_size=3, padding=1, bias=True)
        self.conv2 = Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1, bias=True)

        # Sinusoidal positional embedding (frozen, weights from HF checkpoint)
        self.embed_positions = Embedding(max_source_positions, d_model)
        self.embed_positions.requires_grad_(False)

        self.layers = nn.ModuleList([
            WhisperEncoderLayer(d_model, num_heads, encoder_ffn_dim)
            for _ in range(encoder_layers)
        ])
        self.layer_norm = LayerNorm(d_model)

        self.gelu = GELU()

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_features: (batch, num_mel_bins, time_steps)
                Mel spectrogram features. time_steps must equal
                max_source_positions * conv1.stride * conv2.stride.

        Returns:
            (batch, max_source_positions, d_model)
        """
        # Conv front-end
        hidden_states = self.gelu(self.conv1(input_features))
        hidden_states = self.gelu(self.conv2(hidden_states))
        hidden_states = hidden_states.permute(0, 2, 1)  # (batch, time, d_model)

        # Add sinusoidal positional embeddings
        positions = torch.arange(
            self.embed_positions.vocab_size,
            device=hidden_states.device,
        )
        hidden_states = hidden_states + self.embed_positions(positions)

        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class WhisperDecoder(nn.Module):
    """
    Whisper text decoder.

    - Learned token embedding + learned positional embedding
    - Causal Transformer decoder layers (self-attention + cross-attention)
    - Final LayerNorm
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        decoder_layers: int,
        decoder_ffn_dim: int,
        max_target_positions: int = 448,
    ):
        super().__init__()
        self.embed_tokens = Embedding(vocab_size, d_model)
        self.embed_positions = Embedding(max_target_positions, d_model)

        self.layers = nn.ModuleList([
            WhisperDecoderLayer(d_model, num_heads, decoder_ffn_dim)
            for _ in range(decoder_layers)
        ])
        self.layer_norm = LayerNorm(d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, decoder_seq_len) token IDs
            encoder_hidden_states: (batch, encoder_seq_len, d_model)

        Returns:
            (batch, decoder_seq_len, d_model)
        """
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=input_ids.device)

        hidden_states = self.embed_tokens(input_ids) + self.embed_positions(positions)

        # Build 4D causal mask: (1, 1, seq_len, seq_len)
        causal_mask = torch.triu(
            torch.full(
                (seq_len, seq_len),
                torch.finfo(hidden_states.dtype).min,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            ),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            hidden_states = layer(hidden_states, encoder_hidden_states, causal_mask)

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    Whisper speech recognition model (WhisperForConditionalGeneration).

    Uses level1 operators from KernelBench:
    - ScaledDotProductAttention from level1/attention/_2_Attention
    - GELU from level1/activations/_8_GELU
    - MelSpectrogram from level1/audio/_1_MelSpectrogram (in WhisperFeatureExtractor)
    - Linear from level1/matmul/_10_Linear
    - LayerNorm from level1/normalization/_6_LayerNorm
    - Conv1d from level1/convolutions/_10_Conv1d_Standard
    - Embedding from level1/embeddings/_2_Embedding

    Supports variants: Base, Large-v2, Large-v3

    The model includes a WhisperFeatureExtractor that can convert raw audio
    waveforms directly to log-mel spectrogram features. Use ``forward_from_audio``
    to process raw waveforms end-to-end.
    """

    VARIANTS = VARIANTS

    def __init__(
        self,
        d_model: int = 512,
        encoder_attention_heads: int = 8,
        decoder_attention_heads: int = 8,
        encoder_layers: int = 6,
        decoder_layers: int = 6,
        encoder_ffn_dim: int = 2048,
        decoder_ffn_dim: int = 2048,
        vocab_size: int = 51865,
        num_mel_bins: int = 80,
        max_source_positions: int = 1500,
        max_target_positions: int = 448,
        decoder_start_token_id: int = 50258,
        **kwargs,
    ):
        super().__init__()
        self.decoder_start_token_id = decoder_start_token_id

        # Feature extractor (raw audio -> log-mel spectrogram)
        self.feature_extractor = WhisperFeatureExtractor(n_mels=num_mel_bins)

        self.encoder = WhisperEncoder(
            num_mel_bins=num_mel_bins,
            d_model=d_model,
            num_heads=encoder_attention_heads,
            encoder_layers=encoder_layers,
            encoder_ffn_dim=encoder_ffn_dim,
            max_source_positions=max_source_positions,
        )

        self.decoder = WhisperDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=decoder_attention_heads,
            decoder_layers=decoder_layers,
            decoder_ffn_dim=decoder_ffn_dim,
            max_target_positions=max_target_positions,
        )

        # Output projection (tied to decoder.embed_tokens in HF)
        self.proj_out = Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_features: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with pre-extracted mel spectrogram features.

        Args:
            input_features: (batch, num_mel_bins, time_steps) mel spectrogram
            decoder_input_ids: (batch, decoder_seq_len) token IDs

        Returns:
            logits: (batch, decoder_seq_len, vocab_size)
        """
        encoder_hidden_states = self.encoder(input_features)
        decoder_output = self.decoder(decoder_input_ids, encoder_hidden_states)
        logits = self.proj_out(decoder_output)
        return logits

    def forward_from_audio(
        self,
        waveform: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        End-to-end forward pass from raw audio waveform.

        Uses the built-in WhisperFeatureExtractor (level1/audio/_1_MelSpectrogram)
        to convert raw audio to log-mel spectrogram features before running the
        encoder-decoder model.

        Args:
            waveform: (batch, samples) raw audio at 16 kHz, float32
            decoder_input_ids: (batch, decoder_seq_len) token IDs

        Returns:
            logits: (batch, decoder_seq_len, vocab_size)
        """
        # Feature extraction runs in float32; cast to model dtype for encoder
        input_features = self.feature_extractor(waveform)
        input_features = input_features.to(dtype=self.encoder.conv1.conv1d.weight.dtype)
        return self.forward(input_features, decoder_input_ids)


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
        'd_model': 512,
        'encoder_attention_heads': 8,
        'decoder_attention_heads': 8,
        'encoder_layers': 4,
        'decoder_layers': 4,
        'encoder_ffn_dim': 2048,
        'decoder_ffn_dim': 2048,
        'vocab_size': vocab_size,
        'num_mel_bins': n_mels,
    }]
