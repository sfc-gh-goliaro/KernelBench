import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Zero-Shot Voice Clone (TTS/Speech Synthesis)

    Used by: VALL-E, YourTTS, OpenVoice

    Extracts speaker embedding from reference audio for zero-shot voice
    cloning. Enables synthesis in any voice from a short reference clip.

    Shapes:
        reference_audio: (batch, n_mels, ref_frames) mel spectrogram of reference
        Output: (batch, speaker_dim) speaker embedding
    """

    def __init__(self, n_mels: int = 80, speaker_dim: int = 256,
                 hidden_dim: int = 512):
        """
        Initialize voice cloning encoder.

        Args:
            n_mels: Number of mel bands in input
            speaker_dim: Output speaker embedding dimension
            hidden_dim: Hidden dimension
        """
        super(Model, self).__init__()
        self.n_mels = n_mels
        self.speaker_dim = speaker_dim

        # Convolutional encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(n_mels, hidden_dim, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )

        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Speaker projection
        self.speaker_proj = nn.Sequential(
            nn.Linear(hidden_dim, speaker_dim),
            nn.Tanh()
        )

    def forward(self, reference_audio: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker embedding from reference audio.

        Args:
            reference_audio: Reference mel spectrogram (batch, n_mels, ref_frames)

        Returns:
            Speaker embedding (batch, speaker_dim)
        """
        # Encode
        x = self.encoder(reference_audio)  # (batch, hidden, frames)

        # Transpose for attention
        x = x.transpose(1, 2)  # (batch, frames, hidden)

        # Attention pooling
        attn_weights = self.attention(x)  # (batch, frames, 1)
        attn_weights = F.softmax(attn_weights, dim=1)

        # Weighted sum
        pooled = (x * attn_weights).sum(dim=1)  # (batch, hidden)

        # Project to speaker embedding
        speaker_embed = self.speaker_proj(pooled)

        return speaker_embed
