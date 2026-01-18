import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Audio Feature Extractor

    Used by: Whisper, Qwen2-Audio, SeamlessM4T

    Extracts audio features from mel spectrogram using convolutional layers.
    Downsamples temporal dimension and projects to model hidden size.

    Shapes:
        Input: (batch, n_mels, time_frames) mel spectrogram
        Output: (batch, time_frames // 2, hidden_size) audio features
    """

    def __init__(self, n_mels: int = 80, hidden_size: int = 1024):
        """
        Initialize audio feature extractor.

        Args:
            n_mels: Number of mel bands in input spectrogram
            hidden_size: Output hidden dimension
        """
        super(Model, self).__init__()
        self.n_mels = n_mels
        self.hidden_size = hidden_size

        # Convolutional feature extraction (similar to Whisper encoder)
        self.conv1 = nn.Conv1d(n_mels, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1)

        self.gelu = nn.GELU()

        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Positional embedding (learnable)
        self.max_positions = 1500
        self.positional_embedding = nn.Embedding(self.max_positions, hidden_size)

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Extract audio features from mel spectrogram.

        Args:
            mel_spectrogram: Mel spectrogram (batch, n_mels, time_frames)

        Returns:
            Audio features (batch, time_frames // 2, hidden_size)
        """
        # Apply convolutions
        x = self.gelu(self.conv1(mel_spectrogram))
        x = self.gelu(self.conv2(x))

        # Transpose to (batch, time, hidden)
        x = x.transpose(1, 2)

        # Add positional embeddings
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        x = x + self.positional_embedding(positions)

        # Layer norm
        x = self.layer_norm(x)

        return x


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 8, "n_mels": 80, "time_frames": 3000, "hidden_size": 1024},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("audio", "5_AudioFeatureExtractor")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    mel_spectrogram = DISTRIBUTIONS[dist_name]((p["batch_size"], p["n_mels"], p["time_frames"]), dtype=dtype, device=device)
    return [mel_spectrogram]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["n_mels"], p["hidden_size"]]
