import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Speech Tokenizer (TTS/Speech Synthesis)

    Used by: VALL-E, SpeechGPT, AudioLM

    Converts continuous audio features into discrete speech tokens using
    vector quantization. Enables language model-based speech generation.

    Shapes:
        audio_features: (batch, time_frames, feature_dim) continuous features
        Output: (batch, time_frames) discrete token indices
    """

    def __init__(self, feature_dim: int = 512, codebook_size: int = 1024,
                 num_codebooks: int = 8):
        """
        Initialize speech tokenizer.

        Args:
            feature_dim: Input feature dimension
            codebook_size: Number of codes in each codebook
            num_codebooks: Number of residual vector quantization layers
        """
        super(Model, self).__init__()
        self.feature_dim = feature_dim
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        # Multiple codebooks for residual VQ
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, feature_dim))
            for _ in range(num_codebooks)
        ])

        # Initialize codebooks
        for codebook in self.codebooks:
            nn.init.uniform_(codebook, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Tokenize audio features.

        Args:
            audio_features: Continuous audio features (batch, time, feature_dim)

        Returns:
            Token indices (batch, time, num_codebooks)
        """
        batch_size, time_frames, _ = audio_features.shape
        device = audio_features.device

        residual = audio_features
        all_indices = []

        for i, codebook in enumerate(self.codebooks):
            # Compute distances to codebook entries
            # residual: (batch, time, dim), codebook: (codebook_size, dim)
            distances = torch.cdist(residual, codebook.unsqueeze(0).expand(batch_size, -1, -1))

            # Find nearest code
            indices = distances.argmin(dim=-1)  # (batch, time)
            all_indices.append(indices)

            # Get quantized values
            quantized = F.embedding(indices, codebook)  # (batch, time, dim)

            # Update residual
            residual = residual - quantized

        # Stack all codebook indices
        token_indices = torch.stack(all_indices, dim=-1)  # (batch, time, num_codebooks)

        return token_indices


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 8, "time_frames": 500, "feature_dim": 512, "codebook_size": 1024, "num_codebooks": 8},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("tts", "1_SpeechTokenizer")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    audio_features = DISTRIBUTIONS[dist_name]((p["batch_size"], p["time_frames"], p["feature_dim"]), dtype=dtype, device=device)
    return [audio_features]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["feature_dim"], p["codebook_size"], p["num_codebooks"]]
