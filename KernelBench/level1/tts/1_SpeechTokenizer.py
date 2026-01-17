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

batch_size = 8
time_frames = 500  # ~5 seconds at 100 fps
feature_dim = 512
codebook_size = 1024
num_codebooks = 8

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    audio_features = torch.randn(batch_size, time_frames, feature_dim, device='cuda')
    return [audio_features]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [feature_dim, codebook_size, num_codebooks]
