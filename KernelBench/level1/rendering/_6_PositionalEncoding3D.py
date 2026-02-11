import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    3D Positional Encoding (3D/Neural Rendering)

    Used by: NeRF, Neural SDF, 3D Transformers

    Encodes 3D positions using sinusoidal functions at multiple frequencies.
    Maps low-dimensional inputs to high-dimensional space for better learning.

    Shapes:
        positions: (batch, 3) 3D coordinates
        Output: (batch, 3 * (2 * num_freq + 1)) encoded positions
    """

    def __init__(self, num_frequencies: int = 10, include_input: bool = True,
                 log_sampling: bool = True, max_freq: float = None):
        """
        Initialize 3D positional encoding.

        Args:
            num_frequencies: Number of frequency bands
            include_input: Whether to include original coordinates
            log_sampling: Whether to use log-spaced frequencies
            max_freq: Maximum frequency (for log sampling)
        """
        super(Model, self).__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        self.log_sampling = log_sampling

        if log_sampling:
            if max_freq is None:
                max_freq = num_frequencies - 1
            freq_bands = 2.0 ** torch.linspace(0, max_freq, num_frequencies)
        else:
            freq_bands = torch.linspace(1, 2 ** (num_frequencies - 1), num_frequencies)

        self.register_buffer('freq_bands', freq_bands)

        # Compute output dimension
        self.out_dim = 3 * 2 * num_frequencies
        if include_input:
            self.out_dim += 3

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Encode 3D positions.

        Args:
            positions: 3D coordinates (batch, 3)

        Returns:
            Encoded positions (batch, out_dim)
        """
        # Scale positions by frequencies
        # positions: (batch, 3), freq_bands: (num_freq,)
        scaled = positions.unsqueeze(-1) * self.freq_bands * math.pi
        # scaled: (batch, 3, num_freq)

        # Apply sin and cos
        sin_enc = torch.sin(scaled)  # (batch, 3, num_freq)
        cos_enc = torch.cos(scaled)  # (batch, 3, num_freq)

        # Interleave sin and cos
        encoded = torch.stack([sin_enc, cos_enc], dim=-1)  # (batch, 3, num_freq, 2)
        encoded = encoded.view(positions.shape[0], -1)  # (batch, 3 * num_freq * 2)

        if self.include_input:
            encoded = torch.cat([positions, encoded], dim=-1)

        return encoded
