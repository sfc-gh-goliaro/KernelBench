import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Flow Matching (TTS/Speech Synthesis)

    Used by: VoiceBox, E2-TTS, Matcha-TTS

    Flow matching for continuous speech generation. Learns to transform
    noise into mel spectrograms via optimal transport paths.

    Shapes:
        x_0: (batch, time, mel_dim) source (noise)
        x_1: (batch, time, mel_dim) target (mel spectrogram)
        t: (batch,) timestep in [0, 1]
        Output: (batch, time, mel_dim) interpolated sample and velocity
    """

    def __init__(self, mel_dim: int = 80, hidden_dim: int = 512):
        """
        Initialize flow matching.

        Args:
            mel_dim: Mel spectrogram dimension
            hidden_dim: Hidden dimension for velocity network
        """
        super(Model, self).__init__()
        self.mel_dim = mel_dim
        self.hidden_dim = hidden_dim

        # Velocity prediction network
        self.velocity_net = nn.Sequential(
            nn.Linear(mel_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, mel_dim)
        )

    def interpolate(self, x_0: torch.Tensor, x_1: torch.Tensor,
                   t: torch.Tensor) -> torch.Tensor:
        """
        Linear interpolation between x_0 and x_1.

        Args:
            x_0: Source samples (noise)
            x_1: Target samples (data)
            t: Timesteps (batch,)

        Returns:
            Interpolated samples x_t
        """
        t = t.view(-1, 1, 1)  # (batch, 1, 1)
        return (1 - t) * x_0 + t * x_1

    def target_velocity(self, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        """
        Compute target velocity (x_1 - x_0) for optimal transport.

        Args:
            x_0: Source samples
            x_1: Target samples

        Returns:
            Target velocity
        """
        return x_1 - x_0

    def forward(self, x_0: torch.Tensor, x_1: torch.Tensor,
                t: torch.Tensor) -> tuple:
        """
        Flow matching forward pass.

        Args:
            x_0: Source noise (batch, time, mel_dim)
            x_1: Target mel spectrogram (batch, time, mel_dim)
            t: Timesteps in [0, 1] (batch,)

        Returns:
            Tuple of:
                - x_t: Interpolated sample (batch, time, mel_dim)
                - predicted_velocity: Predicted velocity (batch, time, mel_dim)
                - target_velocity: Target velocity for loss (batch, time, mel_dim)
        """
        # Get interpolated sample
        x_t = self.interpolate(x_0, x_1, t)

        # Expand timestep for conditioning
        batch_size, time_frames, _ = x_t.shape
        t_expanded = t.view(-1, 1, 1).expand(-1, time_frames, 1)

        # Predict velocity
        velocity_input = torch.cat([x_t, t_expanded], dim=-1)
        predicted_velocity = self.velocity_net(velocity_input)

        # Get target velocity
        target_vel = self.target_velocity(x_0, x_1)

        return x_t, predicted_velocity, target_vel
