import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Model(nn.Module):
    """
    Sinusoidal Timestep Encoding (parameter-free)

    Used by: Stable Diffusion 1.5, SDXL, all UNet-based diffusion models

    Maps scalar diffusion timesteps to dense sinusoidal embeddings.
    Matches the diffusers `get_timestep_embedding` function exactly,
    including `flip_sin_to_cos` and `downscale_freq_shift` options.

    This is distinct from embeddings/_3_SinusoidalPosEmbed, which
    precomputes a lookup table for integer sequence positions. This op
    takes arbitrary float timestep values at runtime.

    Shapes:
        Input: (batch,) timestep scalars
        Output: (batch, embedding_dim) sinusoidal features
    """

    def __init__(self, num_channels: int, flip_sin_to_cos: bool = True,
                 downscale_freq_shift: float = 0, max_period: int = 10000):
        """
        Initialize sinusoidal timestep encoding.

        Args:
            num_channels: Output embedding dimension
            flip_sin_to_cos: If True, output is [cos, sin] instead of [sin, cos]
            downscale_freq_shift: Shift applied to frequency denominator
            max_period: Maximum period for sinusoidal encoding
        """
        super(Model, self).__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Compute sinusoidal timestep embeddings.

        Args:
            timesteps: Timestep values (batch,), can be float or int

        Returns:
            Sinusoidal embeddings (batch, num_channels)
        """
        half_dim = self.num_channels // 2

        exponent = -math.log(self.max_period) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim - self.downscale_freq_shift)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if self.flip_sin_to_cos:
            emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

        if self.num_channels % 2 == 1:
            emb = F.pad(emb, (0, 1, 0, 0))

        return emb
