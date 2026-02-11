import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    NeRF MLP (3D/Neural Rendering)

    Used by: NeRF, Instant-NGP, Mip-NeRF

    Neural radiance field MLP that maps 3D positions and view directions
    to color and density. Core component of neural volume rendering.

    Shapes:
        positions: (batch, 3) 3D positions
        directions: (batch, 3) view directions
        Output: (batch, 4) RGB + density
    """

    def __init__(self, pos_dim: int = 3, dir_dim: int = 3,
                 hidden_dim: int = 256, num_layers: int = 8,
                 pos_freq: int = 10, dir_freq: int = 4):
        """
        Initialize NeRF MLP.

        Args:
            pos_dim: Position dimension (3)
            dir_dim: Direction dimension (3)
            hidden_dim: Hidden layer dimension
            num_layers: Number of layers in main network
            pos_freq: Positional encoding frequencies for position
            dir_freq: Positional encoding frequencies for direction
        """
        super(Model, self).__init__()
        self.pos_dim = pos_dim
        self.dir_dim = dir_dim
        self.pos_freq = pos_freq
        self.dir_freq = dir_freq

        # Input dimensions after positional encoding
        pos_enc_dim = pos_dim * (2 * pos_freq + 1)
        dir_enc_dim = dir_dim * (2 * dir_freq + 1)

        # Main network (density branch)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(pos_enc_dim, hidden_dim))

        for i in range(1, num_layers):
            if i == num_layers // 2:
                # Skip connection
                self.layers.append(nn.Linear(hidden_dim + pos_enc_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Density output
        self.density_layer = nn.Linear(hidden_dim, 1)

        # Color branch (conditioned on direction)
        self.feature_layer = nn.Linear(hidden_dim, hidden_dim)
        self.color_layers = nn.Sequential(
            nn.Linear(hidden_dim + dir_enc_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid()
        )

    def positional_encoding(self, x: torch.Tensor, num_freq: int) -> torch.Tensor:
        """Apply positional encoding."""
        freqs = 2.0 ** torch.arange(num_freq, device=x.device).float()
        x_freq = x.unsqueeze(-1) * freqs  # (batch, dim, freq)
        encoded = torch.cat([
            x,
            torch.sin(x_freq).flatten(-2),
            torch.cos(x_freq).flatten(-2)
        ], dim=-1)
        return encoded

    def forward(self, positions: torch.Tensor,
                directions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of NeRF MLP.

        Args:
            positions: 3D positions (batch, 3)
            directions: View directions (batch, 3)

        Returns:
            RGBA values (batch, 4) where last channel is density
        """
        # Positional encoding
        pos_enc = self.positional_encoding(positions, self.pos_freq)
        dir_enc = self.positional_encoding(directions, self.dir_freq)

        # Main network
        x = pos_enc
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) // 2:
                x = torch.cat([x, pos_enc], dim=-1)
            x = F.relu(layer(x))

        # Density (can be negative, apply softplus later)
        density = self.density_layer(x)

        # Color (conditioned on direction)
        features = self.feature_layer(x)
        color_input = torch.cat([features, dir_enc], dim=-1)
        color = self.color_layers(color_input)

        # Combine
        output = torch.cat([color, F.softplus(density)], dim=-1)

        return output
