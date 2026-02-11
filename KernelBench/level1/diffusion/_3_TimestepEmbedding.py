import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Timestep Embedding
    
    Used by: All diffusion models
    
    Timestep to embedding: sinusoidal encoding + Linear + SiLU + Linear.
    Maps scalar timesteps to dense embeddings for conditioning.
    
    Shapes:
        Input: (batch,) timesteps
        Output: (batch, embed_dim)
    """
    
    def __init__(self, embed_dim: int, freq_dim: int = 256, max_period: float = 10000.0):
        """
        Initialize timestep embedding.
        
        Args:
            embed_dim: Output embedding dimension
            freq_dim: Dimension of sinusoidal encoding
            max_period: Maximum period for sinusoidal encoding
        """
        super(Model, self).__init__()
        self.embed_dim = embed_dim
        self.freq_dim = freq_dim
        self.max_period = max_period
        
        # MLP: sinusoidal -> embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def _sinusoidal_encoding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute sinusoidal timestep embeddings."""
        half_dim = self.freq_dim // 2
        
        # Compute frequencies
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=timesteps.device) / half_dim
        )
        
        # Apply frequencies to timesteps
        args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
        
        # Concatenate sin and cos
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        return embedding
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Embed timesteps.
        
        Args:
            timesteps: Timestep values (batch,) in range [0, 1000]
            
        Returns:
            Timestep embeddings (batch, embed_dim)
        """
        # Sinusoidal encoding
        t_emb = self._sinusoidal_encoding(timesteps)
        
        # MLP projection
        t_emb = self.mlp(t_emb)
        
        return t_emb
