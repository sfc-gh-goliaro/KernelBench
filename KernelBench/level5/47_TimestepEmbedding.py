import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Timestep Embedding for Diffusion Models.
    
    Converts scalar timesteps to high-dimensional embeddings using
    sinusoidal encoding followed by MLP projection.
    
    Based on: "Denoising Diffusion Probabilistic Models" and DiT
    """
    def __init__(self, dim, freq_dim=256, max_period=10000):
        """
        :param dim: Output embedding dimension
        :param freq_dim: Dimension of sinusoidal encoding
        :param max_period: Maximum period for sinusoidal encoding
        """
        super(Model, self).__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.max_period = max_period
        
        # MLP to project sinusoidal encoding
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        
        # Precompute frequencies
        half_dim = freq_dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer('freqs', freqs)
    
    def timestep_embedding(self, t):
        """
        Create sinusoidal timestep embeddings.
        
        :param t: Timestep tensor (batch,) or scalar
        :return: Sinusoidal embedding (batch, freq_dim)
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        # Expand timesteps with frequencies
        args = t.unsqueeze(-1) * self.freqs.unsqueeze(0)
        
        # Concatenate sin and cos
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        return embedding
    
    def forward(self, t):
        """
        Generate timestep embeddings.
        
        :param t: Timestep tensor (batch,) - values in [0, 1] or [0, 1000]
        :return: Timestep embedding (batch, dim)
        """
        # Get sinusoidal embedding
        t_emb = self.timestep_embedding(t)
        
        # Project through MLP
        return self.mlp(t_emb)


# Test parameters
batch_size = 64
dim = 1152
freq_dim = 256

def get_inputs():
    # Random timesteps in [0, 1000]
    t = torch.randint(0, 1000, (batch_size,)).float()
    return [t]

def get_init_inputs():
    return [dim, freq_dim]

