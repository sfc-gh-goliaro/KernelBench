import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Fused Timestep Embedding + MLP
    
    Used by: All diffusion models
    
    Sinusoidal embedding + Linear + SiLU + Linear for timestep.
    """
    
    def __init__(self, embed_dim: int, freq_dim: int = 256):
        super(Model, self).__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / half)
        args = timesteps.float().unsqueeze(-1) * freqs
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)


batch_size, embed_dim = 64, 1280

def get_inputs():
    return [torch.randint(0, 1000, (batch_size,), device='cuda')]

def get_init_inputs():
    return [embed_dim]

