import os
import sys
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    VAE Reparameterization Trick
    
    Used by: VAE, CVAE, VQ-VAE
    
    z = mu + sigma * epsilon for backprop through sampling.
    
    Shapes:
        mu: (batch, latent_dim)
        log_var: (batch, latent_dim)
        Output: (batch, latent_dim)
    """
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps
