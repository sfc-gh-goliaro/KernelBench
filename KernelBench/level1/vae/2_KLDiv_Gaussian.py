import torch
import torch.nn as nn

class Model(nn.Module):
    """
    KL Divergence (Gaussian)
    
    Used by: VAE training
    
    KL divergence between learned posterior and standard normal prior.
    
    Shapes:
        mu: (batch, latent_dim)
        log_var: (batch, latent_dim)
        Output: scalar
    """
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        # KL(q||p) = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
        return kl.mean()


batch_size = 64
latent_dim = 256

def get_inputs():
    mu = torch.randn(batch_size, latent_dim, device='cuda')
    log_var = torch.randn(batch_size, latent_dim, device='cuda')
    return [mu, log_var]

def get_init_inputs():
    return []

