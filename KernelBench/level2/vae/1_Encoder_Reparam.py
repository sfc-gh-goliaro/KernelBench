import torch
import torch.nn as nn

class Model(nn.Module):
    """Fused VAE Encoder Output + Reparameterize."""
    
    def __init__(self, hidden_dim: int, latent_dim: int):
        super(Model, self).__init__()
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, h):
        mu, log_var = self.fc_mu(h), self.fc_var(h)
        return mu + torch.exp(0.5 * log_var) * torch.randn_like(log_var), mu, log_var

batch_size, hidden_dim, latent_dim = 64, 512, 256
def get_inputs(): return [torch.randn(batch_size, hidden_dim, device='cuda')]
def get_init_inputs(): return [hidden_dim, latent_dim]

