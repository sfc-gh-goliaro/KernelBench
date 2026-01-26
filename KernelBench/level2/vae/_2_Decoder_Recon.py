import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """Fused VAE Decoder + Reconstruction Loss."""
    
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super(Model, self).__init__()
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim), nn.Sigmoid())
    
    def forward(self, z, target):
        recon = self.decoder(z)
        return recon, F.mse_loss(recon, target)

batch_size, latent_dim, hidden_dim, output_dim = 64, 256, 512, 784
def get_inputs(): return [torch.randn(batch_size, latent_dim, device='cuda'), torch.rand(batch_size, output_dim, device='cuda')]
def get_init_inputs(): return [latent_dim, hidden_dim, output_dim]

