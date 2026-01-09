import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Reconstruction Loss
    
    Used by: VAE, autoencoder
    
    MSE or BCE between input and decoder output.
    
    Shapes:
        x: (batch, ...) original
        x_recon: (batch, ...) reconstructed
        Output: scalar
    """
    
    def __init__(self, loss_type: str = 'mse'):
        super(Model, self).__init__()
        self.loss_type = loss_type
    
    def forward(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        if self.loss_type == 'mse':
            return F.mse_loss(x_recon, x, reduction='mean')
        elif self.loss_type == 'bce':
            return F.binary_cross_entropy(x_recon, x, reduction='mean')
        else:
            return F.l1_loss(x_recon, x, reduction='mean')


batch_size = 64
channels = 3
height = 256
width = 256

def get_inputs():
    x = torch.rand(batch_size, channels, height, width, device='cuda')
    x_recon = torch.rand(batch_size, channels, height, width, device='cuda')
    return [x, x_recon]

def get_init_inputs():
    return ['mse']

