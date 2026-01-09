import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Vector Quantization
    
    Used by: VQ-VAE, VQ-GAN
    
    Find nearest codebook entry with straight-through estimator.
    
    Shapes:
        Input: (batch, seq_len, embed_dim)
        Output: (batch, seq_len, embed_dim) quantized
    """
    
    def __init__(self, num_embeddings: int, embed_dim: int, commitment_cost: float = 0.25):
        super(Model, self).__init__()
        self.codebook = nn.Embedding(num_embeddings, embed_dim)
        self.commitment_cost = commitment_cost
    
    def forward(self, x: torch.Tensor) -> tuple:
        # Flatten
        flat = x.view(-1, x.shape[-1])
        
        # Distances to codebook
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True) +
            self.codebook.weight.pow(2).sum(dim=1) -
            2 * flat @ self.codebook.weight.t()
        )
        
        # Nearest
        indices = distances.argmin(dim=1)
        quantized = self.codebook(indices).view(x.shape)
        
        # Straight-through
        quantized = x + (quantized - x).detach()
        
        # Commitment loss
        loss = F.mse_loss(quantized.detach(), x) * self.commitment_cost
        
        return quantized, loss, indices.view(x.shape[:-1])


batch_size = 8
seq_len = 256
embed_dim = 256
num_embeddings = 8192

def get_inputs():
    x = torch.randn(batch_size, seq_len, embed_dim, device='cuda')
    return [x]

def get_init_inputs():
    return [num_embeddings, embed_dim]

