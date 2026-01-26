import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    VQ-VAE Vector Quantizer
    
    The core quantization layer in VQ-VAE.
    Used by: VQ-VAE, VQ-VAE-2, VQ-GAN, DALL-E tokenizer
    
    Architecture:
        z_e (encoder output) -> Find nearest codebook entry
                             -> z_q (quantized)
                             -> Straight-through estimator for gradients
    
    Key features:
    - Discrete latent space via codebook lookup
    - Straight-through gradient estimator
    - Commitment loss for encoder
    - EMA codebook updates (optional, not shown here)
    """
    def __init__(self, embedding_dim: int, num_embeddings: int, commitment_cost: float = 0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # Codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor) -> tuple:
        """
        Vector quantization with straight-through estimator.
        
        Args:
            z: Encoder output (batch, channels, height, width) or (batch, seq_len, embedding_dim)
        
        Returns:
            Tuple of (quantized, loss, encoding_indices)
        """
        # Handle both conv (BCHW) and sequence (BLD) formats
        if z.dim() == 4:
            # BCHW -> BHWC -> B*H*W, C
            z = z.permute(0, 2, 3, 1).contiguous()
            original_shape = z.shape
            z_flat = z.view(-1, self.embedding_dim)
            is_conv = True
        else:
            # BLD -> B*L, D
            original_shape = z.shape
            z_flat = z.view(-1, self.embedding_dim)
            is_conv = False
        
        # Compute distances to codebook entries
        # d(z, e)^2 = ||z||^2 + ||e||^2 - 2 * z @ e^T
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True) +
            self.embedding.weight.pow(2).sum(dim=1) -
            2 * z_flat @ self.embedding.weight.t()
        )
        
        # Find nearest codebook entry
        encoding_indices = distances.argmin(dim=1)
        
        # Quantize
        z_q = self.embedding(encoding_indices)
        
        # Compute loss
        # Codebook loss: move codebook towards encoder outputs
        codebook_loss = F.mse_loss(z_q.detach(), z_flat)
        # Commitment loss: encoder commits to codebook
        commitment_loss = F.mse_loss(z_q, z_flat.detach())
        loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # Straight-through estimator: copy gradients from z_q to z
        z_q = z_flat + (z_q - z_flat).detach()
        
        # Reshape back
        if is_conv:
            z_q = z_q.view(original_shape)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
            encoding_indices = encoding_indices.view(original_shape[:-1])
        else:
            z_q = z_q.view(original_shape)
            encoding_indices = encoding_indices.view(original_shape[:-1])
        
        return z_q, loss, encoding_indices


# Benchmark configuration
batch_size = 8
embedding_dim = 256
num_embeddings = 512  # Codebook size
height = 16
width = 16

def get_inputs():
    # Simulated encoder output
    z = torch.randn(batch_size, embedding_dim, height, width)
    return [z]

def get_init_inputs():
    return [embedding_dim, num_embeddings]

