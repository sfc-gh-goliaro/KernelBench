import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
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



PARAMETERS = [
    {"batch_size": 8, "seq_len": 256, "embed_dim": 256, "num_embeddings": 8192},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("vae", "3_VectorQuantize")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_len"], p["embed_dim"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["num_embeddings"], p["embed_dim"]]
