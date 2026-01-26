import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Gumbel-Softmax
    
    Used by: Discrete VAE, VQ models
    
    Differentiable discrete sampling using Gumbel-Softmax.
    
    Shapes:
        Input: (batch, num_classes) logits
        Output: (batch, num_classes) soft/hard one-hot
    """
    
    def __init__(self, temperature: float = 1.0, hard: bool = False):
        super(Model, self).__init__()
        self.temperature = temperature
        self.hard = hard
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return F.gumbel_softmax(logits, tau=self.temperature, hard=self.hard)



PARAMETERS = [
    {"batch_size": 64, "num_classes": 512},
    # DALL-E discrete VAE: 8192-token codebook
    {"batch_size": 1, "num_classes": 8192},
    # SD3 discrete tokenizer: large vocabulary sampling
    {"batch_size": 2, "num_classes": 16384},
    # Parti VQ-VAE: discrete latent sampling
    {"batch_size": 4, "num_classes": 8192},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("vae", "4_GumbelSoftmax")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    logits = DISTRIBUTIONS[dist_name]((p["batch_size"], p["num_classes"]), dtype=dtype, device=device)
    return [logits]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return [1.0, True]
