import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    VAE Reparameterization Trick
    
    Used by: VAE, CVAE, VQ-VAE
    """
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 64, "latent_dim": 256},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("vae", "1_Reparameterize")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    
    mu = DISTRIBUTIONS[dist_name]((p["batch_size"], p["latent_dim"]), dtype=dtype, device=device)
    log_var = DISTRIBUTIONS[dist_name]((p["batch_size"], p["latent_dim"]), dtype=dtype, device=device)
    return [mu, log_var]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
