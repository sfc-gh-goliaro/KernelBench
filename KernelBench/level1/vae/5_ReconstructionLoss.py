import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Reconstruction Loss
    
    Used by: VAE, autoencoder
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


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 64, "channels": 3, "height": 256, "width": 256, "loss_type": "mse"},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("vae", "5_ReconstructionLoss")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    
    x = DISTRIBUTIONS["uniform_pos"]((p["batch_size"], p["channels"], p["height"], p["width"]), dtype=dtype, device=device)
    x_recon = DISTRIBUTIONS["uniform_pos"]((p["batch_size"], p["channels"], p["height"], p["width"]), dtype=dtype, device=device)
    return [x, x_recon]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["loss_type"]]
