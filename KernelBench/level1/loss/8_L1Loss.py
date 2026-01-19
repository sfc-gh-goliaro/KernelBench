import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    L1 Loss (Mean Absolute Error)
    
    Used by: Image reconstruction, super-resolution, style transfer
    
    Computes the mean absolute error between predictions and targets.
    L1 loss is more robust to outliers compared to L2 (MSE) loss.
    
    Shapes:
        Input prediction: any shape
        Input target: same shape as prediction
        Output: scalar loss (or per-element if reduction='none')
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize L1 Loss.
        
        Args:
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(Model, self).__init__()
        self.loss_fn = nn.L1Loss(reduction=reduction)
    
    def forward(self, prediction: torch.Tensor, 
                target: torch.Tensor) -> torch.Tensor:
        """
        Compute L1 loss.
        
        Args:
            prediction: Predicted values (any shape)
            target: Target values (same shape as prediction)
            
        Returns:
            L1 loss value
        """
        return self.loss_fn(prediction, target)


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 32, "channels": 3, "height": 256, "width": 256, "reduction": 'mean'},
    # ESRGAN: super-resolution reconstruction (512x512 output)
    {"batch_size": 8, "channels": 3, "height": 512, "width": 512, "reduction": 'mean'},
    # Stable Diffusion VAE: latent space reconstruction (64x64 latent)
    {"batch_size": 4, "channels": 4, "height": 64, "width": 64, "reduction": 'mean'},
    # U-Net: medical image segmentation (224x224 single channel)
    {"batch_size": 16, "channels": 1, "height": 224, "width": 224, "reduction": 'mean'},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("loss", "8_L1Loss")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    prediction = DISTRIBUTIONS[dist_name]((p["batch_size"], p["channels"], p["height"], p["width"]), dtype=dtype, device=device)
    target = DISTRIBUTIONS[dist_name]((p["batch_size"], p["channels"], p["height"], p["width"]), dtype=dtype, device=device)
    return [prediction, target]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["reduction"]]
