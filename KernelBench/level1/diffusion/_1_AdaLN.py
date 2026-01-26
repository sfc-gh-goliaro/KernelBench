import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Adaptive Layer Normalization (AdaLN)
    
    Used by: DiT, SD-3, FLUX, PixArt
    
    Adaptive LayerNorm where scale and shift are predicted from
    timestep/conditioning embedding. Core building block for diffusion
    transformers.
    
    Shapes:
        x: (batch, seq_len, hidden_size)
        conditioning: (batch, cond_dim)
        Output: (batch, seq_len, hidden_size)
    """
    
    def __init__(self, hidden_size: int, cond_dim: int):
        """
        Initialize AdaLN.
        
        Args:
            hidden_size: Hidden dimension
            cond_dim: Conditioning embedding dimension
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        
        # Project conditioning to scale and shift
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_size)
        )
    
    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive layer normalization.
        
        Args:
            x: Input tensor (batch, seq_len, hidden_size)
            conditioning: Conditioning embedding (batch, cond_dim)
            
        Returns:
            Normalized and modulated tensor
        """
        # Get scale and shift from conditioning
        shift, scale = self.adaLN_modulation(conditioning).chunk(2, dim=-1)
        
        # Apply layer norm
        x = self.norm(x)
        
        # Apply adaptive scale and shift
        # shift, scale: (batch, hidden_size) -> (batch, 1, hidden_size)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        return x


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 8, "seq_length": 1024, "hidden_size": 1152, "cond_dim": 1152},
    # SDXL-Turbo: hidden_size=1280, image 512x512 (patches = 1024)
    {"batch_size": 8, "seq_length": 1024, "hidden_size": 1280, "cond_dim": 1280},
    # SDXL-Lightning: hidden_size=1280, image 1024x1024 (patches = 4096)
    {"batch_size": 4, "seq_length": 4096, "hidden_size": 1280, "cond_dim": 1280},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("diffusion", "1_AdaLN")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_length"], p["hidden_size"]), dtype=dtype, device=device)
    conditioning = DISTRIBUTIONS[dist_name]((p["batch_size"], p["cond_dim"]), dtype=dtype, device=device)
    return [x, conditioning]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["hidden_size"], p["cond_dim"]]
