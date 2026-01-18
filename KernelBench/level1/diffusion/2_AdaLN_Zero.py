import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    AdaLN-Zero
    
    Used by: DiT (original)
    
    Zero-initialized AdaLN with additional learnable gate parameters
    (alpha) for residual scaling. Initialization ensures identity
    at start of training.
    
    Shapes:
        x: (batch, seq_len, hidden_size)
        conditioning: (batch, cond_dim)
        Output: (batch, seq_len, hidden_size)
    """
    
    def __init__(self, hidden_size: int, cond_dim: int):
        """
        Initialize AdaLN-Zero.
        
        Args:
            hidden_size: Hidden dimension
            cond_dim: Conditioning embedding dimension
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        
        # Project conditioning to scale, shift, and gate (alpha)
        # Initialize to zero so output starts as identity
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * hidden_size)
        )
        
        # Zero-initialize the output layer
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> tuple:
        """
        Apply AdaLN-Zero.
        
        Args:
            x: Input tensor (batch, seq_len, hidden_size)
            conditioning: Conditioning embedding (batch, cond_dim)
            
        Returns:
            Tuple of (normalized_x, gate) where gate is used for residual
        """
        # Get shift, scale, gate from conditioning
        modulation = self.adaLN_modulation(conditioning)
        shift, scale, gate = modulation.chunk(3, dim=-1)
        
        # Apply layer norm
        x = self.norm(x)
        
        # Apply adaptive scale and shift
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        # Return both normalized output and gate for residual
        return x, gate.unsqueeze(1)


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 8, "seq_length": 1024, "hidden_size": 1152, "cond_dim": 1152},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("diffusion", "2_AdaLN_Zero")

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
