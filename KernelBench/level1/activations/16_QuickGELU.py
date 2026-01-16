import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Quick GELU
    
    Used by: CLIP, OpenCLIP
    
    Approximate GELU using sigmoid: x * sigmoid(1.702 * x).
    Faster than exact GELU while maintaining similar properties.
    
    Shapes:
        Input: any shape
        Output: same shape as input
    """
    
    def __init__(self):
        """Initialize Quick GELU."""
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Quick GELU activation.
        
        Args:
            x: Input tensor of any shape
            
        Returns:
            Activated tensor of same shape
        """
        return x * torch.sigmoid(1.702 * x)


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 32, "seq_length": 197, "hidden_size": 768},  # ViT: 196 patches + 1 CLS
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("activations", "16_QuickGELU")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["seq_length"], p["hidden_size"])
    x = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
