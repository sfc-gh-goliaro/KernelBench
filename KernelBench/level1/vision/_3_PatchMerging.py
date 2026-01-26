import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Patch Merging
    
    Used by: Swin Transformer (between stages)
    
    Spatial downsampling by concatenating 2x2 neighboring patches
    followed by linear projection. Reduces spatial resolution by 2x.
    
    Shapes:
        Input: (batch, height, width, channels)
        Output: (batch, height/2, width/2, 2*channels)
    """
    
    def __init__(self, dim: int):
        """
        Initialize patch merging.
        
        Args:
            dim: Input channel dimension
        """
        super(Model, self).__init__()
        self.dim = dim
        
        # Linear projection from 4*dim to 2*dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge patches to reduce spatial resolution.
        
        Args:
            x: Input tensor (batch, height, width, channels)
            
        Returns:
            Merged tensor (batch, height/2, width/2, 2*channels)
        """
        B, H, W, C = x.shape
        
        # Ensure H and W are even
        assert H % 2 == 0 and W % 2 == 0, f"H ({H}) and W ({W}) must be even"
        
        # Extract 2x2 patches
        x0 = x[:, 0::2, 0::2, :]  # Top-left
        x1 = x[:, 1::2, 0::2, :]  # Bottom-left
        x2 = x[:, 0::2, 1::2, :]  # Top-right
        x3 = x[:, 1::2, 1::2, :]  # Bottom-right
        
        # Concatenate along channel dimension
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4*C)
        
        # Layer norm and linear projection
        x = self.norm(x)
        x = self.reduction(x)  # (B, H/2, W/2, 2*C)
        
        return x


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 8, "height": 56, "width": 56, "channels": 96},
    # Swin-v2-Tiny: stage 1 to 2 (96 channels, 56x56 -> 28x28)
    {"batch_size": 32, "height": 64, "width": 64, "channels": 96},
    # Swin-v2-Base: stage 1 to 2 (128 channels, 96x96 -> 48x48)
    {"batch_size": 16, "height": 96, "width": 96, "channels": 128},
    # Swin-v2-Large: stage 1 to 2 (192 channels, 96x96 -> 48x48)
    {"batch_size": 8, "height": 96, "width": 96, "channels": 192},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("vision", "3_PatchMerging")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["height"], p["width"], p["channels"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["channels"]]
