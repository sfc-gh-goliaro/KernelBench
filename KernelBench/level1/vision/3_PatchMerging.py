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
    """
    
    def __init__(self, dim: int):
        super(Model, self).__init__()
        self.dim = dim
        
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        
        assert H % 2 == 0 and W % 2 == 0, f"H ({H}) and W ({W}) must be even"
        
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        
        x = self.norm(x)
        x = self.reduction(x)
        
        return x


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "height": 56, "width": 56, "channels": 96},
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
