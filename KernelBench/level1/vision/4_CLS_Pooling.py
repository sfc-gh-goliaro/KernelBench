import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    CLS Token Pooling
    
    Used by: ViT, CLIP, SigLIP
    """
    
    def __init__(self, embed_dim: int, pool_type: str = 'cls'):
        super(Model, self).__init__()
        self.embed_dim = embed_dim
        self.pool_type = pool_type
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool_type == 'cls':
            pooled = x[:, 0]
        elif self.pool_type == 'mean':
            pooled = x[:, 1:].mean(dim=1)
        elif self.pool_type == 'mean_all':
            pooled = x.mean(dim=1)
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")
        
        return self.norm(pooled)


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 32, "num_patches": 196, "embed_dim": 768, "pool_type": "cls"},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("vision", "4_CLS_Pooling")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["num_patches"] + 1, p["embed_dim"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["embed_dim"], p["pool_type"]]
