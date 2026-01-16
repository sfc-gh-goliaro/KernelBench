import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    TIES Merging
    
    Used by: TIES merging
    """
    
    def __init__(self, trim_ratio: float = 0.2):
        super(Model, self).__init__()
        self.trim_ratio = trim_ratio
    
    def forward(self, *deltas: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack(deltas)
        
        for i in range(len(deltas)):
            threshold = stacked[i].abs().quantile(self.trim_ratio)
            stacked[i] = torch.where(stacked[i].abs() >= threshold, stacked[i], torch.zeros_like(stacked[i]))
        
        signs = stacked.sign()
        elected_sign = signs.sum(dim=0).sign()
        
        mask = (signs == elected_sign.unsqueeze(0)) | (stacked == 0)
        masked = stacked * mask
        
        counts = (masked != 0).sum(dim=0).clamp(min=1)
        merged = masked.sum(dim=0) / counts
        
        return merged


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"param_shape": (4096, 4096), "num_models": 3, "trim_ratio": 0.2},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("model_merging", "3_TIES")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    deltas = [DISTRIBUTIONS[dist_name](p["param_shape"], dtype=dtype, device=device) * 0.1 for _ in range(p["num_models"])]
    return deltas

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["trim_ratio"]]
