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
    
    Trim low-magnitude, Elect sign, Disjoint merge for sparse merging.
    
    Shapes:
        deltas: list of (param_shape) task vectors
        Output: (param_shape) merged
    """
    
    def __init__(self, trim_ratio: float = 0.2):
        super(Model, self).__init__()
        self.trim_ratio = trim_ratio
    
    def forward(self, *deltas: torch.Tensor) -> torch.Tensor:
        # Stack deltas
        stacked = torch.stack(deltas)  # (num_models, ...)
        
        # Trim: zero out low-magnitude values
        for i in range(len(deltas)):
            threshold = stacked[i].abs().quantile(self.trim_ratio)
            stacked[i] = torch.where(stacked[i].abs() >= threshold, stacked[i], torch.zeros_like(stacked[i]))
        
        # Elect sign: majority vote
        signs = stacked.sign()
        elected_sign = signs.sum(dim=0).sign()
        
        # Disjoint merge: average values with matching sign
        mask = (signs == elected_sign.unsqueeze(0)) | (stacked == 0)
        masked = stacked * mask
        
        # Average non-zero values
        counts = (masked != 0).sum(dim=0).clamp(min=1)
        merged = masked.sum(dim=0) / counts
        
        return merged



PARAMETERS = [
    # Llama-3.1 8B: attention output projection TIES merge
    {"param_shape": (4096, 4096)},
    # Mistral 7B: MLP up projection sparse merging
    {"param_shape": (14336, 4096)},
    # Gemma 2B: compact layer TIES merging
    {"param_shape": (2048, 2048)},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("model_merging", "3_TIES")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    d1 = DISTRIBUTIONS[dist_name]((*p["param_shape"]), dtype=dtype, device=device)
    d2 = DISTRIBUTIONS[dist_name]((*p["param_shape"]), dtype=dtype, device=device)
    d3 = DISTRIBUTIONS[dist_name]((*p["param_shape"]), dtype=dtype, device=device)
    return [d1, d2, d3]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return [0.2]
