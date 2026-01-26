import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Self-Attention over Features (AutoInt)
    
    Used by: AutoInt
    
    Self-attention for automatic feature interaction learning.
    
    Shapes:
        Input: (batch, num_features, embed_dim)
        Output: (batch, num_features, embed_dim)
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 2):
        super(Model, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)



PARAMETERS = [
    {"batch_size": 4096, "num_features": 26, "embed_dim": 64},
    # AutoInt: multi-head self-attention for CTR prediction
    {"batch_size": 2048, "num_features": 39, "embed_dim": 128},
    # Transformer4Rec: sequential recommendation attention
    {"batch_size": 512, "num_features": 50, "embed_dim": 256},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("recommendation", "4_SelfAttn_Features")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["num_features"], p["embed_dim"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["embed_dim"]]
