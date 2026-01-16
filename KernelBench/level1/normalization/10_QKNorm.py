import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    QK Normalization
    
    Used by: Llama 4, some Gemma variants
    
    Applies RMS normalization to query and key tensors before attention.
    """
    
    def __init__(self, head_dim: int = 128, eps: float = 1e-6):
        super(Model, self).__init__()
        self.head_dim = head_dim
        self.eps = eps
        self.q_scale = nn.Parameter(torch.ones(head_dim))
        self.k_scale = nn.Parameter(torch.ones(head_dim))
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple:
        q_variance = q.pow(2).mean(-1, keepdim=True)
        q_normed = q * torch.rsqrt(q_variance + self.eps) * self.q_scale
        
        k_variance = k.pow(2).mean(-1, keepdim=True)
        k_normed = k * torch.rsqrt(k_variance + self.eps) * self.k_scale
        
        return q_normed, k_normed


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("normalization", "10_QKNorm")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    q_shape = (p["batch_size"], p["seq_length"], p["num_heads"], p["head_dim"])
    k_shape = (p["batch_size"], p["seq_length"], p["num_kv_heads"], p["head_dim"])
    q = DISTRIBUTIONS[dist_name](q_shape, dtype=dtype, device=device)
    k = DISTRIBUTIONS[dist_name](k_shape, dtype=dtype, device=device)
    return [q, k]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["head_dim"]]
