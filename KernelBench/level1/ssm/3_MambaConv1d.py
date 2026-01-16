import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Mamba Causal Conv1d
    
    Used by: Mamba, Mamba-2
    """
    
    def __init__(self, d_inner: int, kernel_size: int = 4):
        super(Model, self).__init__()
        self.d_inner = d_inner
        self.kernel_size = kernel_size
        
        self.conv = nn.Conv1d(
            d_inner, d_inner, kernel_size,
            padding=kernel_size - 1,
            groups=d_inner,
            bias=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x[:, :, :seq_len]
        x = F.silu(x)
        x = x.transpose(1, 2)
        
        return x


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "d_inner": 4096, "kernel_size": 4},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("ssm", "3_MambaConv1d")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_length"], p["d_inner"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["d_inner"], p["kernel_size"]]
