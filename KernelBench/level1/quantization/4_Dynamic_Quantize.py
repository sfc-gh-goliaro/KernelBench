import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Dynamic Quantization
    
    Used by: INT8 inference optimization
    """
    
    def __init__(self, num_bits: int = 8, symmetric: bool = True):
        super(Model, self).__init__()
        self.num_bits = num_bits
        self.symmetric = symmetric
        
        if symmetric:
            self.qmax = 2 ** (num_bits - 1) - 1
            self.qmin = -(2 ** (num_bits - 1))
        else:
            self.qmax = 2 ** num_bits - 1
            self.qmin = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.symmetric:
            x_max = x.abs().max()
            scale = x_max / self.qmax
            scale = torch.clamp(scale, min=1e-10)
            x_quant = torch.round(x / scale).clamp(self.qmin, self.qmax)
            x_deq = x_quant * scale
        else:
            x_min, x_max = x.min(), x.max()
            scale = (x_max - x_min) / (self.qmax - self.qmin)
            scale = torch.clamp(scale, min=1e-10)
            zero_point = torch.round(-x_min / scale)
            x_quant = torch.round(x / scale + zero_point).clamp(self.qmin, self.qmax)
            x_deq = (x_quant - zero_point) * scale
        
        return x_deq


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "hidden_size": 4096, "num_bits": 8, "symmetric": True},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("quantization", "4_Dynamic_Quantize")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["seq_length"], p["hidden_size"])
    x = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["num_bits"], p["symmetric"]]
