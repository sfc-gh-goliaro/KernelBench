import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    KV Cache Quantization
    
    Used by: Memory-constrained serving
    """
    
    def __init__(self, num_heads: int, head_dim: int, num_bits: int = 8):
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_bits = num_bits
        
        self.qmax = 2 ** (num_bits - 1) - 1
        self.qmin = -(2 ** (num_bits - 1))
    
    def quantize(self, x: torch.Tensor) -> tuple:
        x_flat = x.view(-1, self.head_dim)
        x_max = x_flat.abs().max(dim=-1, keepdim=True).values
        scale = x_max / self.qmax
        scale = torch.clamp(scale, min=1e-10)
        
        x_quant = torch.round(x_flat / scale).clamp(self.qmin, self.qmax).to(torch.int8)
        x_quant = x_quant.view(x.shape)
        scale = scale.view(x.shape[:-1] + (1,))
        
        return x_quant, scale
    
    def dequantize(self, x_quant: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x_quant.float() * scale
    
    def forward(self, k: torch.Tensor, v: torch.Tensor) -> tuple:
        k_quant, k_scale = self.quantize(k)
        v_quant, v_scale = self.quantize(v)
        return (k_quant, k_scale), (v_quant, v_scale)


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "num_heads": 32, "seq_length": 2048, "head_dim": 128},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("quantization", "3_KVCache_Quantize")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["num_heads"], p["seq_length"], p["head_dim"])
    k = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    v = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [k, v]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["num_heads"], p["head_dim"]]
