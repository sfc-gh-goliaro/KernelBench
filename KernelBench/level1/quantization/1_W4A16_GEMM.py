import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    W4A16 GEMM (4-bit Weight, 16-bit Activation)
    
    Used by: GPTQ, AWQ quantized models
    """
    
    def __init__(self, in_features: int, out_features: int, group_size: int = 128):
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        
        self.num_groups = in_features // group_size
        self.register_buffer('qweight', torch.zeros(out_features, in_features // 2, dtype=torch.uint8))
        self.register_buffer('scales', torch.ones(self.num_groups, out_features))
        self.register_buffer('zeros', torch.zeros(self.num_groups, out_features, dtype=torch.int8))
    
    def _dequantize(self) -> torch.Tensor:
        w_low = self.qweight & 0x0F
        w_high = self.qweight >> 4
        weights = torch.stack([w_low, w_high], dim=-1).view(self.out_features, self.in_features)
        weights = weights.to(torch.float16)
        
        for g in range(self.num_groups):
            start = g * self.group_size
            end = start + self.group_size
            weights[:, start:end] = (weights[:, start:end] - self.zeros[g].unsqueeze(1).float()) * self.scales[g].unsqueeze(1)
        
        return weights.T
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._dequantize()
        return torch.matmul(x, weight)


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "in_features": 4096, "out_features": 11008},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("quantization", "1_W4A16_GEMM")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float16, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["seq_length"], p["in_features"])
    x = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["in_features"], p["out_features"]]
