import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    FP8 GEMM
    
    Used by: FP8 training and inference
    """
    
    def __init__(self, in_features: int, out_features: int, use_bias: bool = False):
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.register_buffer('weight_scale', torch.tensor(1.0))
        self.register_buffer('input_scale', torch.tensor(1.0))
        
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
    
    def _quantize_fp8(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        fp8_max = 448.0
        x_scaled = x / scale
        x_clamped = torch.clamp(x_scaled, -fp8_max, fp8_max)
        x_quantized = x_clamped.to(torch.float16)
        return x_quantized * scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            input_amax = x.abs().max()
            self.input_scale = input_amax / 448.0
        
        x_fp8 = self._quantize_fp8(x, self.input_scale)
        w_fp8 = self._quantize_fp8(self.weight, self.weight_scale)
        
        output = torch.nn.functional.linear(x_fp8, w_fp8, self.bias)
        return output


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "in_features": 4096, "out_features": 4096},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("quantization", "2_FP8_GEMM")

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
