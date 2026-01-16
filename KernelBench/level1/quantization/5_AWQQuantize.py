import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    AWQ (Activation-aware Weight Quantization)
    
    Used by: vLLM, TensorRT-LLM, many production LLM deployments
    """
    
    def __init__(self, in_features: int = 4096, out_features: int = 4096,
                 group_size: int = 128):
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        
        assert in_features % group_size == 0
        self.num_groups = in_features // group_size
        
        self.qweight = nn.Parameter(
            torch.randint(0, 16, (out_features, in_features // 8), dtype=torch.int32),
            requires_grad=False
        )
        self.scales = nn.Parameter(torch.ones(out_features, self.num_groups))
        self.zeros = nn.Parameter(torch.zeros(out_features, self.num_groups))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_unpacked = self._unpack_weights()
        weight_fp = self._dequantize(weight_unpacked)
        return torch.nn.functional.linear(x, weight_fp)
    
    def _unpack_weights(self) -> torch.Tensor:
        unpacked = torch.zeros(self.out_features, self.in_features, 
                               dtype=torch.float32, device=self.qweight.device)
        for i in range(8):
            unpacked[:, i::8] = ((self.qweight >> (i * 4)) & 0xF).float()
        return unpacked
    
    def _dequantize(self, weight_int: torch.Tensor) -> torch.Tensor:
        weight_fp = torch.zeros_like(weight_int)
        for g in range(self.num_groups):
            start = g * self.group_size
            end = start + self.group_size
            weight_fp[:, start:end] = (
                (weight_int[:, start:end] - self.zeros[:, g:g+1]) * 
                self.scales[:, g:g+1]
            )
        return weight_fp


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "in_features": 4096, "out_features": 4096, "group_size": 128},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("quantization", "5_AWQQuantize")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["seq_length"], p["in_features"])
    x = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["in_features"], p["out_features"], p["group_size"]]
