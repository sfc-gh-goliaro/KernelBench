import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Multi-LoRA SGMV (Segmented Grouped Matrix-Vector multiplication)
    
    Used by: Multi-tenant LoRA serving
    """
    
    def __init__(self, in_features: int, out_features: int, num_adapters: int = 8,
                 rank: int = 16, alpha: float = 16.0):
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_adapters = num_adapters
        self.rank = rank
        self.scaling = alpha / rank
        
        self.base_linear = nn.Linear(in_features, out_features, bias=False)
        self.lora_A = nn.Parameter(torch.randn(num_adapters, rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(num_adapters, out_features, rank))
    
    def forward(self, x: torch.Tensor, adapter_indices: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        base_output = self.base_linear(x)
        lora_output = torch.zeros_like(base_output)
        
        for i in range(self.num_adapters):
            mask = adapter_indices == i
            if not mask.any():
                continue
            
            x_adapter = x[mask]
            intermediate = torch.matmul(x_adapter, self.lora_A[i].T)
            adapter_out = torch.matmul(intermediate, self.lora_B[i].T)
            lora_output[mask] = adapter_out
        
        return base_output + self.scaling * lora_output


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 32, "seq_length": 512, "in_features": 4096, "out_features": 4096, "num_adapters": 8, "rank": 16},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("peft", "3_Multi_LoRA_SGMV")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["seq_length"], p["in_features"])
    x = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    adapter_indices = torch.randint(0, p["num_adapters"], (p["batch_size"],), device=device)
    return [x, adapter_indices]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["in_features"], p["out_features"], p["num_adapters"], p["rank"]]
