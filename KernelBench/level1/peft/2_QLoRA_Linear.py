import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    QLoRA Linear Layer
    
    Used by: QLoRA finetuning
    """
    
    def __init__(self, in_features: int, out_features: int, rank: int = 16,
                 alpha: float = 16.0, group_size: int = 128):
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.group_size = group_size
        
        num_groups = in_features // group_size
        self.register_buffer('base_qweight', 
            torch.zeros(out_features, in_features // 2, dtype=torch.uint8))
        self.register_buffer('base_scales',
            torch.ones(num_groups, out_features))
        self.register_buffer('base_zeros',
            torch.zeros(num_groups, out_features, dtype=torch.int8))
        
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
    
    def _dequantize_base(self) -> torch.Tensor:
        w_low = self.base_qweight & 0x0F
        w_high = self.base_qweight >> 4
        weights = torch.stack([w_low, w_high], dim=-1).view(self.out_features, self.in_features)
        weights = weights.float()
        
        for g in range(self.in_features // self.group_size):
            start = g * self.group_size
            end = start + self.group_size
            weights[:, start:end] = (weights[:, start:end] - self.base_zeros[g].unsqueeze(1).float()) * self.base_scales[g].unsqueeze(1)
        
        return weights.T
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_weight = self._dequantize_base()
        base_output = torch.matmul(x, base_weight)
        lora_output = self.lora_B(self.lora_A(x))
        return base_output + self.scaling * lora_output


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "in_features": 4096, "out_features": 4096, "rank": 16},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("peft", "2_QLoRA_Linear")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["seq_length"], p["in_features"])
    x = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["in_features"], p["out_features"], p["rank"]]
