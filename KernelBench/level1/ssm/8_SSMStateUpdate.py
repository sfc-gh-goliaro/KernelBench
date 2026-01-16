import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    SSM State Update
    
    Used by: Mamba, Mamba-2, RetNet, RWKV
    """
    
    def __init__(self, intermediate_size: int = 5120, state_size: int = 16):
        super(Model, self).__init__()
        self.intermediate_size = intermediate_size
        self.state_size = state_size
        
        A = torch.arange(1, state_size + 1, dtype=torch.float32).repeat(intermediate_size, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(intermediate_size))
    
    def forward(self, x: torch.Tensor, h: torch.Tensor, 
                dt: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> tuple:
        A = -torch.exp(self.A_log)
        
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
        dB = dt.unsqueeze(-1) * B.unsqueeze(1)
        
        h_new = dA * h + dB * x.unsqueeze(-1)
        y = (h_new * C.unsqueeze(1)).sum(-1) + self.D * x
        
        return h_new, y


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 32, "intermediate_size": 5120, "state_size": 16},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("ssm", "8_SSMStateUpdate")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["intermediate_size"]), dtype=dtype, device=device)
    h = DISTRIBUTIONS[dist_name]((p["batch_size"], p["intermediate_size"], p["state_size"]), dtype=dtype, device=device)
    dt = DISTRIBUTIONS["uniform_pos"]((p["batch_size"], p["intermediate_size"]), dtype=dtype, device=device) * 0.1
    B = DISTRIBUTIONS[dist_name]((p["batch_size"], p["state_size"]), dtype=dtype, device=device)
    C = DISTRIBUTIONS[dist_name]((p["batch_size"], p["state_size"]), dtype=dtype, device=device)
    
    return [x, h, dt, B, C]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["intermediate_size"], p["state_size"]]
