import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Selective Scan (Mamba S6)
    
    Used by: Mamba, Jamba (SSM layers)
    """
    
    def __init__(self, d_inner: int, d_state: int = 16):
        super(Model, self).__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_inner, -1)
        self.register_buffer('A_log', torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
    
    def forward(self, x: torch.Tensor, delta: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        A = -torch.exp(self.A_log)
        
        delta_A = torch.exp(delta.unsqueeze(-1) * A)
        delta_B = delta.unsqueeze(-1) * B.unsqueeze(2)
        delta_B_x = delta_B * x.unsqueeze(-1)
        
        h = torch.zeros(batch_size, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        
        for t in range(seq_len):
            h = delta_A[:, t] * h + delta_B_x[:, t]
            y = (h * C[:, t].unsqueeze(1)).sum(dim=-1)
            outputs.append(y)
        
        y = torch.stack(outputs, dim=1)
        y = y + x * self.D
        
        return y


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "d_inner": 4096, "d_state": 16},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("ssm", "1_SelectiveScan")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_length"], p["d_inner"]), dtype=dtype, device=device)
    delta = F.softplus(DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_length"], p["d_inner"]), dtype=dtype, device=device))
    B = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_length"], p["d_state"]), dtype=dtype, device=device)
    C = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_length"], p["d_state"]), dtype=dtype, device=device)
    
    return [x, delta, B, C]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["d_inner"], p["d_state"]]
