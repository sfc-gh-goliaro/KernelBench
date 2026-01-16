import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Mamba-2 State Space Duality (SSD)
    
    Used by: Mamba-2, Codestral Mamba
    """
    
    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, 
                 expand: int = 2, chunk_size: int = 256):
        super(Model, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand
        self.chunk_size = chunk_size
        
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner + 2 * d_state + 1, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv, 
            padding=d_conv - 1, groups=self.d_inner
        )
        
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.register_buffer('A_log', torch.log(A))
        
        self.D = nn.Parameter(torch.ones(self.d_inner))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        xz = self.in_proj(x)
        x_proj, z, B, C, dt = xz.split(
            [self.d_inner, self.d_inner, self.d_state, self.d_state, 1], dim=-1
        )
        
        x_conv = x_proj.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        dt = F.softplus(dt)
        A = -torch.exp(self.A_log)
        
        h = torch.zeros(batch_size, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        
        for t in range(seq_len):
            dt_t = dt[:, t]
            dA = torch.exp(dt_t.unsqueeze(-1) * A)
            dB = dt_t.unsqueeze(-1) * B[:, t].unsqueeze(1)
            
            h = dA * h + dB * x_conv[:, t].unsqueeze(-1)
            y = (h * C[:, t].unsqueeze(1)).sum(dim=-1)
            outputs.append(y)
        
        y = torch.stack(outputs, dim=1)
        y = y + x_conv * self.D
        y = y * F.silu(z)
        
        return self.out_proj(y)


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "d_model": 4096, "d_state": 64},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("ssm", "2_Mamba2_SSD")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_length"], p["d_model"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["d_model"], p["d_state"]]
