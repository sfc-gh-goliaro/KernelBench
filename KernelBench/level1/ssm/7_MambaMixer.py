import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Mamba Mixer (Full S6 Block)
    
    Used by: Mamba-1, Jamba (hybrid layers)
    """
    
    def __init__(self, hidden_size: int = 2560, state_size: int = 16,
                 conv_kernel_size: int = 4, expand_factor: int = 2):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.conv_kernel_size = conv_kernel_size
        self.intermediate_size = hidden_size * expand_factor
        
        self.in_proj = nn.Linear(hidden_size, self.intermediate_size * 2, bias=False)
        
        self.conv1d = nn.Conv1d(
            self.intermediate_size, self.intermediate_size,
            kernel_size=conv_kernel_size, padding=conv_kernel_size - 1,
            groups=self.intermediate_size
        )
        
        self.x_proj = nn.Linear(
            self.intermediate_size, 
            state_size * 2 + 1,
            bias=False
        )
        
        self.dt_proj = nn.Linear(1, self.intermediate_size, bias=True)
        
        A = torch.arange(1, state_size + 1, dtype=torch.float32).repeat(self.intermediate_size, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.intermediate_size))
        
        self.out_proj = nn.Linear(self.intermediate_size, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = x.shape
        
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)
        
        x_conv = x_proj.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_length]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        ssm_params = self.x_proj(x_conv)
        dt = ssm_params[:, :, :1]
        B = ssm_params[:, :, 1:1+self.state_size]
        C = ssm_params[:, :, 1+self.state_size:]
        
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)
        
        y = self._selective_scan(x_conv, dt, A, B, C)
        y = y + self.D * x_conv
        y = y * F.silu(z)
        
        return self.out_proj(y)
    
    def _selective_scan(self, x, dt, A, B, C):
        batch_size, seq_length, intermediate = x.shape
        
        h = torch.zeros(batch_size, intermediate, self.state_size, device=x.device)
        outputs = []
        
        for t in range(seq_length):
            x_t = x[:, t, :]
            dt_t = dt[:, t, :]
            B_t = B[:, t, :]
            C_t = C[:, t, :]
            
            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)
            
            h = dA * h + dB * x_t.unsqueeze(-1)
            y_t = (h * C_t.unsqueeze(1)).sum(-1)
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=1)


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "seq_length": 512, "hidden_size": 2560, "state_size": 16, "conv_kernel_size": 4, "expand_factor": 2},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("ssm", "7_MambaMixer")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_length"], p["hidden_size"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["hidden_size"], p["state_size"], p["conv_kernel_size"], p["expand_factor"]]
