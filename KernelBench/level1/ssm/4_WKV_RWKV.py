import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    RWKV WKV (Time Mixing)
    
    Used by: RWKV-4, RWKV-5, RWKV-6
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 1):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.time_decay = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.time_first = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, hidden_size) * 0.5)
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, hidden_size) * 0.5)
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, hidden_size) * 0.5)
    
    def forward(self, x: torch.Tensor, state: torch.Tensor = None) -> tuple:
        batch_size, seq_len, _ = x.shape
        
        if state is None:
            state = torch.zeros(batch_size, self.num_heads, self.head_dim, 3, 
                               device=x.device, dtype=x.dtype)
        
        xx = torch.cat([state[:, :1, :, 0].view(batch_size, 1, -1), x[:, :-1]], dim=1)
        
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        r = torch.sigmoid(r)
        
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        r = r.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        w = -torch.exp(self.time_decay)
        u = self.time_first
        
        outputs = []
        aa, bb, pp = state[:, :, :, 0], state[:, :, :, 1], state[:, :, :, 2]
        
        for t in range(seq_len):
            kt, vt = k[:, t], v[:, t]
            
            ww = u + kt
            qq = torch.maximum(pp, ww)
            e1 = torch.exp(pp - qq)
            e2 = torch.exp(ww - qq)
            
            wkv = (e1 * aa + e2 * vt) / (e1 * bb + e2)
            
            ww = w + pp
            qq = torch.maximum(ww, kt)
            e1 = torch.exp(ww - qq)
            e2 = torch.exp(kt - qq)
            
            aa = e1 * aa + e2 * vt
            bb = e1 * bb + e2
            pp = qq
            
            outputs.append(wkv)
        
        wkv = torch.stack(outputs, dim=1)
        out = (r * wkv).view(batch_size, seq_len, self.hidden_size)
        out = self.output(out)
        
        new_state = torch.stack([aa, bb, pp], dim=-1)
        
        return out, new_state


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "hidden_size": 4096, "num_heads": 32},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("ssm", "4_WKV_RWKV")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_length"], p["hidden_size"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["hidden_size"], p["num_heads"]]
