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
    
    RWKV time-mixing: WKV state update with exponential decay
    and gated output. This is the core recurrence of RWKV.
    
    Shapes:
        Input: (batch, seq_len, hidden_size)
        Output: (batch, seq_len, hidden_size)
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 1):
        """
        Initialize RWKV WKV.
        
        Args:
            hidden_size: Model hidden dimension
            num_heads: Number of heads (RWKV-5+)
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Learnable time decay (per head)
        self.time_decay = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.time_first = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        
        # Projections
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Time mixing parameters
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, hidden_size) * 0.5)
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, hidden_size) * 0.5)
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, hidden_size) * 0.5)
    
    def forward(self, x: torch.Tensor, state: torch.Tensor = None) -> tuple:
        """
        RWKV WKV forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, hidden_size)
            state: Previous state (batch, num_heads, head_dim, 3) or None
            
        Returns:
            Tuple of (output, new_state)
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize state if needed
        if state is None:
            state = torch.zeros(batch_size, self.num_heads, self.head_dim, 3, 
                               device=x.device, dtype=x.dtype)
        
        # Time mixing (shift and blend with previous timestep)
        # In practice, this uses a shifted version of x
        xx = torch.cat([state[:, :1, :, 0].view(batch_size, 1, -1), x[:, :-1]], dim=1)
        
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        
        # Projections
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        r = torch.sigmoid(r)
        
        # Reshape for multi-head
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        r = r.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # WKV computation (sequential for reference; real impl is parallel)
        w = -torch.exp(self.time_decay)  # (num_heads, head_dim)
        u = self.time_first  # (num_heads, head_dim)
        
        outputs = []
        aa, bb, pp = state[:, :, :, 0], state[:, :, :, 1], state[:, :, :, 2]
        
        for t in range(seq_len):
            kt, vt = k[:, t], v[:, t]  # (batch, num_heads, head_dim)
            
            # WKV formula
            ww = u + kt
            qq = torch.maximum(pp, ww)
            e1 = torch.exp(pp - qq)
            e2 = torch.exp(ww - qq)
            
            wkv = (e1 * aa + e2 * vt) / (e1 * bb + e2)
            
            # Update state
            ww = w + pp
            qq = torch.maximum(ww, kt)
            e1 = torch.exp(ww - qq)
            e2 = torch.exp(kt - qq)
            
            aa = e1 * aa + e2 * vt
            bb = e1 * bb + e2
            pp = qq
            
            outputs.append(wkv)
        
        # Stack outputs
        wkv = torch.stack(outputs, dim=1)  # (batch, seq, num_heads, head_dim)
        
        # Apply receptance and output projection
        out = (r * wkv).view(batch_size, seq_len, self.hidden_size)
        out = self.output(out)
        
        # Update state
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
