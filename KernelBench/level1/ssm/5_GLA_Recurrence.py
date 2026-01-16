import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Gated Linear Attention (GLA) Recurrence
    
    Used by: GLA models
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 8, expand_k: float = 1.0, expand_v: float = 2.0):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.key_dim = int(self.head_dim * expand_k)
        self.value_dim = int(self.head_dim * expand_v)
        
        self.q_proj = nn.Linear(hidden_size, num_heads * self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * self.value_dim, bias=False)
        self.g_proj = nn.Linear(hidden_size, num_heads * self.key_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.value_dim, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.key_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.key_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.value_dim)
        g = self.g_proj(x).view(batch_size, seq_len, self.num_heads, self.key_dim)
        
        g = torch.sigmoid(g)
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        k = k * g
        
        S = torch.zeros(batch_size, self.num_heads, self.key_dim, self.value_dim, 
                       device=x.device, dtype=x.dtype)
        
        outputs = []
        for t in range(seq_len):
            S = S + torch.einsum('bhk,bhv->bhkv', k[:, t], v[:, t])
            o = torch.einsum('bhk,bhkv->bhv', q[:, t], S)
            z = torch.einsum('bhk,bhk->bh', q[:, t], k[:, t].cumsum(dim=1)[:, t] if t > 0 else k[:, t])
            o = o / (z.unsqueeze(-1) + 1e-6)
            outputs.append(o)
        
        output = torch.stack(outputs, dim=1)
        output = output.reshape(batch_size, seq_len, -1)
        
        return self.o_proj(output)


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "hidden_size": 4096, "num_heads": 32},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("ssm", "5_GLA_Recurrence")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_length"], p["hidden_size"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["hidden_size"], p["num_heads"]]
