import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    RetNet Retention
    
    Used by: RetNet
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 8, double_v_dim: bool = True):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.v_dim = self.head_dim * 2 if double_v_dim else self.head_dim
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * self.v_dim, bias=False)
        self.g_proj = nn.Linear(hidden_size, num_heads * self.v_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.v_dim, hidden_size, bias=False)
        
        decay_rates = 1 - torch.pow(2, -5 - torch.arange(num_heads, dtype=torch.float))
        self.register_buffer('decay_rates', decay_rates)
        
        self.scale = self.head_dim ** -0.5
    
    def _parallel_retention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_heads, head_dim = q.shape
        
        positions = torch.arange(seq_len, device=q.device)
        decay_mask = positions.unsqueeze(0) - positions.unsqueeze(1)
        decay_mask = decay_mask.float()
        
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device))
        
        D = self.decay_rates.view(1, num_heads, 1, 1) ** decay_mask.unsqueeze(0).unsqueeze(0)
        D = D * causal_mask
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        qk = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        qk = qk * D
        
        output = torch.matmul(qk, v)
        return output.transpose(1, 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.v_dim)
        g = self.g_proj(x).view(batch_size, seq_len, self.num_heads, self.v_dim)
        
        retention_out = self._parallel_retention(q, k, v)
        output = retention_out * F.silu(g)
        
        output = output.reshape(batch_size, seq_len, -1)
        return self.o_proj(output)


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "hidden_size": 4096, "num_heads": 32},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("ssm", "6_Retention")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_length"], p["hidden_size"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["hidden_size"], p["num_heads"]]
