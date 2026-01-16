import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Fused QKV Parallel Linear Projection
    
    Used by: vLLM, TensorRT-LLM, most inference frameworks
    """
    
    def __init__(self, hidden_size: int = 4096, num_heads: int = 32, 
                 num_kv_heads: int = 8, head_dim: int = 128):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        total_size = self.q_size + 2 * self.kv_size
        
        self.qkv_proj = nn.Linear(hidden_size, total_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> tuple:
        batch_size, seq_length, _ = x.shape
        qkv = self.qkv_proj(x)
        
        q = qkv[:, :, :self.q_size]
        k = qkv[:, :, self.q_size:self.q_size + self.kv_size]
        v = qkv[:, :, self.q_size + self.kv_size:]
        
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_length, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_length, self.num_kv_heads, self.head_dim)
        
        return q, k, v


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "hidden_size": 4096, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("linear_projections", "1_QKVParallelLinear")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["seq_length"], p["hidden_size"])
    x = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["hidden_size"], p["num_heads"], p["num_kv_heads"], p["head_dim"]]
