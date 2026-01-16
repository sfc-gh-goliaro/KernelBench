import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Adapter Layer
    
    Used by: Adapter-based fine-tuning (Houlsby adapters)
    """
    
    def __init__(self, hidden_size: int = 4096, adapter_size: int = 64,
                 activation: str = 'relu'):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.adapter_size = adapter_size
        
        self.down_proj = nn.Linear(hidden_size, adapter_size)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
        
        self.up_proj = nn.Linear(adapter_size, hidden_size)
        
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.down_proj(x)
        h = self.activation(h)
        h = self.up_proj(h)
        return x + h


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "hidden_size": 4096, "adapter_size": 64, "activation": "relu"},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("peft", "5_AdapterLayer")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["seq_length"], p["hidden_size"])
    x = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["hidden_size"], p["adapter_size"], p["activation"]]
