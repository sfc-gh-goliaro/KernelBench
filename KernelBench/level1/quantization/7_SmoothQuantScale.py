import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    SmoothQuant Scaling
    
    Used by: TensorRT-LLM, vLLM (W8A8 quantization)
    """
    
    def __init__(self, hidden_size: int = 4096, alpha: float = 0.5):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.scales = nn.Parameter(torch.ones(hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scales


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "hidden_size": 4096, "alpha": 0.5},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("quantization", "7_SmoothQuantScale")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["seq_length"], p["hidden_size"])
    x = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["hidden_size"], p["alpha"]]
