import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Temperature Scaling
    
    Used by: All LLM inference
    """
    
    def __init__(self, temperature: float = 1.0):
        super(Model, self).__init__()
        assert temperature > 0, "Temperature must be positive"
        self.temperature = temperature
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        scaled_logits = logits / self.temperature
        return F.softmax(scaled_logits, dim=-1)


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 64, "vocab_size": 32000, "temperature": 0.7},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("sampling", "4_Temperature")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["vocab_size"])
    logits = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [logits]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["temperature"]]
