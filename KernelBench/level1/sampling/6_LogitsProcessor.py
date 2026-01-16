import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Logits Processor
    
    Used by: All LLM inference
    """
    
    def __init__(self, temperature: float = 1.0, top_k: int = 0):
        super(Model, self).__init__()
        self.temperature = temperature
        self.top_k = top_k
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        if self.top_k > 0:
            top_k_values = torch.topk(logits, self.top_k, dim=-1).values
            threshold = top_k_values[..., -1, None]
            logits = torch.where(logits < threshold, 
                                torch.full_like(logits, float('-inf')), logits)
        
        return logits


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 64, "vocab_size": 32000, "temperature": 0.7, "top_k": 50},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("sampling", "6_LogitsProcessor")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["vocab_size"])
    logits = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [logits]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["temperature"], p["top_k"]]
