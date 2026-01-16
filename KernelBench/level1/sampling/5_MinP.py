import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Min-P Sampling
    
    Used by: Alternative sampling method
    """
    
    def __init__(self, min_p: float = 0.05, temperature: float = 1.0):
        super(Model, self).__init__()
        self.min_p = min_p
        self.temperature = temperature
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        probs = F.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1, keepdim=True).values
        threshold = self.min_p * max_probs
        
        filtered_probs = torch.where(probs >= threshold, probs, torch.zeros_like(probs))
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
        
        sampled_tokens = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)
        
        return sampled_tokens


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 64, "vocab_size": 32000, "min_p": 0.05},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("sampling", "5_MinP")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["vocab_size"])
    logits = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [logits]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["min_p"]]
