import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Top-K Sampling
    
    Used by: All LLM inference
    """
    
    def __init__(self, k: int = 50, temperature: float = 1.0):
        super(Model, self).__init__()
        self.k = k
        self.temperature = temperature
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        top_k_values, top_k_indices = torch.topk(logits, self.k, dim=-1)
        probs = F.softmax(top_k_values, dim=-1)
        sampled_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        batch_indices = torch.arange(logits.shape[0], device=logits.device)
        sampled_tokens = top_k_indices[batch_indices, sampled_idx]
        
        return sampled_tokens


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 64, "vocab_size": 32000, "k": 50},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("sampling", "1_TopK")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["vocab_size"])
    logits = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [logits]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["k"]]
