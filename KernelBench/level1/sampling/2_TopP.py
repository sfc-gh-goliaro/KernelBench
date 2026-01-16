import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Top-P (Nucleus) Sampling
    
    Used by: All LLM inference
    """
    
    def __init__(self, p: float = 0.9, temperature: float = 1.0):
        super(Model, self).__init__()
        self.p = p
        self.temperature = temperature
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > self.p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        sorted_logits[sorted_indices_to_remove] = float('-inf')
        
        probs = F.softmax(sorted_logits, dim=-1)
        sampled_sorted_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        batch_indices = torch.arange(logits.shape[0], device=logits.device)
        sampled_tokens = sorted_indices[batch_indices, sampled_sorted_idx]
        
        return sampled_tokens


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 64, "vocab_size": 32000, "p": 0.9},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("sampling", "2_TopP")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["vocab_size"])
    logits = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [logits]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["p"]]
