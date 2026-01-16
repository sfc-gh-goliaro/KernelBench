import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Combined Top-K and Top-P Sampling
    
    Used by: vLLM, SGLang default sampler
    """
    
    def __init__(self, k: int = 50, p: float = 0.9, temperature: float = 1.0):
        super(Model, self).__init__()
        self.k = k
        self.p = p
        self.temperature = temperature
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        if self.k > 0:
            top_k_threshold = torch.topk(logits, self.k, dim=-1).values[..., -1, None]
            logits = torch.where(logits < top_k_threshold, 
                                torch.full_like(logits, float('-inf')), logits)
        
        if self.p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > self.p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        probs = F.softmax(logits, dim=-1)
        sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        return sampled_tokens


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 64, "vocab_size": 32000, "k": 50, "p": 0.9},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("sampling", "3_TopK_TopP")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["vocab_size"])
    logits = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [logits]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["k"], p["p"]]
