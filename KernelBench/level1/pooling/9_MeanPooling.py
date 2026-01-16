import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Mean Pooling for Sequences
    
    Used by: Sentence transformers, embedding models (E5, BGE, GTE)
    """
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        
        return sum_embeddings / sum_mask


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 32, "seq_length": 512, "hidden_size": 768, "mask_length": 400},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("pooling", "9_MeanPooling")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["seq_length"], p["hidden_size"])
    hidden_states = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    attention_mask = torch.ones(p["batch_size"], p["seq_length"], device=device)
    attention_mask[:, p["mask_length"]:] = 0
    return [hidden_states, attention_mask]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
