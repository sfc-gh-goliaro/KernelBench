import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Last Token Pooling
    
    Used by: Reward models, some embedding models, decoder-only encoders
    """
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states[:, -1, :]
        
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.shape[0]
        
        return hidden_states[torch.arange(batch_size, device=hidden_states.device), 
                            sequence_lengths.long()]


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 32, "seq_length": 512, "hidden_size": 4096},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("pooling", "10_LastTokenPooling")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["seq_length"], p["hidden_size"])
    hidden_states = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    # Variable length sequences
    attention_mask = torch.ones(p["batch_size"], p["seq_length"], device=device)
    for i in range(p["batch_size"]):
        length = torch.randint(100, p["seq_length"] + 1, (1,)).item()
        attention_mask[i, length:] = 0
    return [hidden_states, attention_mask]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
