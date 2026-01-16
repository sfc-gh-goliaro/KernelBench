import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Score Head for Reward Models
    
    Used by: Reward models (Skywork-Reward, ArmoRM, Nemotron-RM)
    """
    
    def __init__(self, hidden_size: int = 4096, num_labels: int = 1):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.score = nn.Linear(hidden_size, num_labels, bias=False)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        scores = self.score(hidden_states)
        if self.num_labels == 1:
            scores = scores.squeeze(-1)
        return scores


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 32, "hidden_size": 4096, "num_labels": 1},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("linear_projections", "5_ScoreHead")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["hidden_size"])
    hidden_states = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [hidden_states]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["hidden_size"], p["num_labels"]]
