import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Classification Head
    
    Used by: BERT, RoBERTa, DeBERTa (sequence classification, token classification)
    """
    
    def __init__(self, hidden_size: int = 768, num_classes: int = 2,
                 dropout_prob: float = 0.1):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        self.dropout = nn.Dropout(dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, num_classes)
    
    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        x = self.dropout(pooled_output)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.out_proj(x)
        return logits


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 32, "hidden_size": 768, "num_classes": 2, "dropout_prob": 0.1},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("linear_projections", "6_ClassificationHead")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["hidden_size"])
    pooled_output = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [pooled_output]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["hidden_size"], p["num_classes"], p["dropout_prob"]]
