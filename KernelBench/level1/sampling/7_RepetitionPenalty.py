import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Repetition Penalty
    
    Used by: Repetition control in generation
    """
    
    def __init__(self, penalty: float = 1.2):
        super(Model, self).__init__()
        self.penalty = penalty
    
    def forward(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, vocab_size = logits.shape
        
        penalty_mask = torch.zeros_like(logits)
        for b in range(batch_size):
            unique_tokens = input_ids[b].unique()
            penalty_mask[b, unique_tokens] = 1.0
        
        penalized_logits = logits.clone()
        
        positive_mask = (logits > 0) & (penalty_mask > 0)
        penalized_logits[positive_mask] = logits[positive_mask] / self.penalty
        
        negative_mask = (logits < 0) & (penalty_mask > 0)
        penalized_logits[negative_mask] = logits[negative_mask] * self.penalty
        
        return penalized_logits


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 64, "vocab_size": 32000, "context_len": 512, "penalty": 1.2},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("sampling", "7_RepetitionPenalty")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    logits = DISTRIBUTIONS[dist_name]((p["batch_size"], p["vocab_size"]), dtype=dtype, device=device)
    input_ids = DISTRIBUTIONS["indices"]((p["batch_size"], p["context_len"]), p["vocab_size"], dtype=torch.int64, device=device)
    return [logits, input_ids]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["penalty"]]
