import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Token Verification (Speculative Decoding)
    
    Used by: All speculative decoding methods
    """
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, draft_probs: torch.Tensor, target_probs: torch.Tensor,
                draft_tokens: torch.Tensor) -> tuple:
        batch_size, num_draft, vocab_size = draft_probs.shape
        device = draft_probs.device
        
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
        draft_indices = torch.arange(num_draft, device=device).unsqueeze(0)
        
        p_draft = draft_probs[batch_indices, draft_indices, draft_tokens]
        p_target = target_probs[batch_indices, draft_indices, draft_tokens]
        
        acceptance_prob = torch.clamp(p_target / (p_draft + 1e-10), max=1.0)
        u = torch.rand_like(acceptance_prob)
        
        accepted = u < acceptance_prob
        cumulative_accepted = accepted.cumprod(dim=1)
        num_accepted = cumulative_accepted.sum(dim=1)
        
        return num_accepted, cumulative_accepted


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 64, "num_draft": 5, "vocab_size": 32000},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("speculative", "2_TokenVerification")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    
    draft_probs = F.softmax(DISTRIBUTIONS[dist_name]((p["batch_size"], p["num_draft"], p["vocab_size"]), dtype=dtype, device=device), dim=-1)
    target_probs = F.softmax(DISTRIBUTIONS[dist_name]((p["batch_size"], p["num_draft"], p["vocab_size"]), dtype=dtype, device=device), dim=-1)
    draft_tokens = DISTRIBUTIONS["indices"]((p["batch_size"], p["num_draft"]), p["vocab_size"], dtype=torch.int64, device=device)
    
    return [draft_probs, target_probs, draft_tokens]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
