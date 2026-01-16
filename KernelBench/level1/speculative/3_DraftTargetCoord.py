import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Draft-Target Coordination for Speculative Decoding
    
    Used by: vLLM, SGLang speculative decoding
    """
    
    def __init__(self, vocab_size: int = 128256):
        super(Model, self).__init__()
        self.vocab_size = vocab_size
    
    def forward(self, draft_probs: torch.Tensor, target_probs: torch.Tensor,
                draft_tokens: torch.Tensor) -> tuple:
        batch_size, num_draft, _ = draft_probs.shape
        device = draft_probs.device
        
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
        pos_indices = torch.arange(num_draft, device=device).unsqueeze(0)
        
        p_draft = draft_probs[batch_indices, pos_indices, draft_tokens]
        p_target = target_probs[batch_indices, pos_indices, draft_tokens]
        
        acceptance_prob = torch.clamp(p_target / (p_draft + 1e-10), max=1.0)
        uniform_samples = torch.rand_like(acceptance_prob)
        accepted = uniform_samples < acceptance_prob
        
        accept_cumsum = accepted.cumprod(dim=1)
        num_accepted = accept_cumsum.sum(dim=1)
        
        max_accepted = num_accepted.max().item()
        output_length = max_accepted + 1
        
        accepted_tokens = torch.zeros(batch_size, output_length, dtype=torch.long, device=device)
        
        for b in range(batch_size):
            n = num_accepted[b].item()
            if n > 0:
                accepted_tokens[b, :n] = draft_tokens[b, :n]
            bonus_token = torch.multinomial(target_probs[b, n], 1)
            accepted_tokens[b, n] = bonus_token
        
        return accepted_tokens, num_accepted


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 32, "num_draft_tokens": 5, "vocab_size": 128256},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("speculative", "3_DraftTargetCoord")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    
    draft_logits = DISTRIBUTIONS[dist_name]((p["batch_size"], p["num_draft_tokens"], p["vocab_size"]), dtype=dtype, device=device)
    target_logits = DISTRIBUTIONS[dist_name]((p["batch_size"], p["num_draft_tokens"] + 1, p["vocab_size"]), dtype=dtype, device=device)
    
    draft_probs = torch.softmax(draft_logits, dim=-1)
    target_probs = torch.softmax(target_logits, dim=-1)
    draft_tokens = DISTRIBUTIONS["indices"]((p["batch_size"], p["num_draft_tokens"]), p["vocab_size"], dtype=torch.int64, device=device)
    
    return [draft_probs, target_probs, draft_tokens]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["vocab_size"]]
