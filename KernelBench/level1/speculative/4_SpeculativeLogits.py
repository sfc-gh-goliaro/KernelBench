import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Speculative Logits Processing
    
    Used by: vLLM, SGLang, TensorRT-LLM speculative decoding
    """
    
    def __init__(self, vocab_size: int = 128256, temperature: float = 1.0):
        super(Model, self).__init__()
        self.vocab_size = vocab_size
        self.temperature = temperature
    
    def forward(self, draft_logits: torch.Tensor, target_logits: torch.Tensor) -> tuple:
        if self.temperature != 1.0:
            draft_logits = draft_logits / self.temperature
            target_logits = target_logits / self.temperature
        
        draft_probs = F.softmax(draft_logits, dim=-1)
        target_probs = F.softmax(target_logits, dim=-1)
        
        num_draft = draft_probs.shape[1]
        kl_div = F.kl_div(
            draft_probs.log(),
            target_probs[:, :num_draft, :],
            reduction='none'
        ).sum(dim=-1)
        
        acceptance_quality = kl_div < 0.1
        
        return draft_probs, target_probs, acceptance_quality


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 32, "num_draft_tokens": 5, "vocab_size": 128256, "temperature": 1.0},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("speculative", "4_SpeculativeLogits")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    
    draft_logits = DISTRIBUTIONS[dist_name]((p["batch_size"], p["num_draft_tokens"], p["vocab_size"]), dtype=dtype, device=device)
    target_logits = DISTRIBUTIONS[dist_name]((p["batch_size"], p["num_draft_tokens"] + 1, p["vocab_size"]), dtype=dtype, device=device)
    
    return [draft_logits, target_logits]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["vocab_size"], p["temperature"]]
