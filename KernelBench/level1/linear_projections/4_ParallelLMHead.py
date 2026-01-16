import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Parallel Language Model Head
    
    Used by: vLLM, TensorRT-LLM, Megatron-LM
    """
    
    def __init__(self, hidden_size: int = 4096, vocab_size: int = 128256):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(x)


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "hidden_size": 4096, "vocab_size": 128256},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("linear_projections", "4_ParallelLMHead")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["seq_length"], p["hidden_size"])
    x = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["hidden_size"], p["vocab_size"]]
