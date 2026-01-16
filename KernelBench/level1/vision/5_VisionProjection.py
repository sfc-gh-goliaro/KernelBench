import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Vision-to-LLM Projection
    
    Used by: LLaVA, Qwen-VL, InternVL
    """
    
    def __init__(self, vision_dim: int, llm_dim: int, hidden_dim: int = None):
        super(Model, self).__init__()
        self.vision_dim = vision_dim
        self.llm_dim = llm_dim
        self.hidden_dim = hidden_dim or llm_dim
        
        self.proj1 = nn.Linear(vision_dim, self.hidden_dim)
        self.proj2 = nn.Linear(self.hidden_dim, llm_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj1(x)
        x = F.gelu(x)
        x = self.proj2(x)
        return x


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "num_patches": 576, "vision_dim": 1024, "llm_dim": 4096},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("vision", "5_VisionProjection")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["num_patches"], p["vision_dim"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["vision_dim"], p["llm_dim"]]
