import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Hash Embedding
    
    Used by: Large-scale RecSys
    """
    
    def __init__(self, num_buckets: int, embed_dim: int):
        super(Model, self).__init__()
        self.num_buckets = num_buckets
        self.embedding = nn.Embedding(num_buckets, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hashed = x % self.num_buckets
        return self.embedding(hashed)


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 4096, "num_features": 26, "num_buckets": 100000, "embed_dim": 16, "max_id": 10000000},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("recommendation", "8_HashEmbedding")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.int64, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS["indices"]((p["batch_size"], p["num_features"]), p["max_id"], dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["num_buckets"], p["embed_dim"]]
