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
    
    Hashing trick for very large categorical feature spaces.
    
    Shapes:
        Input: (batch, num_features) IDs (can be very large)
        Output: (batch, num_features, embed_dim)
    """
    
    def __init__(self, num_buckets: int, embed_dim: int):
        super(Model, self).__init__()
        self.num_buckets = num_buckets
        self.embedding = nn.Embedding(num_buckets, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Hash to bucket
        hashed = x % self.num_buckets
        return self.embedding(hashed)



PARAMETERS = [
    {"batch_size": 4096, "num_features": 26, "num_buckets": 100000, "embed_dim": 16},
    # DLRM: hash embedding for large-scale categorical features
    {"batch_size": 2048, "num_features": 26, "num_buckets": 1000000, "embed_dim": 64},
    # TikTok Monolith: real-time hash embedding lookups
    {"batch_size": 8192, "num_features": 50, "num_buckets": 500000, "embed_dim": 32},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("recommendation", "8_HashEmbedding")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["num_features"])
    x = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["num_buckets"], p["embed_dim"]]
