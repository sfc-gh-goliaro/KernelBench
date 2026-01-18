import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Sparse Embedding Lookup
    
    Used by: DeepFM, WDL, AutoInt
    
    Sparse embedding lookup for categorical features with huge vocab.
    
    Shapes:
        Input: (batch, num_features) categorical indices
        Output: (batch, num_features, embed_dim)
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, num_features: int):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.num_features = num_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)



PARAMETERS = [
    {"batch_size": 4096, "num_features": 26, "vocab_size": 1000000, "embed_dim": 16},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("recommendation", "1_SparseEmbedding")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["num_features"])
    x = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["vocab_size"], p["embed_dim"], p["num_features"]]
