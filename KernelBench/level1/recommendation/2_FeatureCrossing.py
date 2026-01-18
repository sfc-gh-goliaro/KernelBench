import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Feature Crossing
    
    Used by: DCN, DeepFM
    
    Explicit feature crossing via outer product or Hadamard.
    
    Shapes:
        Input: (batch, num_features, embed_dim)
        Output: (batch, interaction_dim)
    """
    
    def __init__(self, num_features: int, embed_dim: int):
        super(Model, self).__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim
        # Pairwise interactions
        self.num_interactions = num_features * (num_features - 1) // 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        interactions = []
        
        for i in range(self.num_features):
            for j in range(i + 1, self.num_features):
                interactions.append((x[:, i] * x[:, j]).sum(dim=-1, keepdim=True))
        
        return torch.cat(interactions, dim=-1)



PARAMETERS = [
    {"batch_size": 4096, "num_features": 26, "embed_dim": 16},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("recommendation", "2_FeatureCrossing")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["num_features"], p["embed_dim"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["num_features"], p["embed_dim"]]
