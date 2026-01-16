import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    AllGather (Simulated)
    
    Used by: Tensor parallel (weight gathering)
    
    Gather tensor shards from all ranks into full tensor on each rank.
    This is a simulated single-GPU version.
    
    Shapes:
        Input: (batch, seq_len, hidden_size // num_ranks) per rank
        Output: (batch, seq_len, hidden_size) gathered tensor
    """
    
    def __init__(self, num_ranks: int = 8, gather_dim: int = -1):
        """
        Initialize simulated AllGather.
        
        Args:
            num_ranks: Number of simulated ranks
            gather_dim: Dimension to gather along
        """
        super(Model, self).__init__()
        self.num_ranks = num_ranks
        self.gather_dim = gather_dim
    
    def forward(self, *shards: torch.Tensor) -> torch.Tensor:
        """
        Simulate AllGather by concatenating shards.
        
        Args:
            shards: Tensor shards from each rank
            
        Returns:
            Gathered full tensor
        """
        return torch.cat(shards, dim=self.gather_dim)


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "hidden_size": 4096, "num_ranks": 8, "gather_dim": -1},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("communication", "2_AllGather")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shard_size = p["hidden_size"] // p["num_ranks"]
    shape = (p["batch_size"], p["seq_length"], shard_size)
    shards = [DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device) 
              for _ in range(p["num_ranks"])]
    return shards

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["num_ranks"], p["gather_dim"]]
