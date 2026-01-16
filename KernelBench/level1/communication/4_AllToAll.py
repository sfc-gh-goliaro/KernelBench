import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    AllToAll (Simulated)
    
    Used by: Expert parallel MoE
    
    Transpose/exchange data between ranks (each rank sends different
    data to each other rank). Used for MoE expert parallelism.
    
    Shapes:
        Input: (num_ranks, tokens_per_rank, hidden_size)
        Output: (num_ranks, tokens_per_rank, hidden_size) transposed across ranks
    """
    
    def __init__(self, num_ranks: int = 8):
        """
        Initialize simulated AllToAll.
        
        Args:
            num_ranks: Number of simulated ranks
        """
        super(Model, self).__init__()
        self.num_ranks = num_ranks
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simulate AllToAll by transposing the rank dimension.
        
        In MoE, this exchanges tokens between expert-parallel ranks
        so each rank processes all tokens for its assigned experts.
        
        Args:
            x: Input tensor (num_ranks, tokens_per_rank, hidden_size)
               where first dim represents data destined for each rank
            
        Returns:
            Transposed tensor where each "rank" now has data from all other ranks
        """
        # In real AllToAll, each rank i sends x[i, j, :] to rank j
        # and receives from rank j its x[j, i, :]
        # This simulates by transposing first two dimensions
        return x.transpose(0, 1).contiguous()


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"num_ranks": 8, "tokens_per_rank": 2048, "hidden_size": 4096},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("communication", "4_AllToAll")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["num_ranks"], p["tokens_per_rank"], p["hidden_size"])
    x = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["num_ranks"]]
