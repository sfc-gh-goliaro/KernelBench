import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    AllReduce (Simulated)
    
    Used by: Tensor parallel (post-GEMM reduction)
    
    Sum tensor values across all distributed ranks, result replicated
    on all ranks. This is a simulated single-GPU version.
    
    Shapes:
        Input: (batch, seq_len, hidden_size)
        Output: (batch, seq_len, hidden_size) - sum across "ranks"
    """
    
    def __init__(self, num_ranks: int = 8):
        """
        Initialize simulated AllReduce.
        
        Args:
            num_ranks: Number of simulated ranks
        """
        super(Model, self).__init__()
        self.num_ranks = num_ranks
    
    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        """
        Simulate AllReduce by summing input tensors.
        
        Args:
            tensors: Variable number of tensors to reduce (simulating different ranks)
            
        Returns:
            Sum of all input tensors
        """
        if len(tensors) == 1:
            # Single tensor: simulate by splitting and reducing
            x = tensors[0]
            # In real TP, this would sum partial results from different ranks
            return x
        else:
            # Multiple tensors: sum them (simulating reduction)
            result = tensors[0]
            for t in tensors[1:]:
                result = result + t
            return result


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "hidden_size": 4096, "num_ranks": 8},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("communication", "1_AllReduce")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["seq_length"], p["hidden_size"])
    # Simulate 2 rank partial results
    x1 = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    x2 = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [x1, x2]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["num_ranks"]]
