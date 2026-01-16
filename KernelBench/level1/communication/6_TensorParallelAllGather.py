import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Tensor Parallel AllGather
    
    Used by: vLLM, TensorRT-LLM, Megatron-LM (tensor parallelism)
    
    AllGather operation specifically for tensor parallelism.
    Gathers partial results from column-parallel linear layers.
    Each rank contributes output_size // world_size columns.
    
    Shapes:
        Input: (batch_size, seq_length, output_size_per_rank)
        Output: (batch_size, seq_length, output_size)
    """
    
    def __init__(self, output_size_per_rank: int = 4096, world_size: int = 8):
        """
        Initialize Tensor Parallel AllGather.
        
        Args:
            output_size_per_rank: Output dimension per tensor parallel rank
            world_size: Number of tensor parallel ranks
        """
        super(Model, self).__init__()
        self.output_size_per_rank = output_size_per_rank
        self.world_size = world_size
        self.total_output_size = output_size_per_rank * world_size
    
    def forward(self, local_output: torch.Tensor) -> torch.Tensor:
        """
        Simulate tensor parallel all-gather.
        
        In actual distributed setting:
        - Each rank has partial output (output_size // world_size columns)
        - AllGather combines all partial outputs
        
        Args:
            local_output: Local partial output (batch, seq, output_size_per_rank)
            
        Returns:
            Full gathered output (batch, seq, total_output_size)
        """
        batch_size, seq_length, _ = local_output.shape
        
        # Simulate gathering from all ranks
        # In practice: dist.all_gather(tensor_list, local_output)
        # Here we simulate by replicating (represents memory bandwidth pattern)
        gathered = local_output.repeat(1, 1, self.world_size)
        
        return gathered


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "output_size_per_rank": 512, "world_size": 8},  # 4096 / 8 ranks
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("communication", "6_TensorParallelAllGather")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["seq_length"], p["output_size_per_rank"])
    local_output = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [local_output]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["output_size_per_rank"], p["world_size"]]
