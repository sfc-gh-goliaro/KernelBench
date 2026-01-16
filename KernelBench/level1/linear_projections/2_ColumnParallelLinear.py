import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Column Parallel Linear Layer
    
    Used by: Megatron-LM, vLLM, TensorRT-LLM (tensor parallelism)
    """
    
    def __init__(self, input_size: int = 4096, output_size: int = 14336,
                 num_partitions: int = 1, partition_idx: int = 0):
        super(Model, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_partitions = num_partitions
        self.partition_idx = partition_idx
        
        assert output_size % num_partitions == 0
        self.output_size_per_partition = output_size // num_partitions
        
        self.weight = nn.Parameter(
            torch.randn(self.output_size_per_partition, input_size) * 0.02
        )
        self.bias = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "input_size": 4096, "output_size": 14336, "num_partitions": 1, "partition_idx": 0},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("linear_projections", "2_ColumnParallelLinear")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["seq_length"], p["input_size"])
    x = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["input_size"], p["output_size"], p["num_partitions"], p["partition_idx"]]
