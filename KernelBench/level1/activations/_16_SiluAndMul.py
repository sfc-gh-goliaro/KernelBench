import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    SiLU and Multiply (SwiGLU Activation)
    
    Used by: Llama, Mistral, Qwen, most modern LLMs
    
    Gated activation: SiLU(x1) * x2 where x1 and x2 are split from input.
    This is the core activation pattern in SwiGLU FFN blocks.
    SiLU(x) = x * sigmoid(x)
    
    Shapes:
        Input: (batch_size, seq_length, 2 * intermediate_size)
        Output: (batch_size, seq_length, intermediate_size)
    """
    
    def __init__(self):
        """Initialize SiLU and Multiply."""
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU activation.
        
        Args:
            x: Input tensor (batch_size, seq_length, 2 * intermediate_size)
               Contains concatenated gate and up projections
            
        Returns:
            Activated tensor (batch_size, seq_length, intermediate_size)
        """
        # Split into gate and value
        gate, value = x.chunk(2, dim=-1)
        
        # SiLU(gate) * value
        return F.silu(gate) * value


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "intermediate_size": 14336},
    # Llama-3.1-8B: intermediate_size=14336
    {"batch_size": 8, "seq_length": 4096, "intermediate_size": 14336},
    # Llama-3.1-70B: intermediate_size=28672
    {"batch_size": 4, "seq_length": 4096, "intermediate_size": 28672},
    # Mistral-7B-v0.3: intermediate_size=14336
    {"batch_size": 8, "seq_length": 4096, "intermediate_size": 14336},
    # DeepSeek-V2-Lite: intermediate_size=10944
    {"batch_size": 8, "seq_length": 4096, "intermediate_size": 10944},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("activations", "16_SiluAndMul")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_length"], 2 * p["intermediate_size"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
