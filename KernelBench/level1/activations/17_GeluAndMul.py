import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    GELU and Multiply (GeGLU Activation)
    
    Used by: Gemma, some T5 variants, PaLM
    
    Gated activation: GELU(x1) * x2 where x1 and x2 are split from input.
    This is the core activation pattern in GeGLU FFN blocks.
    
    Shapes:
        Input: (batch_size, seq_length, 2 * intermediate_size)
        Output: (batch_size, seq_length, intermediate_size)
    """
    
    def __init__(self, approximate: str = 'none'):
        """
        Initialize GELU and Multiply.
        
        Args:
            approximate: GELU approximation ('none' for exact, 'tanh' for faster)
        """
        super(Model, self).__init__()
        self.approximate = approximate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GeGLU activation.
        
        Args:
            x: Input tensor (batch_size, seq_length, 2 * intermediate_size)
               Contains concatenated gate and up projections
            
        Returns:
            Activated tensor (batch_size, seq_length, intermediate_size)
        """
        # Split into gate and value
        gate, value = x.chunk(2, dim=-1)
        
        # GELU(gate) * value
        return F.gelu(gate, approximate=self.approximate) * value


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "intermediate_size": 14336},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("activations", "17_GeluAndMul")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_length"], 2 * p["intermediate_size"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return ['none']
