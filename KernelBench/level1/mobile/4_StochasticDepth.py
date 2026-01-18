import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Stochastic Depth (Drop Path)
    
    Used by: EfficientNet, ConvNeXt, ViT
    
    Drop path / stochastic depth for regularization during training.
    Randomly drops entire residual branches.
    
    Shapes:
        Input: any shape
        Output: same shape as input
    """
    
    def __init__(self, drop_prob: float = 0.1):
        """
        Initialize stochastic depth.
        
        Args:
            drop_prob: Probability of dropping the path
        """
        super(Model, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply stochastic depth.
        
        Args:
            x: Residual branch output
            
        Returns:
            Possibly dropped output (scaled during training)
        """
        if not self.training or self.drop_prob == 0:
            return x
        
        keep_prob = 1 - self.drop_prob
        # Per-sample drop (same drop decision for all spatial locations)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        
        if keep_prob > 0:
            random_tensor.div_(keep_prob)
        
        return x * random_tensor


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 32, "channels": 96, "height": 56, "width": 56},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("mobile", "4_StochasticDepth")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["channels"], p["height"], p["width"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return [0.1]
