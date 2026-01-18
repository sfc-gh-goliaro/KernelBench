import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    SGD Optimizer Step
    
    Used by: CNN training, finetuning
    
    Stochastic gradient descent with optional momentum and dampening.
    
    Shapes:
        param: any shape
        grad: same shape as param
        Output: updated param
    """
    
    def __init__(self, lr: float = 0.01, momentum: float = 0.9, 
                 dampening: float = 0.0, weight_decay: float = 0.0):
        """
        Initialize SGD optimizer step.
        
        Args:
            lr: Learning rate
            momentum: Momentum factor
            dampening: Dampening for momentum
            weight_decay: L2 regularization coefficient
        """
        super(Model, self).__init__()
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
    
    def forward(self, param: torch.Tensor, grad: torch.Tensor,
                momentum_buffer: torch.Tensor = None) -> tuple:
        """
        Perform one SGD step.
        
        Args:
            param: Parameters to update
            grad: Gradients
            momentum_buffer: Momentum buffer (None for first step)
            
        Returns:
            Tuple of (updated_param, updated_momentum_buffer)
        """
        # Weight decay
        if self.weight_decay != 0:
            grad = grad + self.weight_decay * param
        
        # Momentum
        if self.momentum != 0:
            if momentum_buffer is None:
                momentum_buffer = grad.clone()
            else:
                momentum_buffer = self.momentum * momentum_buffer + (1 - self.dampening) * grad
            grad = momentum_buffer
        
        # Update parameters
        param = param - self.lr * grad
        
        return param, momentum_buffer


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"param_size": (4096, 4096)},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("optimizers", "3_SGD")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    param = DISTRIBUTIONS[dist_name]((*p["param_size"]), dtype=dtype, device=device)
    grad = DISTRIBUTIONS[dist_name]((*p["param_size"]), dtype=dtype, device=device)
    momentum_buffer = DISTRIBUTIONS[dist_name]((*p["param_size"]), dtype=dtype, device=device)
    return [param, grad, momentum_buffer]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return [0.01, 0.9, 0.0, 0.0]
