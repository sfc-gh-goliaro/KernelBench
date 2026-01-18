import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Adam Optimizer Step
    
    Used by: All DNN training
    
    Adam optimizer: momentum + adaptive learning rate with bias correction.
    This implements a single optimizer step.
    
    Shapes:
        param: any shape
        grad: same shape as param
        Output: updated param
    """
    
    def __init__(self, lr: float = 1e-3, betas: tuple = (0.9, 0.999), eps: float = 1e-8):
        """
        Initialize Adam optimizer step.
        
        Args:
            lr: Learning rate
            betas: Coefficients for running averages (beta1, beta2)
            eps: Term for numerical stability
        """
        super(Model, self).__init__()
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
    
    def forward(self, param: torch.Tensor, grad: torch.Tensor, 
                m: torch.Tensor, v: torch.Tensor, step: int) -> tuple:
        """
        Perform one Adam step.
        
        Args:
            param: Parameters to update
            grad: Gradients
            m: First moment estimate (momentum)
            v: Second moment estimate (adaptive LR)
            step: Current step number (for bias correction)
            
        Returns:
            Tuple of (updated_param, updated_m, updated_v)
        """
        # Update biased first moment estimate
        m = self.beta1 * m + (1 - self.beta1) * grad
        
        # Update biased second moment estimate
        v = self.beta2 * v + (1 - self.beta2) * grad.pow(2)
        
        # Bias correction
        m_hat = m / (1 - self.beta1 ** step)
        v_hat = v / (1 - self.beta2 ** step)
        
        # Update parameters
        param = param - self.lr * m_hat / (v_hat.sqrt() + self.eps)
        
        return param, m, v


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"param_size": (4096, 4096)},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("optimizers", "1_Adam")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    param = DISTRIBUTIONS[dist_name]((*p["param_size"]), dtype=dtype, device=device)
    grad = DISTRIBUTIONS[dist_name]((*p["param_size"]), dtype=dtype, device=device)
    m = DISTRIBUTIONS[dist_name]((*p["param_size"]), dtype=dtype, device=device)
    v = DISTRIBUTIONS[dist_name]((*p["param_size"]), dtype=dtype, device=device)
    return [param, grad, m, v, step]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return [1e-3, (0.9, 0.999), 1e-8]
