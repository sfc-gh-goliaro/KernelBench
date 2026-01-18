import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    LAMB Optimizer Step
    
    Used by: Large-batch distributed training
    
    Layer-wise Adaptive Moments for large batch training with
    trust ratio normalization.
    
    Shapes:
        param: any shape
        grad: same shape as param
        Output: updated param
    """
    
    def __init__(self, lr: float = 1e-3, betas: tuple = (0.9, 0.999),
                 eps: float = 1e-6, weight_decay: float = 0.01):
        """
        Initialize LAMB optimizer step.
        
        Args:
            lr: Learning rate
            betas: Coefficients for running averages
            eps: Term for numerical stability
            weight_decay: Weight decay coefficient
        """
        super(Model, self).__init__()
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
    
    def forward(self, param: torch.Tensor, grad: torch.Tensor,
                m: torch.Tensor, v: torch.Tensor, step: int) -> tuple:
        """
        Perform one LAMB step.
        
        Args:
            param: Parameters to update
            grad: Gradients
            m: First moment estimate
            v: Second moment estimate
            step: Current step number
            
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
        
        # Compute Adam update direction
        adam_update = m_hat / (v_hat.sqrt() + self.eps)
        
        # Add weight decay
        if self.weight_decay != 0:
            adam_update = adam_update + self.weight_decay * param
        
        # Compute trust ratio
        param_norm = param.norm()
        update_norm = adam_update.norm()
        
        if param_norm > 0 and update_norm > 0:
            trust_ratio = param_norm / update_norm
        else:
            trust_ratio = 1.0
        
        # Update parameters with trust ratio
        param = param - self.lr * trust_ratio * adam_update
        
        return param, m, v


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"param_size": (4096, 4096)},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("optimizers", "4_LAMB")

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
    return [1e-3, (0.9, 0.999), 1e-6, 0.01]
