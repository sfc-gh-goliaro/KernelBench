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
    """
    
    def __init__(self, lr: float = 1e-3, betas: tuple = (0.9, 0.999),
                 eps: float = 1e-6, weight_decay: float = 0.01):
        super(Model, self).__init__()
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
    
    def forward(self, param: torch.Tensor, grad: torch.Tensor,
                m: torch.Tensor, v: torch.Tensor, step: int) -> tuple:
        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * grad.pow(2)
        
        m_hat = m / (1 - self.beta1 ** step)
        v_hat = v / (1 - self.beta2 ** step)
        
        adam_update = m_hat / (v_hat.sqrt() + self.eps)
        
        if self.weight_decay != 0:
            adam_update = adam_update + self.weight_decay * param
        
        param_norm = param.norm()
        update_norm = adam_update.norm()
        
        if param_norm > 0 and update_norm > 0:
            trust_ratio = param_norm / update_norm
        else:
            trust_ratio = 1.0
        
        param = param - self.lr * trust_ratio * adam_update
        
        return param, m, v


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"param_shape": (4096, 4096), "lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-6, "weight_decay": 0.01, "step": 100},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("optimizers", "4_LAMB")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    param = DISTRIBUTIONS[dist_name](p["param_shape"], dtype=dtype, device=device)
    grad = DISTRIBUTIONS[dist_name](p["param_shape"], dtype=dtype, device=device)
    m = torch.zeros(p["param_shape"], dtype=dtype, device=device)
    v = torch.zeros(p["param_shape"], dtype=dtype, device=device)
    return [param, grad, m, v, p["step"]]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["lr"], p["betas"], p["eps"], p["weight_decay"]]
