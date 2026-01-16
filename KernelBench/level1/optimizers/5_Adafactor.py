import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Adafactor Optimizer Step
    
    Used by: T5, large model training
    """
    
    def __init__(self, lr: float = None, eps: tuple = (1e-30, 1e-3),
                 clip_threshold: float = 1.0, decay_rate: float = -0.8,
                 beta1: float = None, weight_decay: float = 0.0):
        super(Model, self).__init__()
        self.lr = lr
        self.eps1, self.eps2 = eps
        self.clip_threshold = clip_threshold
        self.decay_rate = decay_rate
        self.beta1 = beta1
        self.weight_decay = weight_decay
    
    def _rms(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.pow(2).mean().sqrt()
    
    def forward(self, param: torch.Tensor, grad: torch.Tensor,
                exp_avg_sq_row: torch.Tensor, exp_avg_sq_col: torch.Tensor,
                step: int) -> tuple:
        rho = min(self.decay_rate, -(step ** -0.8))
        rho = 1.0 - (1.0 + rho)
        
        grad_sqr = grad.pow(2) + self.eps1
        
        row_mean = grad_sqr.mean(dim=-1)
        col_mean = grad_sqr.mean(dim=-2)
        
        exp_avg_sq_row = rho * exp_avg_sq_row + (1 - rho) * row_mean
        exp_avg_sq_col = rho * exp_avg_sq_col + (1 - rho) * col_mean
        
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean()).unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2)
        v = r_factor * c_factor
        
        update = grad / v.sqrt()
        
        rms = self._rms(update)
        if rms > self.clip_threshold:
            update = update * (self.clip_threshold / rms)
        
        if self.lr is None:
            lr = max(self.eps2, self._rms(param)) * (step ** -0.5)
        else:
            lr = self.lr
        
        if self.weight_decay != 0:
            param = param - self.weight_decay * lr * param
        
        param = param - lr * update
        
        return param, exp_avg_sq_row, exp_avg_sq_col


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"rows": 4096, "cols": 4096, "step": 100},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("optimizers", "5_Adafactor")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    param = DISTRIBUTIONS[dist_name]((p["rows"], p["cols"]), dtype=dtype, device=device)
    grad = DISTRIBUTIONS[dist_name]((p["rows"], p["cols"]), dtype=dtype, device=device)
    exp_avg_sq_row = torch.ones(p["rows"], dtype=dtype, device=device)
    exp_avg_sq_col = torch.ones(p["cols"], dtype=dtype, device=device)
    return [param, grad, exp_avg_sq_row, exp_avg_sq_col, p["step"]]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
