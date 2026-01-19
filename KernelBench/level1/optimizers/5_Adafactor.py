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
    
    Memory-efficient optimizer using factored second moment estimates.
    Reduces memory compared to Adam for large matrices.
    
    Shapes:
        param: (rows, cols) matrix
        grad: same shape as param
        Output: updated param
    """
    
    def __init__(self, lr: float = None, eps: tuple = (1e-30, 1e-3),
                 clip_threshold: float = 1.0, decay_rate: float = -0.8,
                 beta1: float = None, weight_decay: float = 0.0):
        """
        Initialize Adafactor optimizer step.
        
        Args:
            lr: Learning rate (None for automatic scaling)
            eps: Regularization constants (eps1, eps2)
            clip_threshold: Threshold for gradient clipping
            decay_rate: Decay rate for second moment
            beta1: Momentum coefficient (None for no momentum)
            weight_decay: Weight decay coefficient
        """
        super(Model, self).__init__()
        self.lr = lr
        self.eps1, self.eps2 = eps
        self.clip_threshold = clip_threshold
        self.decay_rate = decay_rate
        self.beta1 = beta1
        self.weight_decay = weight_decay
    
    def _rms(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute root mean square."""
        return tensor.pow(2).mean().sqrt()
    
    def forward(self, param: torch.Tensor, grad: torch.Tensor,
                exp_avg_sq_row: torch.Tensor, exp_avg_sq_col: torch.Tensor,
                step: int) -> tuple:
        """
        Perform one Adafactor step.
        
        Args:
            param: Parameters to update (2D matrix)
            grad: Gradients
            exp_avg_sq_row: Row factor of second moment
            exp_avg_sq_col: Column factor of second moment
            step: Current step number
            
        Returns:
            Tuple of (updated_param, updated_row, updated_col)
        """
        # Compute decay
        rho = min(self.decay_rate, -(step ** -0.8))
        rho = 1.0 - (1.0 + rho)
        
        # Update factored second moment estimates
        grad_sqr = grad.pow(2) + self.eps1
        
        # Row and column means
        row_mean = grad_sqr.mean(dim=-1)
        col_mean = grad_sqr.mean(dim=-2)
        
        # Update running averages
        exp_avg_sq_row = rho * exp_avg_sq_row + (1 - rho) * row_mean
        exp_avg_sq_col = rho * exp_avg_sq_col + (1 - rho) * col_mean
        
        # Reconstruct full second moment
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean()).unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2)
        v = r_factor * c_factor
        
        # Compute update
        update = grad / v.sqrt()
        
        # Gradient clipping
        rms = self._rms(update)
        if rms > self.clip_threshold:
            update = update * (self.clip_threshold / rms)
        
        # Learning rate
        if self.lr is None:
            lr = max(self.eps2, self._rms(param)) * (step ** -0.5)
        else:
            lr = self.lr
        
        # Weight decay
        if self.weight_decay != 0:

PARAMETERS = [
    {"param_size": (4096, 4096)},
    # T5-XXL training: encoder-decoder attention weight
    {"param_size": (4096, 1024)},
    # PaLM-540B training: memory-efficient MLP layer
    {"param_size": (8192, 32768)},
    # mT5-large training: cross-attention projection
    {"param_size": (1024, 1024)},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("optimizers", "5_Adafactor")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    param = DISTRIBUTIONS[dist_name]((rows, cols), dtype=dtype, device=device)
    grad = DISTRIBUTIONS[dist_name]((rows, cols), dtype=dtype, device=device)
    exp_avg_sq_row = DISTRIBUTIONS[dist_name]((rows), dtype=dtype, device=device)
    exp_avg_sq_col = DISTRIBUTIONS[dist_name]((cols), dtype=dtype, device=device)
    return [param, grad, exp_avg_sq_row, exp_avg_sq_col, step]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
        
        return param, exp_avg_sq_row, exp_avg_sq_col


# ============================================================================
# Benchmark Configuration
# ============================================================================

rows, cols = 4096, 4096
