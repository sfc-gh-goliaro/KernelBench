import os
import sys
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    AdamW Optimizer Step
    
    Used by: Transformer training (default)
    
    AdamW with decoupled weight decay. Weight decay is applied directly
    to parameters rather than through gradient modification.
    
    Shapes:
        param: any shape
        grad: same shape as param
        Output: updated param
    """
    
    def __init__(self, lr: float = 1e-3, betas: tuple = (0.9, 0.999), 
                 eps: float = 1e-8, weight_decay: float = 0.01):
        """
        Initialize AdamW optimizer step.
        
        Args:
            lr: Learning rate
            betas: Coefficients for running averages
            eps: Term for numerical stability
            weight_decay: Decoupled weight decay coefficient
        """
        super(Model, self).__init__()
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
    
    def forward(self, param: torch.Tensor, grad: torch.Tensor,
                m: torch.Tensor, v: torch.Tensor, step: int) -> tuple:
        """
        Perform one AdamW step.
        
        Args:
            param: Parameters to update
            grad: Gradients
            m: First moment estimate
            v: Second moment estimate
            step: Current step number
            
        Returns:
            Tuple of (updated_param, updated_m, updated_v)
        """
        # Decoupled weight decay (applied directly to params)
        param = param * (1 - self.lr * self.weight_decay)
        
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
