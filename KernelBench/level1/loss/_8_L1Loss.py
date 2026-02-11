import os
import sys
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    L1 Loss (Mean Absolute Error)
    
    Used by: Image reconstruction, super-resolution, style transfer
    
    Computes the mean absolute error between predictions and targets.
    L1 loss is more robust to outliers compared to L2 (MSE) loss.
    
    Shapes:
        Input prediction: any shape
        Input target: same shape as prediction
        Output: scalar loss (or per-element if reduction='none')
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize L1 Loss.
        
        Args:
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(Model, self).__init__()
        self.loss_fn = nn.L1Loss(reduction=reduction)
    
    def forward(self, prediction: torch.Tensor, 
                target: torch.Tensor) -> torch.Tensor:
        """
        Compute L1 loss.
        
        Args:
            prediction: Predicted values (any shape)
            target: Target values (same shape as prediction)
            
        Returns:
            L1 loss value
        """
        return self.loss_fn(prediction, target)
