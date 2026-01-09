import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Weight Averaging
    
    Used by: Model soup, ensemble
    
    Simple weight averaging: (w1 + w2 + ... + wn) / n
    
    Shapes:
        weights: list of (param_shape) tensors
        Output: (param_shape) averaged
    """
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, *weights: torch.Tensor) -> torch.Tensor:
        return sum(weights) / len(weights)


param_shape = (4096, 4096)

def get_inputs():
    w1 = torch.randn(*param_shape, device='cuda')
    w2 = torch.randn(*param_shape, device='cuda')
    w3 = torch.randn(*param_shape, device='cuda')
    return [w1, w2, w3]

def get_init_inputs():
    return []

