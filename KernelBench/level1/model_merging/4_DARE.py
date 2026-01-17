import torch
import torch.nn as nn

class Model(nn.Module):
    """
    DARE (Drop And REscale)
    
    Used by: DARE merging
    
    Randomly drop deltas, rescale remaining.
    
    Shapes:
        delta: (param_shape) task vector
        Output: (param_shape) processed
    """
    
    def __init__(self, drop_rate: float = 0.9):
        super(Model, self).__init__()
        self.drop_rate = drop_rate
    
    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        if self.training:
            mask = torch.bernoulli(torch.full_like(delta, 1 - self.drop_rate))
            # Rescale to maintain expected value
            return delta * mask / (1 - self.drop_rate)
        else:
            return delta


param_shape = (4096, 4096)

def get_inputs():
    delta = torch.randn(*param_shape, device='cuda') * 0.1
    return [delta]

def get_init_inputs():
    return [0.9]

