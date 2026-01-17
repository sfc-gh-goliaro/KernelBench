import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Wide Component (Wide & Deep)
    
    Used by: WDL (Wide & Deep Learning)
    
    Wide linear component with cross-product feature transformations.
    
    Shapes:
        Input: (batch, input_dim)
        Output: (batch, 1)
    """
    
    def __init__(self, input_dim: int):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


batch_size = 4096
input_dim = 1000

def get_inputs():
    x = torch.randn(batch_size, input_dim, device='cuda')
    return [x]

def get_init_inputs():
    return [input_dim]

