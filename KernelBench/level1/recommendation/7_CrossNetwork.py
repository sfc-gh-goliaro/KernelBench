import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Cross Network Layer (DCN)
    
    Used by: DCN (Deep & Cross Network)
    
    Cross layer: x_0 * x_l^T * w + b + x_l for explicit crossing.
    
    Shapes:
        Input: (batch, input_dim)
        Output: (batch, input_dim)
    """
    
    def __init__(self, input_dim: int, num_layers: int = 3):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(input_dim, 1) * 0.01) for _ in range(num_layers)])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        xl = x
        for w, b in zip(self.weights, self.biases):
            # x_l+1 = x_0 * (x_l^T * w) + b + x_l
            xl = x0 * (xl @ w) + b + xl
        return xl


batch_size = 4096
input_dim = 416

def get_inputs():
    x = torch.randn(batch_size, input_dim, device='cuda')
    return [x]

def get_init_inputs():
    return [input_dim]

