import torch
import torch.nn as nn


TASK_CONFIG = {
    'comparison_mode': 'default',
    'atol': 1e-4,
    'rtol': 1e-4,
}

class Model(nn.Module):
    """
    A model that performs a reverse cumulative sum operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the reverse cumulative sum.
    """

    def __init__(self, dim):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cumsum(x.flip(self.dim), dim=self.dim).flip(self.dim)

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    return [torch.rand(batch_size, *input_shape)]

def get_init_inputs():
    return [dim]
