import torch
import torch.nn as nn


TASK_CONFIG = {
    'comparison_mode': 'default',
    'atol': 1e-4,
    'rtol': 1e-4,
}

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies ReLU, and divides by a constant.
    """
    def __init__(self, in_features, out_features, divisor):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.divisor = divisor

    def forward(self, x):
        x = self.linear(x)
        x = torch.relu(x)
        x = x / self.divisor
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
divisor = 2.0

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, divisor]