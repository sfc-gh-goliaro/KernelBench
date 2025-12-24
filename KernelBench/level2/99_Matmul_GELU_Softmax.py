import torch
import torch.nn as nn


TASK_CONFIG = {
    'comparison_mode': 'log_domain',
    'atol': 1e-5,
    'rtol': 1e-3,
}

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies GELU, and then applies Softmax.
    """
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        x = torch.nn.functional.gelu(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192

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
    return [in_features, out_features]