import torch
import torch.nn as nn


TASK_CONFIG = {
    'comparison_mode': 'relative',
    'atol': 1e-5,
    'rtol': 1e-3,
}

class Model(nn.Module):
    """
    A model that computes Hinge Loss for binary classification tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        return torch.mean(torch.clamp(1 - predictions * targets, min=0))

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
    return [torch.rand(batch_size, *input_shape), torch.randint(0, 2, (batch_size,)).float() * 2 - 1]

def get_init_inputs():
    return []