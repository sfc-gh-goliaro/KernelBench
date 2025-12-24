import torch
import torch.nn as nn


TASK_CONFIG = {
    'comparison_mode': 'relative',
    'atol': 1e-5,
    'rtol': 1e-3,
}

class Model(nn.Module):
    """
    Simple model that performs a convolution, applies Batch Normalization, and scales the output.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = x * self.scaling_factor
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
scaling_factor = 2.0

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor]