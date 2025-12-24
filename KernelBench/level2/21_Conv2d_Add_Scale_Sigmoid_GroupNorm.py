import torch
import torch.nn as nn


TASK_CONFIG = {
    'comparison_mode': 'relative',
    'atol': 1e-5,
    'rtol': 1e-3,
}

class Model(nn.Module):
    """
    Model that performs a convolution, adds a bias term, scales, applies sigmoid, and performs group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = x + self.bias
        x = x * self.scale
        x = torch.sigmoid(x)
        x = self.group_norm(x)
        return x

batch_size = 128
in_channels = 8
out_channels = 32
height = width = 256
kernel_size = 3
num_groups = 8
bias_shape = (out_channels, 1, 1)
scale_shape = (out_channels, 1, 1)

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
    return [in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape]