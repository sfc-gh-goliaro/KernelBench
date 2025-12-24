import torch
import torch.nn as nn


TASK_CONFIG = {
    'comparison_mode': 'default',
    'atol': 1e-4,
    'rtol': 1e-4,
}

class Model(nn.Module):
    """
    Model that performs a 3D convolution, applies ReLU, LeakyReLU, GELU, Sigmoid activations, and bias in sequence.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        x = torch.nn.functional.gelu(x)
        x = torch.sigmoid(x)
        x = x + self.bias
        return x

batch_size = 64
in_channels = 8
out_channels = 32
depth, height, width = 32, 64, 64
kernel_size = 3
bias_shape = (out_channels, 1, 1, 1)

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]