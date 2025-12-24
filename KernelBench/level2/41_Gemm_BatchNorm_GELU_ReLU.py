import torch
import torch.nn as nn


TASK_CONFIG = {
    'comparison_mode': 'relative',
    'atol': 1e-5,
    'rtol': 1e-3,
}

class Model(nn.Module):
    """
    Model that performs a GEMM, BatchNorm, GELU, and ReLU in sequence.
    """
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.gemm(x)
        x = self.batch_norm(x)
        x = torch.nn.functional.gelu(x)
        x = torch.relu(x)
        return x

batch_size = 16384
in_features = 4096
out_features = 4096

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