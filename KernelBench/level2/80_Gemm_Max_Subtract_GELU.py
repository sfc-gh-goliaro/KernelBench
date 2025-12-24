import torch
import torch.nn as nn


TASK_CONFIG = {
    'comparison_mode': 'default',
    'atol': 1e-4,
    'rtol': 1e-4,
}

class Model(nn.Module):
    """
    Model that performs a GEMM, followed by a max operation, subtraction, and GELU activation.
    """
    def __init__(self, in_features, out_features, max_dim):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        x = self.gemm(x)
        x = torch.max(x, dim=self.max_dim, keepdim=True).values
        x = x - x.mean(dim=1, keepdim=True)
        x = torch.nn.functional.gelu(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
max_dim = 1

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
    return [in_features, out_features, max_dim]