import torch
import torch.nn as nn


TASK_CONFIG = {
    'comparison_mode': 'default',
    'atol': 1e-4,
    'rtol': 1e-4,
}

class Model(nn.Module):
    """
    Model that performs a sequence of operations:
        - Matrix multiplication
        - Summation
        - Max
        - Average pooling
        - LogSumExp
        - LogSumExp
    """
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        x = self.linear(x)  # (batch_size, out_features)
        x = torch.sum(x, dim=1, keepdim=True) # (batch_size, 1)
        x = torch.max(x, dim=1, keepdim=True)[0] # (batch_size, 1)
        x = torch.mean(x, dim=1, keepdim=True) # (batch_size, 1)
        x = torch.logsumexp(x, dim=1, keepdim=True) # (batch_size, 1)
        x = torch.logsumexp(x, dim=1, keepdim=True) # (batch_size, 1)
        return x

batch_size = 1024
in_features  = 8192  
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