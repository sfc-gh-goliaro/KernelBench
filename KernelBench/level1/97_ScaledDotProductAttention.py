import torch
import torch.nn as nn


TASK_CONFIG = {
    'comparison_mode': 'default',
    'atol': 1e-4,
    'rtol': 1e-4,
}

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        return out

batch_size = 32
num_heads = 32
sequence_length = 512
embedding_dimension = 1024

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    Q = torch.rand(batch_size, num_heads, sequence_length, embedding_dimension, device='cuda', dtype=torch.float16)
    K = torch.rand(batch_size, num_heads, sequence_length, embedding_dimension, device='cuda', dtype=torch.float16)
    V = torch.rand(batch_size, num_heads, sequence_length, embedding_dimension, device='cuda', dtype=torch.float16)
    return [Q, K, V]

def get_init_inputs():
    return []
