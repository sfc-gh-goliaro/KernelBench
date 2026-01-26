import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Fused Score Head (Reward Model)
    
    Used by: Reward models
    
    Last token pooling + Linear + ReLU + Linear for scalar output.
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int = None):
        super(Model, self).__init__()
        intermediate_size = intermediate_size or hidden_size
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, 1)
    
    def forward(self, hidden_states: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        last_token = hidden_states[torch.arange(batch_size), input_lengths - 1]
        return self.fc2(F.relu(self.fc1(last_token)))


batch_size, seq_len, hidden_size = 8, 512, 4096

def get_inputs():
    hidden = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
    lengths = torch.randint(10, seq_len, (batch_size,), device='cuda')
    return [hidden, lengths]

def get_init_inputs():
    return [hidden_size]

