import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Gumbel-Softmax
    
    Used by: Discrete VAE, VQ models
    
    Differentiable discrete sampling using Gumbel-Softmax.
    
    Shapes:
        Input: (batch, num_classes) logits
        Output: (batch, num_classes) soft/hard one-hot
    """
    
    def __init__(self, temperature: float = 1.0, hard: bool = False):
        super(Model, self).__init__()
        self.temperature = temperature
        self.hard = hard
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return F.gumbel_softmax(logits, tau=self.temperature, hard=self.hard)


batch_size = 64
num_classes = 512

def get_inputs():
    logits = torch.randn(batch_size, num_classes, device='cuda')
    return [logits]

def get_init_inputs():
    return [1.0, True]

