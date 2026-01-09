import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Fused RMSNorm + Gate/Up Projection
    
    Used by: All modern LLMs
    
    Pre-FFN RMSNorm fused with gate and up projections.
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int, eps: float = 1e-6):
        super(Model, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x / rms * self.weight
        return F.silu(self.gate_proj(x)) * self.up_proj(x)


batch_size, seq_len, hidden_size, intermediate_size = 8, 2048, 4096, 11008

def get_inputs():
    return [torch.randn(batch_size, seq_len, hidden_size, device='cuda')]

def get_init_inputs():
    return [hidden_size, intermediate_size]

