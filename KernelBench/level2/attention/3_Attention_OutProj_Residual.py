import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Fused Attention Output + Residual
    
    Used by: All transformer architectures
    
    Attention output projection fused with residual addition.
    """
    
    def __init__(self, hidden_size: int):
        super(Model, self).__init__()
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
    
    def forward(self, attn_output: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.o_proj(attn_output) + residual


batch_size, seq_len, hidden_size = 8, 2048, 4096

def get_inputs():
    attn_out = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
    residual = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
    return [attn_out, residual]

def get_init_inputs():
    return [hidden_size]

