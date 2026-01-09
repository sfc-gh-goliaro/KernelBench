import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    SwiGLU (SiLU-Gated Linear Unit)
    
    Used by: Llama, Qwen, Mistral, Gemma, Yi
    
    SiLU(gate) × up projection for FFN.
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super(Model, self).__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.gate_proj(x)) * self.up_proj(x)


batch_size, seq_len, hidden_size, intermediate_size = 8, 2048, 4096, 11008

def get_inputs():
    return [torch.randn(batch_size, seq_len, hidden_size, device='cuda')]

def get_init_inputs():
    return [hidden_size, intermediate_size]

