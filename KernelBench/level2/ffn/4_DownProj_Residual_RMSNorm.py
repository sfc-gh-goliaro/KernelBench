import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Fused Down Projection + Residual + RMSNorm
    
    Used by: LLMs with cross-layer optimization
    
    Down projection + residual + next layer's RMSNorm.
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int, eps: float = 1e-6):
        super(Model, self).__init__()
        self.eps = eps
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.norm_weight = nn.Parameter(torch.ones(hidden_size))
    
    def forward(self, ffn_out: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        x = self.down_proj(ffn_out) + residual
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.norm_weight


batch_size, seq_len, hidden_size, intermediate_size = 8, 2048, 4096, 11008

def get_inputs():
    ffn_out = torch.randn(batch_size, seq_len, intermediate_size, device='cuda')
    residual = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
    return [ffn_out, residual]

def get_init_inputs():
    return [hidden_size, intermediate_size]

