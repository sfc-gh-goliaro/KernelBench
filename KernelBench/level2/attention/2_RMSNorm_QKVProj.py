import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Fused RMSNorm + QKV Projection
    
    Used by: All modern LLMs with pre-norm
    
    Pre-attention RMSNorm fused with QKV linear projection.
    """
    
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, eps: float = 1e-6):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.qkv_proj = nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x / rms * self.weight
        # QKV projection
        return self.qkv_proj(x)


batch_size, seq_len, hidden_size, num_heads, head_dim = 8, 2048, 4096, 32, 128

def get_inputs():
    return [torch.randn(batch_size, seq_len, hidden_size, device='cuda')]

def get_init_inputs():
    return [hidden_size, num_heads, head_dim]

