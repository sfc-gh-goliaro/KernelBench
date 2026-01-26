import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Fused AdaLN + Attention
    
    Used by: DiT, SD-3, FLUX
    
    Adaptive LayerNorm (timestep-conditioned) fused with attention.
    """
    
    def __init__(self, hidden_size: int, num_heads: int, cond_dim: int):
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.modulation = nn.Linear(cond_dim, 2 * hidden_size)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        
        shift, scale = self.modulation(cond).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = F.scaled_dot_product_attention(q, k, v)
        return self.proj(attn.transpose(1, 2).reshape(batch, seq_len, -1))


batch_size, seq_len, hidden_size, num_heads, cond_dim = 8, 1024, 1152, 16, 1152

def get_inputs():
    x = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
    cond = torch.randn(batch_size, cond_dim, device='cuda')
    return [x, cond]

def get_init_inputs():
    return [hidden_size, num_heads, cond_dim]

