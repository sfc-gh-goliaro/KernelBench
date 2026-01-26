import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Fused QKV Projection + RoPE
    
    Used by: Llama, Qwen, Mistral, Gemma
    
    Fused Q/K/V projection with rotary position embedding applied to Q and K.
    """
    
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, max_seq_len: int = 8192):
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        self.qkv_proj = nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=False)
        
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)
        
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor = None) -> tuple:
        batch_size, seq_len, _ = x.shape
        
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(2)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(2)
        
        def apply_rope(x):
            x1, x2 = x[..., :self.head_dim//2], x[..., self.head_dim//2:]
            return x * cos + torch.cat((-x2, x1), dim=-1) * sin
        
        q, k = apply_rope(q), apply_rope(k)
        return q, k, v


batch_size, seq_len, hidden_size, num_heads, head_dim = 8, 2048, 4096, 32, 128

def get_inputs():
    return [torch.randn(batch_size, seq_len, hidden_size, device='cuda')]

def get_init_inputs():
    return [hidden_size, num_heads, head_dim]

