import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    """SD UNet spatial self-attention."""
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.to_qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.to_out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) for t in qkv]
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.to_out(out)


class CrossAttention(nn.Module):
    """SD UNet cross-attention to text conditioning."""
    def __init__(self, hidden_size: int, context_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.to_q = nn.Linear(hidden_size, hidden_size)
        self.to_k = nn.Linear(context_dim, hidden_size)
        self.to_v = nn.Linear(context_dim, hidden_size)
        self.to_out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        context_len = context.shape[1]
        
        q = self.to_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(batch_size, context_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(batch_size, context_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.to_out(out)


class FeedForward(nn.Module):
    """SD UNet feed-forward with GEGLU."""
    def __init__(self, hidden_size: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = hidden_size * mult
        # GEGLU splits inner_dim for gate
        self.net = nn.Sequential(
            nn.Linear(hidden_size, inner_dim * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, hidden_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GEGLU(nn.Module):
    """Gated GELU activation."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class Model(nn.Module):
    """
    Stable Diffusion Cross-Attention Block
    
    The core attention block in UNet-based diffusion models.
    Used by: Stable Diffusion 1.5, 2.x, SDXL (transformer blocks in UNet)
    
    Architecture:
        x -> LayerNorm -> Self-Attention -> + residual
          -> LayerNorm -> Cross-Attention (to text) -> + residual
          -> LayerNorm -> FeedForward (GEGLU) -> + residual
    
    Key features:
    - Spatial self-attention over image features
    - Cross-attention to text embeddings (CLIP)
    - GEGLU activation in feed-forward
    """
    def __init__(self, hidden_size: int, context_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        
        # Self-attention
        self.norm1 = nn.LayerNorm(hidden_size)
        self.self_attn = SelfAttention(hidden_size, num_heads)
        
        # Cross-attention
        self.norm2 = nn.LayerNorm(hidden_size)
        self.cross_attn = CrossAttention(hidden_size, context_dim, num_heads)
        
        # Feed-forward
        self.norm3 = nn.LayerNorm(hidden_size)
        self.ff = FeedForward(hidden_size, dropout=dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Spatial features (batch, h*w, hidden_size)
            context: Text embeddings (batch, seq_len, context_dim)
        """
        # Self-attention
        x = x + self.self_attn(self.norm1(x))
        
        # Cross-attention to text
        x = x + self.cross_attn(self.norm2(x), context)
        
        # Feed-forward
        x = x + self.ff(self.norm3(x))
        
        return x


# Benchmark configuration (SDXL dimensions)
batch_size = 2
spatial_size = 64 * 64  # 64x64 latent
hidden_size = 640
context_dim = 2048  # CLIP embedding dim
text_seq_len = 77
num_heads = 10

def get_inputs():
    x = torch.randn(batch_size, spatial_size, hidden_size)
    context = torch.randn(batch_size, text_seq_len, context_dim)
    return [x, context]

def get_init_inputs():
    return [hidden_size, context_dim, num_heads]

