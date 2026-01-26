import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DETRSelfAttention(nn.Module):
    """DETR encoder self-attention."""
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pos_embed: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Add position embedding to Q and K
        q = k = x
        if pos_embed is not None:
            q = q + pos_embed
            k = k + pos_embed
        
        q = self.q_proj(q).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(out)


class DETRFFN(nn.Module):
    """DETR feed-forward network."""
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class Model(nn.Module):
    """
    DETR Encoder Layer
    
    The core repeated block in DETR encoder.
    Used by: DETR, Deformable DETR, DINO-DETR, DAB-DETR
    
    Architecture (Post-LN):
        x -> Self-Attention (+ pos) -> Dropout -> + residual -> LayerNorm
          -> FFN -> Dropout -> + residual -> LayerNorm
    
    Key features:
    - Position embedding added to Q and K
    - Post-LayerNorm architecture
    - Standard transformer encoder structure
    """
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = DETRSelfAttention(hidden_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = DETRFFN(hidden_size, intermediate_size, dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pos_embed: torch.Tensor = None) -> torch.Tensor:
        # Self-attention with post-norm
        residual = x
        x = self.self_attn(x, pos_embed)
        x = self.dropout(x)
        x = self.norm1(residual + x)
        
        # FFN with post-norm
        residual = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.norm2(residual + x)
        
        return x


# Benchmark configuration (DETR-R50)
batch_size = 2
seq_len = 850  # ~29x29 feature map
hidden_size = 256
num_heads = 8
intermediate_size = 2048
dropout = 0.0

def get_inputs():
    x = torch.randn(batch_size, seq_len, hidden_size)
    pos_embed = torch.randn(batch_size, seq_len, hidden_size)
    return [x, pos_embed]

def get_init_inputs():
    return [hidden_size, num_heads, intermediate_size, dropout]

