import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class QuickGELU(nn.Module):
    """CLIP's fast GELU approximation."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class CLIPAttention(nn.Module):
    """CLIP vision attention."""
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(out)


class CLIPMLP(nn.Module):
    """CLIP MLP with QuickGELU."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.act = QuickGELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Model(nn.Module):
    """
    CLIP Vision Transformer Block
    
    The core repeated block in CLIP vision encoders.
    Used by: CLIP, SigLIP, EVA, OpenCLIP, and vision encoders in VLMs
    
    Architecture (Pre-LN):
        x -> LayerNorm -> Self-Attention -> + residual
          -> LayerNorm -> MLP (QuickGELU) -> + residual
    
    Key features:
    - Pre-LayerNorm architecture
    - QuickGELU activation (faster approximation)
    - Bidirectional attention over image patches
    """
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.attn = CLIPAttention(hidden_size, num_heads)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.mlp = CLIPMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with pre-norm
        x = x + self.attn(self.ln_1(x))
        
        # MLP with pre-norm
        x = x + self.mlp(self.ln_2(x))
        
        return x


# Benchmark configuration (CLIP ViT-L/14 dimensions)
batch_size = 8
num_patches = 257  # 256 patches + 1 CLS token (224/14)^2 + 1
hidden_size = 1024
num_heads = 16
intermediate_size = 4096

def get_inputs():
    return [torch.randn(batch_size, num_patches, hidden_size)]

def get_init_inputs():
    return [hidden_size, num_heads, intermediate_size]

