import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization with learnable scale and shift from conditioning."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTAttention(nn.Module):
    """DiT self-attention."""
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.proj(out)


class DiTMLP(nn.Module):
    """DiT MLP with GELU."""
    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x), approximate='tanh')
        x = self.dropout(x)
        return self.fc2(x)


class Model(nn.Module):
    """
    DiT (Diffusion Transformer) Block
    
    The core repeated block in DiT-style diffusion transformers.
    Used by: DiT, SD-3, FLUX, PixArt, HunyuanDiT
    
    Architecture (AdaLN-Zero):
        Condition -> MLP -> [scale1, shift1, gate1, scale2, shift2, gate2]
        
        x -> AdaLN(scale1, shift1) -> Attention -> * gate1 -> + residual
          -> AdaLN(scale2, shift2) -> MLP -> * gate2 -> + residual
    
    Key features:
    - AdaLN-Zero: condition modulates layernorm + gated output
    - Zero-initialized gates (starts as identity)
    - Timestep conditioning via MLP
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Attention
        self.norm1 = AdaLayerNorm(hidden_size)
        self.attn = DiTAttention(hidden_size, num_heads, dropout)
        
        # MLP
        self.norm2 = AdaLayerNorm(hidden_size)
        self.mlp = DiTMLP(hidden_size, mlp_ratio, dropout)
        
        # AdaLN-Zero: condition -> 6 modulation parameters
        # [scale1, shift1, gate1, scale2, shift2, gate2]
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size)
        )
        
        # Initialize gates to zero (identity at start)
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, num_patches, hidden_size)
            conditioning: Conditioning tensor (batch, hidden_size) - e.g., timestep embedding
        """
        # Get modulation parameters
        modulation = self.adaLN_modulation(conditioning)
        shift1, scale1, gate1, shift2, scale2, gate2 = modulation.chunk(6, dim=-1)
        
        # Attention block with AdaLN-Zero
        x = x + gate1.unsqueeze(1) * self.attn(self.norm1(x, scale1, shift1))
        
        # MLP block with AdaLN-Zero
        x = x + gate2.unsqueeze(1) * self.mlp(self.norm2(x, scale2, shift2))
        
        return x


# Benchmark configuration (DiT-XL/2 dimensions)
batch_size = 4
num_patches = 256  # 16x16 patches for 256x256 image at patch_size=16
hidden_size = 1152
num_heads = 16
mlp_ratio = 4.0

def get_inputs():
    x = torch.randn(batch_size, num_patches, hidden_size)
    conditioning = torch.randn(batch_size, hidden_size)  # Timestep embedding
    return [x, conditioning]

def get_init_inputs():
    return [hidden_size, num_heads, mlp_ratio]

