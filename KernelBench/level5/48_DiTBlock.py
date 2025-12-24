import torch
import torch.nn as nn
import torch.nn.functional as F
import math


TASK_CONFIG = {
    'comparison_mode': 'default',
    'atol': 1e-4,
    'rtol': 1e-4,
}

class Model(nn.Module):
    """
    Diffusion Transformer (DiT) Block.
    
    Transformer block with AdaLN-Zero conditioning for diffusion models.
    Used in FLUX.1 and similar image generation models.
    
    Based on: "Scalable Diffusion Models with Transformers"
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True):
        """
        :param dim: Hidden dimension
        :param num_heads: Number of attention heads
        :param mlp_ratio: MLP hidden dim ratio
        :param qkv_bias: Whether to use bias in QKV projection
        """
        super(Model, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Layer norms (no learnable params - handled by AdaLN)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        
        # Attention
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
        # MLP
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_hidden, dim)
        )
        
        # AdaLN modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim)
        )
        
        # Zero-init the modulation
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(self, x, cond):
        """
        Forward pass for DiT block.
        
        :param x: Input tensor (batch, seq_len, dim)
        :param cond: Conditioning tensor (batch, dim) - timestep embedding
        :return: Output tensor (batch, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Get modulation parameters
        modulation = self.adaLN_modulation(cond)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            modulation.unsqueeze(1).chunk(6, dim=-1)
        
        # Self-attention with AdaLN
        x_norm = self.norm1(x)
        x_mod = x_norm * (1 + scale_msa) + shift_msa
        
        # QKV projection
        qkv = self.qkv(x_mod)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        out = self.proj(out)
        
        # Gated residual
        x = x + gate_msa * out
        
        # MLP with AdaLN
        x_norm = self.norm2(x)
        x_mod = x_norm * (1 + scale_mlp) + shift_mlp
        mlp_out = self.mlp(x_mod)
        
        # Gated residual
        x = x + gate_mlp * mlp_out
        
        return x


# Test parameters
batch_size = 8
seq_len = 256  # 16x16 patches
dim = 1152     # DiT-XL
num_heads = 16

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    x = torch.randn(batch_size, seq_len, dim)
    cond = torch.randn(batch_size, dim)  # Timestep + class embedding
    return [x, cond]

def get_init_inputs():
    return [dim, num_heads]

