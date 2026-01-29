import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Self-Attention over Features (AutoInt)
    
    Used by: AutoInt
    
    Self-attention for automatic feature interaction learning.
    
    Shapes:
        Input: (batch, num_features, embed_dim)
        Output: (batch, num_features, embed_dim)
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 2):
        super(Model, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)
