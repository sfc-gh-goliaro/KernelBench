import torch
import torch.nn as nn
from typing import Optional


class Model(nn.Module):
    """
    Adaptive Layer Normalization (AdaLN-Continuous)

    Used by: SD-3/3.5 (AdaLayerNormContinuous), FLUX, PixArt, DiT

    Adaptive LayerNorm where scale and shift are predicted from a conditioning
    embedding via SiLU + Linear. The conditioning is projected to
    2 * hidden_size, then split into (scale, shift) and applied as:
        output = LayerNorm(x) * (1 + scale) + shift

    Matches diffusers AdaLayerNormContinuous exactly when using default
    norm_type='layer_norm'.

    State-dict key layout (for HuggingFace weight compatibility):
        silu   — parameter-free SiLU activation
        linear — nn.Linear(cond_dim, 2 * hidden_size)
        norm   — nn.LayerNorm(hidden_size)

    Shapes:
        x: (batch, seq_len, hidden_size)
        conditioning: (batch, cond_dim)
        Output: (batch, seq_len, hidden_size)
    """

    def __init__(self, hidden_size: int, cond_dim: int,
                 elementwise_affine: bool = False, eps: float = 1e-6,
                 bias: bool = True):
        """
        Initialize AdaLN.

        Args:
            hidden_size: Hidden dimension
            cond_dim: Conditioning embedding dimension
            elementwise_affine: Whether LayerNorm has learnable affine params
            eps: LayerNorm epsilon
            bias: Whether the Linear projection has bias
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size

        self.silu = nn.SiLU()
        self.linear = nn.Linear(cond_dim, 2 * hidden_size, bias=bias)
        self.norm = nn.LayerNorm(hidden_size, eps=eps,
                                 elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive layer normalization.

        Args:
            x: Input tensor (batch, seq_len, hidden_size)
            conditioning: Conditioning embedding (batch, cond_dim)

        Returns:
            Normalized and modulated tensor (batch, seq_len, hidden_size)
        """
        emb = self.linear(self.silu(conditioning).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=-1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x
