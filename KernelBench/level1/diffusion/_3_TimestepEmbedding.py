import torch
import torch.nn as nn
from typing import Optional


class Model(nn.Module):
    """
    Timestep Embedding MLP

    Used by: All diffusion models (SDXL, SD-3/3.5, FLUX, DiT, PixArt)

    Maps pre-computed sinusoidal timestep encodings (or any conditioning
    vector) to dense embeddings via Linear + activation + Linear.

    This is the MLP portion only — sinusoidal encoding is handled by the
    separate SinusoidalTimesteps level1 op.

    Matches diffusers TimestepEmbedding exactly.

    State-dict key layout (for HuggingFace weight compatibility):
        linear_1 — nn.Linear(in_channels, time_embed_dim)
        act      — activation function (SiLU by default)
        linear_2 — nn.Linear(time_embed_dim, out_dim)
        cond_proj — optional nn.Linear(cond_proj_dim, in_channels)
        post_act  — optional post-activation

    Shapes:
        Input: (batch, in_channels)
        Output: (batch, out_dim)
    """

    def __init__(self, in_channels: int, time_embed_dim: int,
                 act_fn: str = "silu", out_dim: Optional[int] = None,
                 post_act_fn: Optional[str] = None,
                 cond_proj_dim: Optional[int] = None,
                 bias: bool = True):
        """
        Initialize timestep embedding MLP.

        Args:
            in_channels: Input dimension (e.g. 256 for sinusoidal encoding dim)
            time_embed_dim: Hidden/output embedding dimension
            act_fn: Activation function ('silu', 'mish', 'gelu')
            out_dim: Output dimension (defaults to time_embed_dim)
            post_act_fn: Optional post-activation after linear_2
            cond_proj_dim: If set, adds a conditioning projection
            bias: Whether linear layers have bias
        """
        super(Model, self).__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=bias)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        if act_fn == "silu":
            self.act = nn.SiLU()
        elif act_fn == "mish":
            self.act = nn.Mish()
        elif act_fn == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.SiLU()

        time_embed_dim_out = out_dim if out_dim is not None else time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, bias=bias)

        if post_act_fn is not None:
            if post_act_fn == "silu":
                self.post_act = nn.SiLU()
            else:
                self.post_act = None
        else:
            self.post_act = None

    def forward(self, sample: torch.Tensor,
                condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Embed timestep encodings.

        Args:
            sample: Pre-computed sinusoidal encoding (batch, in_channels)
            condition: Optional conditioning tensor (batch, cond_proj_dim)

        Returns:
            Timestep embeddings (batch, out_dim)
        """
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)
        if self.act is not None:
            sample = self.act(sample)
        sample = self.linear_2(sample)
        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample
