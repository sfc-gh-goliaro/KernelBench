"""
RWKV-6 LerpLinear (Lerp Mixing + Linear Projection)

Used by: RWKV-6

Applies a linear projection after mixing the input with a token-shift
interpolation:

    y = linear( x + delta * mu )

where delta = token_shift(x) is the shift difference and mu is either:
  - a learnable per-feature parameter (learnable_mu=True, default), or
  - a data-dependent vector passed in by the caller (learnable_mu=False).

The linear projection is either a plain nn.Linear (bias-free) or a
low-rank LoRA factorisation.

LoRA factorisation (when low_rank_dim is provided):
    y = Linear_up( act( Linear_down( mixed_x ) ) )
where Linear_down is bias-free and Linear_up has optional bias.

Shapes:
    x:      (B, T, input_dim)
    delta:  (B, T, input_dim)
    mu:     (B, T, input_dim)  [only when learnable_mu=False]
    Output: (B, T, output_dim)
"""

import torch
import torch.nn as nn


class _LoRA(nn.Module):
    """
    Low-rank linear factorisation: down -> activation -> up.

    Wraps an nn.Sequential in a `.lora` attribute so that the state dict
    keys match the fla library's LoRA layout (e.g. `linear.lora.0.weight`).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        low_rank_dim: int,
        bias: bool = True,
        activation: str = "tanh",
    ):
        super().__init__()

        if activation == "tanh":
            act = nn.Tanh()
        elif activation == "sigmoid":
            act = nn.Sigmoid()
        elif activation == "relu":
            act = nn.ReLU()
        elif activation is None:
            act = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.lora = nn.Sequential(
            nn.Linear(input_dim, low_rank_dim, bias=False),
            act,
            nn.Linear(low_rank_dim, output_dim, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora(x)


class Model(nn.Module):
    """
    Linear with lerp-based time-shift mixing.

    When learnable_mu=True (default):
        Computes: linear(x + delta * self.mu)
        forward(x, delta) — mu is an nn.Parameter

    When learnable_mu=False:
        Computes: linear(x + delta * mu)
        forward(x, delta, mu=mu) — mu is passed in (data-dependent)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        low_rank_dim: int = None,
        learnable_mu: bool = True,
        lora_bias: bool = True,
        lora_activation: str = "tanh",
    ):
        """
        Args:
            input_dim:       Input feature dimension.
            output_dim:      Output feature dimension.
            low_rank_dim:    If provided, use a LoRA factorisation for the
                             linear projection.  Otherwise use a plain
                             nn.Linear (bias=False).
            learnable_mu:    If True, mu is a learnable nn.Parameter of shape
                             (input_dim,).  If False, mu must be passed as
                             an argument to forward().
            lora_bias:       Whether the LoRA up-projection has a bias term.
                             Only used when low_rank_dim is not None.
            lora_activation: Activation between LoRA down/up projections.
                             One of 'tanh', 'sigmoid', 'relu', or None.
                             Only used when low_rank_dim is not None.
        """
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.low_rank_dim = low_rank_dim
        self.learnable_mu = learnable_mu

        # Linear projection (plain or LoRA)
        if low_rank_dim is None:
            self.linear = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.linear = _LoRA(
                input_dim, output_dim, low_rank_dim,
                bias=lora_bias,
                activation=lora_activation,
            )

        # Learnable interpolation weight (only when not data-dependent)
        if learnable_mu:
            self.mu = nn.Parameter(torch.zeros(input_dim))

    def forward(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        mu: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Apply lerp mixing and project.

        Args:
            x:     Input hidden states (B, T, input_dim)
            delta: Token-shift delta (B, T, input_dim)
            mu:    Data-dependent modulation (B, T, input_dim).
                   Required when learnable_mu=False, ignored otherwise.

        Returns:
            Projected output (B, T, output_dim)
        """
        if self.learnable_mu:
            mu = self.mu
        return self.linear(x + delta * mu)
