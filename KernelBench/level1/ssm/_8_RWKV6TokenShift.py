"""
RWKV-6 Token Shift

Used by: RWKV-6

Computes the token-shift delta for a sequence of hidden states:

    delta = shifted(x) - x

where shifted(x) is x shifted one position to the right with the last
position dropped.  Position 0 of the shifted sequence is filled with
either zeros (default) or a caller-supplied previous hidden state
(for cached autoregressive generation).

This is the fundamental mixing signal for RWKV-6's LerpLinear
projections.

Shapes:
    x:           (B, T, D)
    prev_hidden: (B, D)    [optional]
    Output:      (B, T, D)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Token shift delta: shifted(x) - x.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        x: torch.Tensor,
        prev_hidden: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute token-shift delta.

        Args:
            x:           Input tensor (B, T, D)
            prev_hidden: Previous last hidden state (B, D) from cache.
                         If provided, used as position -1 instead of zeros.

        Returns:
            Delta tensor (B, T, D) where delta[t] = x[t-1] - x[t]
            (with x[-1] = prev_hidden if given, else 0).
        """
        shifted = F.pad(x, (0, 0, 1, -1))  # shift right, zero-fill position 0
        if prev_hidden is not None:
            shifted[:, 0] = prev_hidden
        return shifted - x
