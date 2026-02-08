"""
RWKV-6 Naive Recurrent Linear Attention

Used by: RWKV-6

Naive (sequential) recurrent implementation of the RWKV-6 linear attention
mechanism.  The recurrence at each time step is:

    o_i = sum_d ( (h + u * k_i * v_i) * q_i )
    h   = h * exp(w_i) + k_i * v_i

where q (receptance), k, v, w (log-decay) are all per-head per-timestep
tensors and u (bonus) is a per-head parameter that up-weights the
current-token contribution.

Shapes:
    q, k:       (B, H, T, K)
    v:          (B, H, T, V)
    w:          (B, H, T, K)   – negative log-decay values
    u:          (H, K)         – bonus parameter
    initial_state: (B, H, K, V) or None
    Output:     (B, H, T, V)
    final_state: (B, H, K, V) or None
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Optional, Tuple


class Model(nn.Module):
    """
    RWKV-6 naive recurrent linear attention.

    Matches fla's naive_recurrent_rwkv6 exactly.
    This is a pure-PyTorch reference; production kernels would fuse the loop.
    """

    def __init__(self, scale: float = 1.0):
        """
        Args:
            scale: Scaling factor applied to q at each time step.
        """
        super(Model, self).__init__()
        self.scale = scale

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        RWKV-6 naive recurrent forward pass.

        Args:
            q: Receptance tensor (B, H, T, K)
            k: Key tensor (B, H, T, K)
            v: Value tensor (B, H, T, V)
            w: Log-decay tensor (B, H, T, K) – negative values
            u: Bonus parameter (H, K)
            initial_state: Optional initial hidden state (B, H, K, V)
            output_final_state: Whether to return the final hidden state

        Returns:
            Tuple of (output, final_state) where:
                output: (B, H, T, V)
                final_state: (B, H, K, V) or None
        """
        orig_dtype = q.dtype
        B, H, T, K = q.shape
        V = v.shape[-1]
        q, k, v, w, u = (x.float() for x in (q, k, v, w, u))
        h = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
        o = torch.zeros_like(v)

        if initial_state is not None:
            h = h + initial_state.float()

        for i in range(T):
            q_i = q[:, :, i, :] * self.scale
            k_i = k[:, :, i]
            v_i = v[:, :, i, :]
            w_i = w[:, :, i].exp()
            kv_i = k_i[..., None] * v_i[..., None, :]
            o_i = (h + u[None, ..., None] * kv_i) * q_i[..., None]
            o[:, :, i] = o_i.sum(-2)
            h = h * w_i[..., None] + kv_i

        ht = h if output_final_state else None
        return o.to(orig_dtype), ht
