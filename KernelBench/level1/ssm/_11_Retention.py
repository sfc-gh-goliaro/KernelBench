"""
RetNet Retention Naive Recurrent

Used by: RetNet

Naive (sequential) recurrent implementation of the RetNet retention
mechanism.  The recurrence at each time step is:

    h_t = h_{t-1} * gamma + k_t^T @ v_t
    o_t = q_t @ h_t

where q, k, v are per-head per-timestep tensors and gamma is a per-head
fixed decay factor.  In the standard multi-scale retention formulation:

    gamma_h = 1 - 2^{-(5 + h)}   for head index h

so different heads learn at different retention scales.

Shapes:
    q:       (B, T, H, K)
    k:       (B, T, H, K)
    v:       (B, T, H, V)
    initial_state: (B, H, K, V) or None
    Output:  (B, T, H, V)
    final_state: (B, H, K, V) or None
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class Model(nn.Module):
    """
    RetNet naive recurrent retention.

    Matches fla's fused_recurrent_retention semantics exactly.
    This is a pure-PyTorch reference; production kernels would fuse the loop.
    """

    def __init__(self, num_heads: int, scale: float = None):
        """
        Args:
            num_heads: Number of attention heads (used to compute per-head
                       decay rates gamma).
            scale: Scaling factor applied to q at each time step.
                   If None, defaults to K^{-0.5}.
        """
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.scale = scale

        # Per-head decay: gamma_h = 1 - 2^{-(5+h)}
        # Pre-compute both log_gamma and gamma to avoid repeated exp() calls.
        gamma = 1 - torch.tensor(2.0).pow(
            -5.0 - torch.arange(num_heads, dtype=torch.float32)
        )
        self.register_buffer("log_gamma", gamma.log())  # (H,)
        self.register_buffer("gamma", gamma)             # (H,)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Retention naive recurrent forward pass.

        Args:
            q: Query tensor (B, T, H, K)
            k: Key tensor (B, T, H, K)
            v: Value tensor (B, T, H, V)
            initial_state: Optional initial hidden state (B, H, K, V)
            output_final_state: Whether to return the final hidden state

        Returns:
            Tuple of (output, final_state) where:
                output: (B, T, H, V)
                final_state: (B, H, K, V) or None
        """
        orig_dtype = q.dtype
        B, T, H, K = q.shape
        V = v.shape[-1]
        scale = self.scale if self.scale is not None else K ** -0.5

        q, k, v = (x.float() for x in (q, k, v))

        # Per-head decay factor (pre-computed buffer)
        gamma = self.gamma  # (H,)

        h = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
        if initial_state is not None:
            h = h + initial_state.float()

        o = torch.zeros(B, T, H, V, dtype=torch.float32, device=q.device)

        for t in range(T):
            q_t = q[:, t] * scale            # (B, H, K)
            k_t = k[:, t]                     # (B, H, K)
            v_t = v[:, t]                     # (B, H, V)

            # Decay and accumulate: h = h * gamma + k^T v
            # gamma is (H,), broadcast to (1, H, 1, 1)
            h = h * gamma[None, :, None, None] + k_t.unsqueeze(-1) * v_t.unsqueeze(-2)

            # Output: o = q @ h  -> (B, H, V)
            o[:, t] = (q_t.unsqueeze(-1) * h).sum(-2)

        ht = h if output_final_state else None
        return o.to(orig_dtype), ht
