"""
Gated Linear Attention (GLA) Naive Recurrent

Used by: GLA

Naive (sequential) recurrent implementation of the Gated Linear Attention
mechanism.  The recurrence at each time step is:

    h_t = h_{t-1} * exp(gk_t) + k_t^T @ v_t
    o_t = q_t @ h_t

where q, k, v are per-head per-timestep tensors, gk is the log-space
gating factor (typically `logsigmoid(gate_proj) / normalizer`), and the
output is scaled by `1 / sqrt(K)`.

Shapes:
    q:       (B, T, H, K)
    k:       (B, T, H, K)
    v:       (B, T, H, V)
    gk:      (B, T, H, K)   – log-space gate values
    initial_state: (B, H, K, V) or None
    Output:  (B, T, H, V)
    final_state: (B, H, K, V) or None
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class Model(nn.Module):
    """
    GLA naive recurrent linear attention.

    Matches fla's fused_recurrent_gla semantics exactly.
    This is a pure-PyTorch reference; production kernels would fuse the loop.
    """

    def __init__(self, scale: float = None):
        """
        Args:
            scale: Scaling factor applied to q at each time step.
                   If None, defaults to K^{-0.5}.
        """
        super(Model, self).__init__()
        self.scale = scale

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gk: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        GLA naive recurrent forward pass.

        Args:
            q: Query tensor (B, T, H, K)
            k: Key tensor (B, T, H, K)
            v: Value tensor (B, T, H, V)
            gk: Log-space gate tensor (B, T, H, K)
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

        q, k, v, gk = (x.float() for x in (q, k, v, gk))

        h = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
        if initial_state is not None:
            h = h + initial_state.float()

        o = torch.zeros(B, T, H, V, dtype=torch.float32, device=q.device)

        for t in range(T):
            q_t = q[:, t] * scale            # (B, H, K)
            k_t = k[:, t]                     # (B, H, K)
            v_t = v[:, t]                     # (B, H, V)
            gk_t = gk[:, t].exp()             # (B, H, K)

            # Decay and accumulate: h = h * gk + k^T v
            h = h * gk_t.unsqueeze(-1) + k_t.unsqueeze(-1) * v_t.unsqueeze(-2)

            # Output: o = q @ h  -> (B, H, V)
            o[:, t] = (q_t.unsqueeze(-1) * h).sum(-2)

        ht = h if output_final_state else None
        return o.to(orig_dtype), ht


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 2
seq_len = 256
num_heads = 4
head_k_dim = 64
head_v_dim = 128


def get_inputs():
    import torch.nn.functional as F
    q = torch.randn(batch_size, seq_len, num_heads, head_k_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_k_dim)
    v = torch.randn(batch_size, seq_len, num_heads, head_v_dim)
    gk = F.logsigmoid(torch.randn(batch_size, seq_len, num_heads, head_k_dim))
    return [q, k, v, gk]


def get_init_inputs():
    return []
