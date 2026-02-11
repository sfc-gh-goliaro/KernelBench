import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Mamba-1 SSM Single-Step Update (Decode Path)

    Used by: Mamba-1, Falcon-Mamba

    Single-step SSM state update for autoregressive decode in Mamba-1 models.
    Computes one recurrence step:
        h_new = dA * h + dB * x
        y = (h_new @ C) + D * x
        y = y * silu(gate)

    This is the decode-path complement to Mamba1SelectiveScan (prefill).
    Does NOT own parameters — takes all inputs from the mixer.

    Shapes:
        x:     (batch, intermediate_size, 1) — single-step hidden state
        dA:    (batch, intermediate_size, 1, state_size) — discretized A
        dB_x:  (batch, intermediate_size, 1, state_size) — discretized B*x
        C:     (batch, 1, state_size) — readout matrix
        D:     (intermediate_size,) — skip connection
        gate:  (batch, intermediate_size, 1) — gating tensor
        ssm_state: (batch, intermediate_size, state_size) — current state
        Output: y (batch, intermediate_size, 1), h_new (batch, intermediate_size, state_size)
    """

    def __init__(self):
        """Initialize Mamba-1 SSM Step (no owned parameters)."""
        super(Model, self).__init__()

    def forward(
        self,
        x: torch.Tensor,
        dA: torch.Tensor,
        dB_x: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        gate: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> tuple:
        """
        Compute one step of Mamba-1 SSM recurrence.

        Args:
            x: Hidden state (batch, intermediate_size, 1)
            dA: Discretized A (batch, intermediate_size, 1, state_size)
            dB_x: Discretized B*x (batch, intermediate_size, 1, state_size)
            C: Readout matrix (batch, 1, state_size)
            D: Skip connection (intermediate_size,)
            gate: Gating tensor (batch, intermediate_size, 1)
            ssm_state: Current SSM state (batch, intermediate_size, state_size)

        Returns:
            Tuple of:
                y: Output (batch, intermediate_size, 1)
                h_new: Updated state (batch, intermediate_size, state_size)
        """
        dtype = x.dtype

        # State update: h_new = dA * h + dB_x
        h_new = dA[:, :, 0, :] * ssm_state + dB_x[:, :, 0, :]

        # Output: y = h_new @ C^T + D * x
        y = torch.matmul(
            h_new.to(dtype), C[:, 0, :].unsqueeze(-1)
        )  # [batch, intermediate_size, 1]
        y = y + x * D[None, :, None]
        y = y * F.silu(gate)

        return y, h_new
