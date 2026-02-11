import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Mamba-1 Selective Scan (Sequential)

    Used by: Mamba-1, Falcon-Mamba, Jamba (SSM layers)

    Implements the Mamba-1 selective scan recurrence without owning parameters.
    This operator takes pre-discretized inputs and performs the sequential
    state-space recurrence:
        h[t] = dA[t] * h[t-1] + dB_x[t]
        y[t] = (h[t] @ C[t]) + D * x[t]
    followed by gating: y = y * act(gate)

    This is the core scan loop extracted from the Mamba mixer, designed to be
    a reusable level1 operator that can be shared across Mamba-1 variants.

    The operator does NOT own parameters (A_log, D, etc.) — these are passed
    as inputs from the mixer that owns them.

    Shapes:
        x:     (batch, intermediate_size, seq_len) — hidden states after conv
        dA:    (batch, intermediate_size, seq_len, state_size) — discretized A
        dB_x:  (batch, intermediate_size, seq_len, state_size) — discretized B * x
        C:     (batch, seq_len, state_size) — readout matrix
        D:     (intermediate_size,) — skip connection
        gate:  (batch, intermediate_size, seq_len) — gating tensor
        ssm_state: (batch, intermediate_size, state_size) or None — initial state
        Output: (batch, intermediate_size, seq_len), final_ssm_state
    """

    def __init__(self):
        """Initialize Mamba-1 Selective Scan (no owned parameters)."""
        super(Model, self).__init__()

    def forward(
        self,
        x: torch.Tensor,
        dA: torch.Tensor,
        dB_x: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        gate: torch.Tensor,
        ssm_state: torch.Tensor = None,
    ) -> tuple:
        """
        Run Mamba-1 selective scan.

        Args:
            x: Hidden states (batch, intermediate_size, seq_len)
            dA: Discretized A (batch, intermediate_size, seq_len, state_size)
            dB_x: Discretized B*x (batch, intermediate_size, seq_len, state_size)
            C: Readout matrix (batch, seq_len, state_size)
            D: Skip connection (intermediate_size,)
            gate: Gating tensor (batch, intermediate_size, seq_len)
            ssm_state: Initial SSM state (batch, intermediate_size, state_size) or None

        Returns:
            Tuple of:
                scan_output: (batch, intermediate_size, seq_len)
                final_ssm_state: (batch, intermediate_size, state_size)
        """
        batch_size = x.shape[0]
        seq_len = x.shape[2]
        dtype = x.dtype

        if ssm_state is None:
            ssm_state = torch.zeros(
                batch_size, x.shape[1], C.shape[-1],
                device=x.device, dtype=dtype,
            )

        scan_outputs = []
        for i in range(seq_len):
            ssm_state = (
                dA[:, :, i, :] * ssm_state + dB_x[:, :, i, :]
            )  # [batch, intermediate_size, state_size]
            scan_output = torch.matmul(
                ssm_state.to(dtype), C[:, i, :].unsqueeze(-1)
            )  # [batch, intermediate_size, 1]
            scan_outputs.append(scan_output[:, :, 0])

        scan_output = torch.stack(scan_outputs, dim=-1)  # [batch, intermediate_size, seq_len]
        scan_output = scan_output + (x * D[None, :, None])
        scan_output = scan_output * F.silu(gate)

        return scan_output, ssm_state
