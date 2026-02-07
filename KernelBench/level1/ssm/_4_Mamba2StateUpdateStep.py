import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Mamba-2 State Update Step (Decode Path)

    Used by: Mamba-2, Mamba, RetNet, RWKV

    Single-timestep SSM state update for autoregressive decode:
        h_new = dA * h + dB * x
        y = C @ h_new + D * x

    Takes pre-discretized dA and dB as inputs (the discretization via
    softplus/exp is done upstream). This operator is a pure computation
    kernel with no learnable parameters — A, D, and discretization logic
    live in the parent mixer module.

    Multi-head aware: operates on (batch, num_heads, head_dim, state_size)
    shaped states natively.

    Shapes:
        x: (batch, num_heads, head_dim) — input at current timestep
        h: (batch, num_heads, head_dim, state_size) — previous SSM state
        dA: (batch, num_heads, head_dim, state_size) — discretized A
        dB: (batch, num_heads, head_dim, state_size) — discretized B (dt * B)
        C: (batch, num_heads, state_size) — state-to-output matrix
        D: (num_heads,) — skip connection
        Output h_new: (batch, num_heads, head_dim, state_size)
        Output y: (batch, 1, num_heads * head_dim)
    """

    def __init__(self):
        """Initialize Mamba-2 State Update Step (no learnable parameters)."""
        super(Model, self).__init__()

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        dA: torch.Tensor,
        dB: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
    ) -> tuple:
        """
        Compute one step of SSM state update.

        Args:
            x: Input at current timestep (batch, num_heads, head_dim)
            h: Previous state (batch, num_heads, head_dim, state_size)
            dA: Discretized A, exp(dt * A)
                (batch, num_heads, head_dim, state_size)
            dB: Discretized B, dt * B expanded
                (batch, num_heads, head_dim, state_size)
            C: State-to-output matrix (batch, num_heads, state_size)
            D: Skip connection (num_heads,)

        Returns:
            Tuple of (h_new, y):
                h_new: Updated state (batch, num_heads, head_dim, state_size)
                y: Output (batch, 1, num_heads * head_dim)
        """
        batch_size = x.shape[0]
        num_heads = x.shape[1]
        head_dim = x.shape[2]

        # State update: h_new = dA * h + dB * x
        dBx = dB * x[..., None]  # (batch, num_heads, head_dim, state_size)
        h_new = h * dA + dBx

        # Output: y = C @ h_new (batched matmul)
        # Reshape for bmm: (batch*num_heads, head_dim, state_size) @ (batch*num_heads, state_size, 1)
        h_reshaped = h_new.view(
            batch_size * num_heads, head_dim, -1
        )
        C_reshaped = C.view(batch_size * num_heads, -1, 1)
        y = torch.bmm(h_reshaped, C_reshaped)
        y = y.view(batch_size, num_heads, head_dim)

        # Skip connection: y = y + D * x
        D_expanded = D[..., None].expand(num_heads, head_dim)
        y = (y + x * D_expanded).to(y.dtype)

        # Reshape to (batch, 1, num_heads * head_dim)
        y = y.reshape(batch_size, -1)[:, None, ...]

        return h_new, y


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 4
num_heads = 128
head_dim = 64
state_size = 128


def get_inputs():
    x = torch.randn(batch_size, num_heads, head_dim)
    h = torch.randn(batch_size, num_heads, head_dim, state_size)
    dA = torch.randn(batch_size, num_heads, head_dim, state_size).exp()
    dB = torch.randn(batch_size, num_heads, head_dim, state_size)
    C = torch.randn(batch_size, num_heads, state_size)
    D = torch.ones(num_heads)
    return [x, h, dA, dB, C, D]


def get_init_inputs():
    return []
