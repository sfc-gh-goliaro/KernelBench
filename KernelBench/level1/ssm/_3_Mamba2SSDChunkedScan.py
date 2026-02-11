import torch
import torch.nn as nn
import torch.nn.functional as F


def _pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int):
    """Pad tensor with zeros on the seq_len dim (dim=1)."""
    if pad_size == 0:
        return input_tensor
    pad_shape = (
        (0, 0, 0, 0, 0, pad_size, 0, 0)
        if len(input_tensor.shape) == 4
        else (0, 0, 0, pad_size, 0, 0)
    )
    return F.pad(input_tensor, pad_shape, mode="constant", value=0)


def _reshape_into_chunks(input_tensor, pad_size, chunk_size):
    """Pad and reshape into chunk sequences."""
    input_tensor = _pad_tensor_by_size(input_tensor, pad_size)
    if len(input_tensor.shape) == 3:
        return input_tensor.reshape(
            input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2]
        )
    else:
        return input_tensor.reshape(
            input_tensor.shape[0], -1, chunk_size,
            input_tensor.shape[2], input_tensor.shape[3]
        )


def _segment_sum(input_tensor):
    """Stable segment sum using cumulative sums and masking."""
    chunk_size = input_tensor.size(-1)
    input_tensor = input_tensor[..., None].expand(
        *input_tensor.size(), chunk_size
    )
    mask = torch.tril(
        torch.ones(
            chunk_size, chunk_size,
            device=input_tensor.device, dtype=torch.bool
        ),
        diagonal=-1,
    )
    input_tensor = input_tensor.masked_fill(~mask, 0)
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)
    mask = torch.tril(
        torch.ones(
            chunk_size, chunk_size,
            device=input_tensor.device, dtype=torch.bool
        ),
        diagonal=0,
    )
    tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
    return tensor_segsum


class Model(nn.Module):
    """
    Mamba-2 SSD Chunked Parallel Scan (Prefill Path)

    Used by: Mamba-2, Codestral Mamba

    The core Mamba-2 State-Space Duality (SSD) computation using chunked
    parallel scan. This is the most compute-intensive part of Mamba-2
    and the primary kernel optimization target.

    Given pre-discretized inputs (x already scaled by dt, A already scaled
    by dt), computes:
    1. Intra-chunk diagonal blocks via segment_sum
    2. Inter-chunk state accumulation (B terms)
    3. Inter-chunk SSM recurrence (A terms)
    4. State-to-output via C

    All inputs should already have group-expanded B and C (i.e.,
    repeat_interleave applied before calling this operator).

    Shapes:
        x: (batch, seq_len, num_heads, head_dim) — pre-discretized (x * dt)
        A: (batch, seq_len, num_heads) — pre-discretized (A * dt)
        B: (batch, seq_len, num_heads, state_size) — group-expanded
        C: (batch, seq_len, num_heads, state_size) — group-expanded
        D: (num_heads,) — skip connection
        previous_state: (batch, num_heads, head_dim, state_size) or None
        Output y: (batch, seq_len, num_heads * head_dim)
        Output final_state: (batch, num_heads, head_dim, state_size)
    """

    def __init__(self, chunk_size: int = 256):
        """
        Initialize Mamba-2 SSD Chunked Scan.

        Args:
            chunk_size: Chunk size for parallel scan
        """
        super(Model, self).__init__()
        self.chunk_size = chunk_size

    def forward(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        x_raw: torch.Tensor,
        previous_state: torch.Tensor = None,
    ) -> tuple:
        """
        SSD chunked scan forward pass.

        Args:
            x: Pre-discretized input, x * dt
                (batch, seq_len, num_heads, head_dim), float32
            A: Pre-discretized state decay, A * dt
                (batch, seq_len, num_heads), float32
            B: Group-expanded input-to-state matrix
                (batch, seq_len, num_heads, state_size), float32
            C: Group-expanded state-to-output matrix
                (batch, seq_len, num_heads, state_size), float32
            D: Skip connection weights (num_heads,)
            x_raw: Original (un-discretized) hidden states for D residual
                (batch, seq_len, num_heads, head_dim), float32
            previous_state: Previous SSM state from cache, or None
                (batch, num_heads, head_dim, state_size)

        Returns:
            Tuple of (y, final_state):
                y: Output (batch, seq_len, num_heads * head_dim)
                final_state: Final SSM state (batch, num_heads, head_dim, state_size)
        """
        batch_size, seq_len, num_heads, head_dim = x.shape
        state_size = B.shape[-1]
        chunk_size = self.chunk_size

        pad_size = (chunk_size - seq_len % chunk_size) % chunk_size

        # D residual: D * x_raw (before chunking)
        D_residual = D[..., None] * _pad_tensor_by_size(x_raw, pad_size)

        # Rearrange into blocks/chunks
        x, A, B, C = [
            _reshape_into_chunks(t, pad_size, chunk_size)
            for t in (x, A, B, C)
        ]

        # A: (batch, num_chunks, chunk_size, num_heads) -> (batch, num_heads, num_chunks, chunk_size)
        A = A.permute(0, 3, 1, 2)
        A_cumsum = torch.cumsum(A, dim=-1)

        # 1. Intra-chunk (diagonal blocks)
        L = torch.exp(_segment_sum(A))

        G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, :, :]
        G = G_intermediate.sum(dim=-1)

        M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
        M = M_intermediate.sum(dim=-1)

        Y_diag = (M[..., None] * x[:, :, None]).sum(dim=3)

        # 2. State for each intra-chunk (B terms)
        decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
        B_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]
        states = (B_decay[..., None, :] * x[..., None]).sum(dim=2)

        # 3. Inter-chunk SSM recurrence (A terms)
        if previous_state is not None:
            previous_states = previous_state[:, None, ...].to(
                device=states.device
            )
        else:
            previous_states = torch.zeros_like(states[:, :1])
        states = torch.cat([previous_states, states], dim=1)
        decay_chunk = torch.exp(
            _segment_sum(F.pad(A_cumsum[:, :, :, -1], (1, 0)))
        )
        decay_chunk = decay_chunk.transpose(1, 3)
        new_states = (
            decay_chunk[..., None, None] * states[:, :, None, ...]
        ).sum(dim=1)
        states, final_state = new_states[:, :-1], new_states[:, -1]

        # 4. State -> output (C terms)
        state_decay_out = torch.exp(A_cumsum)
        C_times_states = C[..., None, :] * states[:, :, None, ...]
        state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
        Y_off = (
            C_times_states.sum(-1) * state_decay_out_permuted[..., None]
        )

        # Combine diagonal and off-diagonal
        y = Y_diag + Y_off
        y = y.reshape(batch_size, -1, num_heads, head_dim)

        # Add D residual
        y = y + D_residual

        # Trim padding
        if pad_size > 0:
            y = y[:, :seq_len, :, :]

        # Flatten heads into features
        y = y.reshape(batch_size, seq_len, -1)

        return y, final_state
