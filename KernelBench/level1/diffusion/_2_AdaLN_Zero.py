import torch
import torch.nn as nn
from typing import Tuple


class Model(nn.Module):
    """
    AdaLN-Zero (Adaptive Layer Normalization with Zero-init)

    Used by: DiT, SD-3/3.5 (AdaLayerNormZero), FLUX

    Adaptive LayerNorm where shift, scale, and gate parameters are predicted
    from a conditioning embedding via SiLU + Linear. The linear projection
    outputs ``num_output_chunks * hidden_size`` values, which are chunked and
    returned for the caller to apply.

    The first chunk pair (shift_msa, scale_msa) is applied to the normalized
    input immediately:
        norm_x = LayerNorm(x) * (1 + scale_msa) + shift_msa

    Remaining chunks (gate_msa, shift_mlp, scale_mlp, gate_mlp, ...) are
    returned as-is for the caller to use in subsequent residual/FFN paths.

    Configurable ``num_output_chunks``:
        - 3: DiT original (shift, scale, gate) → returns (norm_x, gate)
        - 6: SD3/3.5 JointTransformerBlock (shift_msa, scale_msa, gate_msa,
              shift_mlp, scale_mlp, gate_mlp) → returns
              (norm_x, gate_msa, shift_mlp, scale_mlp, gate_mlp)

    Matches diffusers AdaLayerNormZero exactly when num_output_chunks=6.

    State-dict key layout (for HuggingFace weight compatibility):
        silu   — parameter-free SiLU activation
        linear — nn.Linear(hidden_size, num_output_chunks * hidden_size)
        norm   — nn.LayerNorm(hidden_size)

    Shapes:
        x: (batch, seq_len, hidden_size)
        conditioning: (batch, hidden_size)
        Output: tuple of tensors (see above)
    """

    def __init__(self, hidden_size: int, num_output_chunks: int = 6,
                 eps: float = 1e-6, bias: bool = True):
        """
        Initialize AdaLN-Zero.

        Args:
            hidden_size: Hidden dimension (also used as cond_dim since
                         diffusers AdaLayerNormZero uses same dim for both)
            num_output_chunks: Number of chunks to produce from the linear
                               projection. Must be >= 3.
            eps: LayerNorm epsilon
            bias: Whether the Linear projection has bias
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_output_chunks = num_output_chunks

        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_size, num_output_chunks * hidden_size,
                                bias=bias)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=eps)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Apply AdaLN-Zero.

        Args:
            x: Input tensor (batch, seq_len, hidden_size)
            emb: Conditioning embedding (batch, hidden_size)

        Returns:
            Tuple whose first element is the normalized+modulated hidden
            states, followed by the remaining chunks:
            - num_output_chunks=3: (norm_x, gate)
            - num_output_chunks=6: (norm_x, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        """
        emb = self.linear(self.silu(emb))
        chunks = emb.chunk(self.num_output_chunks, dim=-1)

        # First two chunks are always shift_msa, scale_msa
        shift_msa, scale_msa = chunks[0], chunks[1]

        # Apply layer norm + adaptive scale/shift
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]

        # Return normalized x + remaining chunks
        return (x,) + chunks[2:]
