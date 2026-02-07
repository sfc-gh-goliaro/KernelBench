import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Mamba Causal Depthwise Conv1d

    Used by: Mamba, Mamba-2

    Causal depthwise 1D convolution used in SSM models for local context
    mixing before the selective scan. Applies depthwise conv with causal
    padding (kernel_size - 1 left-padding), followed by optional SiLU
    activation.

    This operator handles the full-sequence (prefill) convolution path.
    For single-step decode with cached conv states, see MambaCausalConv1dStep.

    Shapes:
        Input:  (batch, seq_len, d_inner)
        Output: (batch, seq_len, d_inner)
    """

    def __init__(self, d_inner: int, kernel_size: int = 4, bias: bool = True,
                 apply_silu: bool = True):
        """
        Initialize Mamba Causal Conv1d.

        Args:
            d_inner: Number of channels (processed depthwise)
            kernel_size: Convolution kernel size
            bias: Whether to include a bias term
            apply_silu: Whether to apply SiLU activation after convolution
        """
        super(Model, self).__init__()
        self.d_inner = d_inner
        self.kernel_size = kernel_size
        self.apply_silu = apply_silu

        # Depthwise causal convolution
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size,
            padding=kernel_size - 1,  # Causal padding
            groups=d_inner,  # Depthwise
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply causal depthwise convolution with optional SiLU.

        Args:
            x: Input tensor (batch, seq_len, d_inner)

        Returns:
            Output tensor (batch, seq_len, d_inner)
        """
        seq_len = x.shape[1]

        # Transpose to channels-first for nn.Conv1d: (batch, d_inner, seq_len)
        x = x.transpose(1, 2)

        # Apply depthwise convolution
        x = self.conv1d(x)

        # Truncate to original length (remove right-side padding for causality)
        x = x[..., :seq_len]

        # Optional SiLU activation (standard in Mamba/Mamba-2)
        if self.apply_silu:
            x = F.silu(x)

        # Transpose back: (batch, seq_len, d_inner)
        x = x.transpose(1, 2)

        return x


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 16
seq_len = 1024
d_inner = 10240  # Mamba-2 Codestral: intermediate_size + 2 * n_groups * state_size


def get_inputs():
    return [torch.randn(batch_size, seq_len, d_inner)]


def get_init_inputs():
    return [d_inner]
