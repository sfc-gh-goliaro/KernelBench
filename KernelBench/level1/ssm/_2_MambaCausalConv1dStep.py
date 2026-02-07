import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Mamba Causal Conv1d Step (Decode Path)

    Used by: Mamba, Mamba-2

    Single-step cached convolution for autoregressive decode in SSM models.
    Instead of running a full nn.Conv1d over the whole sequence, this operator:
    1. Rolls the conv state cache left by one position
    2. Inserts the new token into the last position
    3. Computes the convolution output as a dot-product with the kernel weights
    4. Adds optional bias and applies SiLU activation

    This is the decode-path complement to MambaCausalConv1d (which handles prefill).

    This operator takes conv weights and bias as inputs rather than owning them,
    since in practice they are shared with the MambaCausalConv1d operator used
    during prefill.

    Shapes:
        new_token:   (batch, d_inner) — new input to insert into cache
        conv_state:  (batch, d_inner, kernel_size) — cached conv state
        conv_weight: (d_inner, 1, kernel_size) — depthwise conv weights
        conv_bias:   (d_inner,) or None — optional bias
        Output:      (batch, d_inner), updated conv_state (batch, d_inner, kernel_size)
    """

    def __init__(self, apply_silu: bool = True):
        """
        Initialize Mamba Causal Conv1d Step.

        Args:
            apply_silu: Whether to apply SiLU activation after convolution
        """
        super(Model, self).__init__()
        self.apply_silu = apply_silu

    def forward(
        self,
        new_token: torch.Tensor,
        conv_state: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor = None,
    ) -> tuple:
        """
        Compute one step of cached causal convolution.

        Args:
            new_token: New input token (batch, d_inner)
            conv_state: Cached conv state (batch, d_inner, kernel_size)
            conv_weight: Depthwise conv weights (d_inner, 1, kernel_size)
            conv_bias: Optional bias (d_inner,) or None

        Returns:
            Tuple of (output, new_conv_state):
                output: Convolution output (batch, d_inner)
                new_conv_state: Updated cache (batch, d_inner, kernel_size)
        """
        # Roll state left and insert new token at the end
        new_conv_state = conv_state.roll(shifts=-1, dims=-1)
        new_conv_state[:, :, -1] = new_token

        # Dot-product with conv weights: sum over kernel_size dimension
        # weight shape: (d_inner, 1, kernel_size), squeeze to (d_inner, kernel_size)
        output = torch.sum(
            new_conv_state * conv_weight.squeeze(1), dim=-1
        )

        # Add bias
        if conv_bias is not None:
            output = output + conv_bias

        # Optional SiLU activation
        if self.apply_silu:
            output = F.silu(output)

        return output, new_conv_state


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 16
d_inner = 10240
kernel_size = 4


def get_inputs():
    return [
        torch.randn(batch_size, d_inner),                  # new_token
        torch.randn(batch_size, d_inner, kernel_size),      # conv_state
        torch.randn(d_inner, 1, kernel_size),               # conv_weight
        torch.randn(d_inner),                               # conv_bias
    ]


def get_init_inputs():
    return []
