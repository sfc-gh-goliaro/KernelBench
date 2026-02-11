import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Gated RMSNorm

    Used by: Mamba-2, Bamba, Falcon-H1, Zamba2, Qwen3-Next, GraniteMoeHybrid

    RMSNorm with SiLU-gated multiplicative modulation. The gate tensor
    is activated via SiLU and multiplied with hidden_states either before
    or after normalization, controlled by `norm_before_gate`.

    Supports grouped normalization (Falcon-H1, Zamba2) where the norm
    variance is computed per group of `hidden_size // n_groups` features
    rather than over the full hidden dimension.

    Implements exact HuggingFace MambaRMSNormGated behavior:
    - Casts to float32 for numerical stability
    - Uses rsqrt for computation
    - Returns result in input dtype

    Shapes:
        hidden_states: (batch, seq_len, hidden_size)
        gate:          (batch, seq_len, hidden_size) or None
        Output:        (batch, seq_len, hidden_size)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6,
                 n_groups: int = 1, norm_before_gate: bool = False):
        """
        Initialize Gated RMSNorm.

        Args:
            hidden_size: Number of features in the input tensor.
            eps: Small value added to denominator for numerical stability.
            n_groups: Number of groups for grouped normalization. When 1,
                computes variance over full hidden_size (standard RMSNorm).
                When > 1, computes variance per group of hidden_size // n_groups
                features (Falcon-H1, Zamba2 style).
            norm_before_gate: Controls gate application order.
                False (default): gate applied BEFORE norm, i.e.
                    output = weight * RMSNorm(x * SiLU(gate))
                    Used by: Mamba-2, Bamba, GraniteMoeHybrid
                True: gate applied AFTER norm, i.e.
                    output = weight * RMSNorm(x) * SiLU(gate)
                    Used by: Qwen3-Next, Falcon-H1 (with norm_before_gate=True)
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.n_groups = n_groups
        self.norm_before_gate = norm_before_gate

        # Learnable weight (always present in all HF variants)
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor,
                gate: torch.Tensor = None) -> torch.Tensor:
        """
        Apply gated RMSNorm.

        Args:
            hidden_states: Input tensor (batch, seq_len, hidden_size)
            gate: Gate tensor (batch, seq_len, hidden_size) or None.
                When None, behaves as standard RMSNorm.

        Returns:
            Normalized and optionally gated tensor (batch, seq_len, hidden_size)
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        # Apply gate BEFORE norm (Mamba2 / Bamba style)
        if not self.norm_before_gate and gate is not None:
            hidden_states = hidden_states * F.silu(gate.to(torch.float32))

        # Grouped or standard RMSNorm
        if self.n_groups > 1:
            # Grouped normalization: compute variance per group
            *prefix_dims, last_dim = hidden_states.shape
            group_size = last_dim // self.n_groups
            hidden_states = hidden_states.view(
                *prefix_dims, self.n_groups, group_size
            )
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
            hidden_states = (
                self.weight.view(self.n_groups, group_size) * hidden_states
            )
            hidden_states = hidden_states.view(*prefix_dims, last_dim)
        else:
            # Standard RMSNorm over full hidden_size
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
            hidden_states = self.weight * hidden_states

        # Apply gate AFTER norm (Qwen3-Next style)
        if self.norm_before_gate and gate is not None:
            hidden_states = hidden_states * F.silu(gate.to(torch.float32))

        return hidden_states.to(input_dtype)
