import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union

class Model(nn.Module):
    """
    Multi-Tenant DoRA (Weight-Decomposed Low-Rank Adaptation)

    Used by: DoRA fine-tuning with multi-tenant serving

    DoRA decomposes weight updates into magnitude and direction components:
    W = m * (W0 + BA) / ||W0 + BA||

    This multi-tenant version supports serving many concurrent DoRA adapters
    where different tokens can use different adapters.

    Key features:
    - Token-level adapter assignment
    - Heterogeneous ranks support
    - Batched magnitude and direction computation
    - Shared frozen base weights

    Shapes:
        x: (total_tokens, hidden_size) - flattened batch of tokens
        adapter_ids: (total_tokens,) - adapter index per token
        Output: (total_tokens, hidden_size)
    """

    def __init__(self, in_features: int, out_features: int,
                 num_adapters: int = 100,
                 rank: Union[int, List[int]] = 16,
                 alpha: float = 32.0):
        """
        Initialize multi-tenant DoRA layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            num_adapters: Maximum number of concurrent DoRA adapters
            rank: DoRA rank - either single int (uniform) or list (heterogeneous)
            alpha: Scaling factor
        """
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_adapters = num_adapters
        self.alpha = alpha

        # Handle uniform vs heterogeneous ranks
        if isinstance(rank, int):
            self.adapter_ranks = [rank] * num_adapters
        else:
            assert len(rank) == num_adapters
            self.adapter_ranks = rank

        self.max_rank = max(self.adapter_ranks)

        # Per-adapter scaling factors
        self.register_buffer('scalings',
            torch.tensor([alpha / r for r in self.adapter_ranks]))

        self.register_buffer('ranks',
            torch.tensor(self.adapter_ranks, dtype=torch.long))

        # Frozen base weight (shared across all adapters)
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.02,
            requires_grad=False
        )

        # Multi-tenant low-rank adaptation matrices
        self.lora_A = nn.Parameter(torch.zeros(num_adapters, self.max_rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(num_adapters, out_features, self.max_rank))

        # Multi-tenant magnitude vectors (DoRA-specific)
        # Each adapter has its own magnitude vector
        with torch.no_grad():
            base_norm = self.weight.norm(dim=1)
        self.magnitudes = nn.Parameter(
            base_norm.unsqueeze(0).expand(num_adapters, -1).clone()
        )

        # Initialize adapters
        for i, r in enumerate(self.adapter_ranks):
            nn.init.kaiming_uniform_(self.lora_A.data[i, :r, :])

    def _create_rank_mask(self, adapter_ids: torch.Tensor) -> torch.Tensor:
        """Create mask for valid rank positions per token."""
        token_ranks = self.ranks[adapter_ids]
        rank_positions = torch.arange(self.max_rank, device=adapter_ids.device)
        return rank_positions.unsqueeze(0) < token_ranks.unsqueeze(1)

    def forward(self, x: torch.Tensor, adapter_ids: torch.Tensor) -> torch.Tensor:
        """
        Multi-tenant DoRA forward pass.

        Args:
            x: Input tokens (total_tokens, in_features)
            adapter_ids: Adapter index per token (total_tokens,)
                        Use -1 for tokens without adapters.

        Returns:
            Output tensor (total_tokens, out_features)
        """
        total_tokens = x.shape[0]
        device = x.device

        # Handle tokens without adapters (use base weight only)
        valid_mask = adapter_ids >= 0
        base_output = F.linear(x, self.weight)

        if not valid_mask.any():
            return base_output

        valid_indices = torch.where(valid_mask)[0]
        valid_adapter_ids = adapter_ids[valid_indices]
        x_valid = x[valid_indices]
        num_valid = x_valid.shape[0]

        # Gather adapter components
        A_gathered = self.lora_A[valid_adapter_ids]  # (num_valid, max_rank, in_features)
        B_gathered = self.lora_B[valid_adapter_ids]  # (num_valid, out_features, max_rank)
        magnitudes_gathered = self.magnitudes[valid_adapter_ids]  # (num_valid, out_features)
        scalings_gathered = self.scalings[valid_adapter_ids]  # (num_valid,)

        # Compute adapted weight per token: W0 + scaling * B @ A
        # This is done implicitly through the forward pass

        # For DoRA, we need to compute the direction-normalized output
        # First compute LoRA contribution: x @ A^T @ B^T
        intermediate = torch.einsum('nh,nrh->nr', x_valid, A_gathered)

        # Mask for heterogeneous ranks
        if self.max_rank != min(self.adapter_ranks):
            rank_mask = self._create_rank_mask(valid_adapter_ids)
            intermediate = intermediate * rank_mask.float()

        lora_contribution = torch.einsum('nr,nor->no', intermediate, B_gathered)
        lora_contribution = lora_contribution * scalings_gathered.unsqueeze(-1)

        # Base output for valid tokens
        base_valid = base_output[valid_indices]

        # Combined output before normalization
        combined = base_valid + lora_contribution

        # Compute adapted weight norm for direction normalization
        # ||W0 + scaling * BA||_row for each output dimension
        # Approximation: use running norm update instead of full weight materialization
        BA = torch.einsum('nrh,nor->noh', A_gathered, B_gathered)  # (num_valid, out, in)
        BA = BA * scalings_gathered.unsqueeze(-1).unsqueeze(-1)

        adapted_weight = self.weight.unsqueeze(0) + BA  # (num_valid, out, in)
        weight_norms = adapted_weight.norm(dim=-1)  # (num_valid, out)

        # Direction normalization and magnitude scaling
        direction = combined / (weight_norms + 1e-8)
        dora_output = magnitudes_gathered * direction

        # Scatter back
        output = base_output.clone()
        output[valid_indices] = dora_output

        return output


# ============================================================================
# Benchmark Configuration (Multi-tenant serving scenario)
# ============================================================================

total_tokens = 8192
in_features = 4096
out_features = 4096
num_adapters = 100
adapter_ranks = [8, 16, 32] * 33 + [16]  # Heterogeneous ranks

def get_inputs():
    """Generate input tensors for multi-tenant DoRA benchmarking."""
    x = torch.randn(total_tokens, in_features, device='cuda')
    adapter_ids = torch.randint(0, num_adapters, (total_tokens,), device='cuda')
    return [x, adapter_ids]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [in_features, out_features, num_adapters, adapter_ranks]
