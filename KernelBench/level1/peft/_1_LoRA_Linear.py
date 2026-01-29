import torch
import torch.nn as nn
from typing import List, Union, Tuple

class Model(nn.Module):
    """
    Multi-Tenant LoRA with SGMV/BGMV (Punica/S-LoRA Style)

    Used by: Punica, S-LoRA, vLLM multi-LoRA, LoRAX

    Efficient multi-tenant LoRA serving where different tokens in a batch
    can use different LoRA adapters. Supports heterogeneous adapter ranks
    and uses batched/segmented computation to avoid explicit loops.

    Key features:
    - Token-level adapter assignment
    - Heterogeneous ranks: each adapter can have different rank
    - BGMV (Batched Grouped Matrix-Vector) computation
    - SGMV (Segmented) computation mode for grouped tokens
    - Supports 100s to 1000s of concurrent adapters

    Shapes:
        x: (total_tokens, hidden_size) - flattened batch of tokens
        adapter_ids: (total_tokens,) - adapter index per token
        Output: (total_tokens, hidden_size)
    """

    def __init__(self, in_features: int, out_features: int,
                 num_adapters: int = 100,
                 rank: Union[int, List[int]] = 16,
                 alpha: float = 16.0):
        """
        Initialize multi-tenant LoRA layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            num_adapters: Maximum number of concurrent LoRA adapters
            rank: LoRA rank - either single int (uniform) or list (heterogeneous)
            alpha: LoRA alpha for scaling
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

        # Store actual ranks for masking with heterogeneous ranks
        self.register_buffer('ranks',
            torch.tensor(self.adapter_ranks, dtype=torch.long))

        # Base linear (shared, frozen during serving)
        self.base_linear = nn.Linear(in_features, out_features, bias=False)

        # LoRA adapter pool: unified layout padded to max_rank
        # A matrices: down-projection (in_features -> rank)
        # B matrices: up-projection (rank -> out_features)
        self.lora_A = nn.Parameter(torch.zeros(num_adapters, self.max_rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(num_adapters, out_features, self.max_rank))

        # Initialize adapters
        for i, r in enumerate(self.adapter_ranks):
            nn.init.kaiming_uniform_(self.lora_A.data[i, :r, :])
            # B initialized to zero so LoRA starts as identity

    def _create_rank_mask(self, adapter_ids: torch.Tensor) -> torch.Tensor:
        """Create mask for valid rank positions per token."""
        token_ranks = self.ranks[adapter_ids]
        rank_positions = torch.arange(self.max_rank, device=adapter_ids.device)
        return rank_positions.unsqueeze(0) < token_ranks.unsqueeze(1)

    def forward(self, x: torch.Tensor, adapter_ids: torch.Tensor) -> torch.Tensor:
        """
        Multi-tenant LoRA forward with BGMV-style computation.

        Args:
            x: Input tokens (total_tokens, in_features)
            adapter_ids: Adapter index per token (total_tokens,)
                        Use -1 for tokens without adapters.

        Returns:
            Output tensor (total_tokens, out_features)
        """
        device = x.device

        # Base model output (shared computation)
        base_output = self.base_linear(x)

        # Handle tokens without adapters
        valid_mask = adapter_ids >= 0
        if not valid_mask.any():
            return base_output

        valid_indices = torch.where(valid_mask)[0]
        valid_adapter_ids = adapter_ids[valid_indices]
        x_valid = x[valid_indices]

        # Gather adapter matrices for each token
        A_gathered = self.lora_A[valid_adapter_ids]  # (num_valid, max_rank, in_features)
        B_gathered = self.lora_B[valid_adapter_ids]  # (num_valid, out_features, max_rank)

        # BGMV expand: x @ A^T -> (num_valid, max_rank)
        intermediate = torch.einsum('nh,nrh->nr', x_valid, A_gathered)

        # Mask for heterogeneous ranks (only needed if ranks vary)
        if self.max_rank != min(self.adapter_ranks):
            rank_mask = self._create_rank_mask(valid_adapter_ids)
            intermediate = intermediate * rank_mask.float()

        # BGMV shrink: intermediate @ B^T -> (num_valid, out_features)
        lora_output = torch.einsum('nr,nor->no', intermediate, B_gathered)

        # Apply per-adapter scaling
        token_scalings = self.scalings[valid_adapter_ids].unsqueeze(-1)
        lora_output = lora_output * token_scalings

        # Scatter back to output
        output = base_output.clone()
        output[valid_indices] = output[valid_indices] + lora_output

        return output


# ============================================================================
# Benchmark Configuration (Multi-tenant serving scenario)
# ============================================================================

import os
import sys
