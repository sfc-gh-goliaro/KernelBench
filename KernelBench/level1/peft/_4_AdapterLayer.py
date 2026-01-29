import torch
import torch.nn as nn
from typing import List, Union

class Model(nn.Module):
    """
    Multi-Tenant Adapter Layer (Houlsby-style)

    Used by: Adapter-based fine-tuning with multi-tenant serving

    Lightweight bottleneck layer for parameter-efficient fine-tuning.
    This multi-tenant version supports serving many concurrent adapter
    configurations where different tokens use different adapters.

    Architecture per adapter: x + W_up(activation(W_down(x)))

    Key features:
    - Token-level adapter assignment
    - Heterogeneous bottleneck sizes
    - Batched adapter computation
    - Residual connection preserved

    Shapes:
        x: (total_tokens, hidden_size) - flattened batch of tokens
        adapter_ids: (total_tokens,) - adapter index per token
        Output: (total_tokens, hidden_size)
    """

    def __init__(self, hidden_size: int, num_adapters: int = 100,
                 adapter_size: Union[int, List[int]] = 64):
        """
        Initialize multi-tenant adapter layer.

        Args:
            hidden_size: Input/output hidden dimension
            num_adapters: Maximum number of concurrent adapters
            adapter_size: Bottleneck size - int (uniform) or list (heterogeneous)
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_adapters = num_adapters

        # Handle uniform vs heterogeneous adapter sizes
        if isinstance(adapter_size, int):
            self.adapter_sizes = [adapter_size] * num_adapters
        else:
            assert len(adapter_size) == num_adapters
            self.adapter_sizes = adapter_size

        self.max_adapter_size = max(self.adapter_sizes)

        self.register_buffer('sizes',
            torch.tensor(self.adapter_sizes, dtype=torch.long))

        # Multi-tenant adapter weights (padded to max size)
        # Down projection: hidden_size -> adapter_size
        self.down_weights = nn.Parameter(
            torch.zeros(num_adapters, self.max_adapter_size, hidden_size)
        )
        self.down_biases = nn.Parameter(
            torch.zeros(num_adapters, self.max_adapter_size)
        )

        # Up projection: adapter_size -> hidden_size
        self.up_weights = nn.Parameter(
            torch.zeros(num_adapters, hidden_size, self.max_adapter_size)
        )
        self.up_biases = nn.Parameter(
            torch.zeros(num_adapters, hidden_size)
        )

        # Initialize adapters
        for i, size in enumerate(self.adapter_sizes):
            nn.init.kaiming_uniform_(self.down_weights.data[i, :size, :])
            # Up projection initialized to zero for identity init
            # (already zeros)

    def _create_size_mask(self, adapter_ids: torch.Tensor) -> torch.Tensor:
        """Create mask for valid adapter positions per token."""
        token_sizes = self.sizes[adapter_ids]
        size_positions = torch.arange(self.max_adapter_size, device=adapter_ids.device)
        return size_positions.unsqueeze(0) < token_sizes.unsqueeze(1)

    def forward(self, x: torch.Tensor, adapter_ids: torch.Tensor) -> torch.Tensor:
        """
        Multi-tenant adapter forward pass.

        Args:
            x: Input tokens (total_tokens, hidden_size)
            adapter_ids: Adapter index per token (total_tokens,)
                        Use -1 for tokens without adapters (identity).

        Returns:
            Output tensor (total_tokens, hidden_size)
        """
        # Handle tokens without adapters (identity function)
        valid_mask = adapter_ids >= 0
        if not valid_mask.any():
            return x

        valid_indices = torch.where(valid_mask)[0]
        valid_adapter_ids = adapter_ids[valid_indices]
        x_valid = x[valid_indices]

        # Gather adapter weights for each token
        down_w = self.down_weights[valid_adapter_ids]  # (num_valid, max_size, hidden)
        down_b = self.down_biases[valid_adapter_ids]   # (num_valid, max_size)
        up_w = self.up_weights[valid_adapter_ids]      # (num_valid, hidden, max_size)
        up_b = self.up_biases[valid_adapter_ids]       # (num_valid, hidden)

        # Down projection: x @ W_down^T + b_down
        h = torch.einsum('nh,nah->na', x_valid, down_w) + down_b

        # Mask for heterogeneous sizes
        if self.max_adapter_size != min(self.adapter_sizes):
            size_mask = self._create_size_mask(valid_adapter_ids)
            h = h * size_mask.float()

        # Activation (GELU is common for modern adapters)
        h = torch.nn.functional.gelu(h)

        # Up projection: h @ W_up^T + b_up
        adapter_out = torch.einsum('na,nha->nh', h, up_w) + up_b

        # Apply with residual connection
        output = x.clone()
        output[valid_indices] = x_valid + adapter_out

        return output


# ============================================================================
# Benchmark Configuration (Multi-tenant serving scenario)
# ============================================================================

import os
import sys
