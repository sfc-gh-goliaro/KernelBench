import os
import sys
import torch
import torch.nn as nn
from typing import Tuple

class Model(nn.Module):
    """
    Multi-Tenant QLoRA with Quantized Base Model

    Used by: S-LoRA with quantization, vLLM quantized multi-LoRA

    Multi-tenant LoRA serving with 4-bit quantized base model weights.
    Enables serving many concurrent LoRA adapters on memory-constrained GPUs
    by keeping base weights quantized while adapters remain in FP16/BF16.

    Key features:
    - 4-bit quantized base weights (NF4 or INT4)
    - FP16 LoRA adapters for each tenant
    - Token-level adapter assignment
    - Batched computation across adapters

    Shapes:
        x: (total_tokens, in_features) - flattened batch of tokens
        adapter_ids: (total_tokens,) - adapter index per token
        Output: (total_tokens, out_features)
    """

    def __init__(self, in_features: int, out_features: int, num_adapters: int = 100,
                 rank: int = 16, alpha: float = 16.0, group_size: int = 128):
        """
        Initialize multi-tenant QLoRA layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            num_adapters: Maximum number of concurrent LoRA adapters
            rank: LoRA rank
            alpha: LoRA alpha for scaling
            group_size: Quantization group size for base weights
        """
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_adapters = num_adapters
        self.rank = rank
        self.scaling = alpha / rank
        self.group_size = group_size

        # 4-bit quantized base weights
        # Packed format: 2 x 4-bit values per uint8
        num_groups = (in_features + group_size - 1) // group_size
        self.register_buffer('base_qweight',
            torch.zeros(out_features, in_features // 2, dtype=torch.uint8))
        self.register_buffer('base_scales',
            torch.ones(out_features, num_groups, dtype=torch.float16))
        self.register_buffer('base_zeros',
            torch.zeros(out_features, num_groups, dtype=torch.float16))

        # Multi-tenant LoRA adapter pool (FP16)
        self.lora_A = nn.Parameter(torch.zeros(num_adapters, rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(num_adapters, out_features, rank))

        # Initialize adapters
        for i in range(num_adapters):
            nn.init.kaiming_uniform_(self.lora_A.data[i])

    def _dequantize_base_vectorized(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply quantized base weights to input using fused dequant-matmul.

        Simulates the fused kernel that dequantizes on-the-fly during GEMM.

        Args:
            x: Input tensor (total_tokens, in_features)

        Returns:
            Output tensor (total_tokens, out_features)
        """
        # Unpack 4-bit values
        w_low = (self.base_qweight & 0x0F).to(torch.float16)
        w_high = (self.base_qweight >> 4).to(torch.float16)
        # (out_features, in_features // 2) each

        # Interleave to get full weights
        weights = torch.stack([w_low, w_high], dim=-1).view(
            self.out_features, self.in_features
        )

        # Dequantize with group-wise scales and zeros
        # Reshape for group-wise operations
        weights = weights.view(self.out_features, -1, self.group_size)
        scales = self.base_scales.unsqueeze(-1)  # (out, groups, 1)
        zeros = self.base_zeros.unsqueeze(-1)    # (out, groups, 1)

        weights = (weights - zeros) * scales
        weights = weights.view(self.out_features, self.in_features)

        # Matrix multiply
        return torch.matmul(x, weights.T)

    def forward(self, x: torch.Tensor, adapter_ids: torch.Tensor) -> torch.Tensor:
        """
        Multi-tenant QLoRA forward pass.

        Combines quantized base model computation with batched LoRA adapters.

        Args:
            x: Input tokens (total_tokens, in_features)
            adapter_ids: Adapter index per token (total_tokens,)
                        Use -1 for tokens that don't use any adapter.

        Returns:
            Output tensor (total_tokens, out_features)
        """
        # Quantized base model forward
        base_output = self._dequantize_base_vectorized(x)

        # Handle adapter computation
        valid_mask = adapter_ids >= 0
        if not valid_mask.any():
            return base_output

        # Get valid tokens and their adapters
        valid_indices = torch.where(valid_mask)[0]
        valid_adapter_ids = adapter_ids[valid_indices]
        x_valid = x[valid_indices]

        # Gather adapters for each token
        A_gathered = self.lora_A[valid_adapter_ids]  # (num_valid, rank, in_features)
        B_gathered = self.lora_B[valid_adapter_ids]  # (num_valid, out_features, rank)

        # Batched LoRA computation
        intermediate = torch.einsum('nh,nrh->nr', x_valid, A_gathered)
        lora_output = torch.einsum('nr,nor->no', intermediate, B_gathered)

        # Combine outputs
        output = base_output.clone()
        output[valid_indices] = output[valid_indices] + self.scaling * lora_output

        return output
