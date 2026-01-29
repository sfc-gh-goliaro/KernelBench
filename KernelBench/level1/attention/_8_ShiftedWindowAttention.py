import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Shifted Window Attention with Paged Feature Cache

    Used by: Swin Transformer

    Shifted window attention with cyclic shift for cross-window
    connections. Uses paged feature cache where feature patches
    are stored in non-contiguous blocks accessed via a page table.
    
    NOTE: This operator takes pre-computed Q, K, V tensors (already in window format).
    The QKV projection should be done separately using a Linear operator.

    Shapes:
        q: (num_windows * batch, window_size * window_size, num_heads, head_dim) - query
        k: (num_windows * batch, window_size * window_size, num_heads, head_dim) - key
        v: (num_windows * batch, window_size * window_size, num_heads, head_dim) - value
        attn_mask: (num_windows, window_size^2, window_size^2) - attention mask for shifted windows
        Output: (num_windows * batch, window_size * window_size, dim)
    """

    def __init__(self, dim: int, window_size: int = 7, num_heads: int = 8):
        """
        Initialize shifted window attention.

        Args:
            dim: Input dimension (num_heads * head_dim)
            window_size: Size of attention window
            num_heads: Number of attention heads
        """
        super(Model, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        # Create relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Shifted window attention forward pass.

        Args:
            q: Query (num_windows * batch, num_heads, window_size^2, head_dim)
            k: Key (num_windows * batch, num_heads, window_size^2, head_dim)
            v: Value (num_windows * batch, num_heads, window_size^2, head_dim)
            attn_mask: Optional attention mask for shifted windows (num_windows, window_size^2, window_size^2)

        Returns:
            Output tensor (num_windows * batch, num_heads, window_size^2, head_dim)
        """
        B_windows, num_heads, seq_len, head_dim = q.shape

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        # Apply attention mask if provided
        if attn_mask is not None:
            num_windows = attn_mask.shape[0]
            attn = attn.view(-1, num_windows, self.num_heads, seq_len, seq_len)
            attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(B_windows, self.num_heads, seq_len, seq_len)

        attn = F.softmax(attn, dim=-1)

        output = attn @ v

        return output  # (num_windows * batch, heads, window_size^2, head_dim)


# ============================================================================
# Benchmark Configuration
# ============================================================================
