"""
Vision 2D Rotary Position Embedding

Used by: Qwen2-VL, Qwen3-VL, Qwen3-Omni (vision encoders)

Computes 2D spatial rotary position embeddings for vision transformer patches.
Given a grid of patches described by (temporal, height, width) per image, builds
2D (h, w) position IDs with spatial-merge-aware ordering, then converts them to
rotary cos/sin embeddings using precomputed inverse frequencies.

The operator owns inv_freq and handles the full pipeline:
  grid_thw -> 2D position IDs -> frequency table lookup -> (cos, sin)

Shapes:
    Input: grid_thw (num_images, 3) -- temporal, height, width per image
    Output: (cos, sin) each of shape (total_patches, head_dim)
"""

import torch
import torch.nn as nn
from typing import Tuple


class Model(nn.Module):
    """
    Vision 2D Rotary Position Embedding.

    Computes spatial rotary embeddings for vision patches. Each patch gets a 2D
    position (h, w) within its image, ordered by spatial merge blocks. The
    resulting frequencies are gathered from a precomputed table and returned
    as (cos, sin) pairs ready for application via rotate_half.

    Args:
        dim: Frequency dimension. Typically head_dim // 2, since the 2D RoPE
             uses dim // 2 frequency pairs for height and dim // 2 for width,
             concatenated to cover head_dim // 2 total, then doubled to head_dim.
        theta: Base for the frequency computation (default: 10000.0).
    """

    def __init__(self, dim: int, theta: float = 10000.0):
        super(Model, self).__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        # Store float32 copy that survives .to(bfloat16)
        self._inv_freq_float32 = inv_freq
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply(self, fn):
        """Override to keep inv_freq in float32 when model dtype changes."""
        super()._apply(fn)
        if hasattr(self, '_inv_freq_float32'):
            target_device = self.inv_freq.device
            self._inv_freq_float32 = self._inv_freq_float32.to(device=target_device)
            self.inv_freq = self._inv_freq_float32.clone()
        return self

    def _get_inv_freq(self) -> torch.Tensor:
        """Return inv_freq in float32, updating device if needed."""
        if self._inv_freq_float32.device != self.inv_freq.device:
            self._inv_freq_float32 = self._inv_freq_float32.to(device=self.inv_freq.device)
        return self._inv_freq_float32

    def _build_position_ids(
        self, grid_thw: torch.Tensor, spatial_merge_size: int,
    ) -> torch.Tensor:
        """
        Build 2D (h, w) position IDs for all patches across all images.

        Positions are ordered within spatial merge blocks: patches within the
        same merge block are grouped together, matching how PatchMerger will
        later merge them.

        Args:
            grid_thw: (num_images, 3) -- temporal, height, width per image
            spatial_merge_size: number of patches merged along each spatial dim

        Returns:
            pos_ids: (total_patches, 2) -- (row, col) position per patch
        """
        device = grid_thw.device
        pos_ids_list = []

        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            merged_h = h // spatial_merge_size
            merged_w = w // spatial_merge_size

            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(spatial_merge_size, device=device)
            intra_col = torch.arange(spatial_merge_size, device=device)

            # (merged_h, merged_w, merge_size, merge_size) ordering
            row_idx = (
                block_rows[:, None, None, None] * spatial_merge_size
                + intra_row[None, None, :, None]
            )
            col_idx = (
                block_cols[None, :, None, None] * spatial_merge_size
                + intra_col[None, None, None, :]
            )
            row_idx = row_idx.expand(merged_h, merged_w, spatial_merge_size, spatial_merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, spatial_merge_size, spatial_merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)  # (h*w, 2)
            if t > 1:
                coords = coords.repeat(t, 1)
            pos_ids_list.append(coords)

        return torch.cat(pos_ids_list, dim=0)

    def forward(
        self,
        grid_thw: torch.Tensor,
        spatial_merge_size: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute vision rotary position embeddings.

        Args:
            grid_thw: (num_images, 3) tensor of (temporal, height, width) per image.
            spatial_merge_size: merge factor along each spatial dimension.

        Returns:
            Tuple of (cos, sin), each of shape (total_patches, head_dim), where
            head_dim = dim * 2 (the doubled concatenation of h and w frequencies).
        """
        inv_freq = self._get_inv_freq()

        # Build frequency table: (max_grid_size, dim // 2)
        max_grid_size = int(grid_thw[:, 1:].max().item())
        seq = torch.arange(max_grid_size, device=inv_freq.device, dtype=torch.float32)
        freq_table = torch.outer(seq, inv_freq)  # (max_grid_size, dim // 2)

        # Build 2D position IDs and gather frequencies
        pos_ids = self._build_position_ids(grid_thw, spatial_merge_size)
        # pos_ids: (total_patches, 2), freq_table: (max_grid_size, dim//2)
        # Gather: (total_patches, 2, dim//2) -> flatten -> (total_patches, dim)
        rotary_pos_emb = freq_table[pos_ids].flatten(1)

        # Double for rotate_half: (total_patches, dim*2) = (total_patches, head_dim)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        return emb.cos(), emb.sin()
