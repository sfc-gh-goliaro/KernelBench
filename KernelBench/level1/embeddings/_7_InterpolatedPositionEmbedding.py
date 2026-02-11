"""
Interpolated 2D Position Embedding

Used by: Qwen3-VL (vision encoder)

Learnable 2D position embeddings stored as an nn.Embedding table over a fixed
grid (e.g. 48x48 = 2304 positions). At runtime, bilinear interpolation maps
arbitrary (h, w) patch grids to the learned grid, enabling resolution-flexible
position encoding.

The operator also reorders patches into spatial-merge order: within each
merge_size x merge_size block, patches are grouped contiguously. This matches
the ordering expected by downstream PatchMerger modules.

Pipeline:
  1. For each image's (h, w) grid, compute bilinear interpolation indices and
     weights mapping to the learned num_grid_per_side x num_grid_per_side grid.
  2. Gather 4 corner embeddings per position, weight them, and sum.
  3. Repeat along the temporal dimension.
  4. Permute into spatial-merge block order.
  5. Concatenate across all images.

Shapes:
    Input: grid_thw (num_images, 3) -- temporal, height, width per image
    Output: (total_patches, hidden_size) position embeddings
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Interpolated 2D Position Embedding.

    Owns an nn.Embedding table of size (num_grid_per_side^2, hidden_size)
    representing a fixed 2D grid of learned position vectors. At forward time,
    bilinear interpolation maps variable-resolution patch grids to this fixed
    grid, and the result is reordered into spatial-merge block order.

    Args:
        hidden_size: Dimension of each position embedding vector.
        num_position_embeddings: Total number of grid positions in the learned
            table. Must be a perfect square. Default: 2304 (48x48).
        spatial_merge_size: Number of patches merged along each spatial
            dimension by the downstream PatchMerger. Controls the block
            reordering in the output. Default: 2.
    """

    def __init__(
        self,
        hidden_size: int,
        num_position_embeddings: int = 2304,
        spatial_merge_size: int = 2,
    ):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.num_grid_per_side = int(num_position_embeddings ** 0.5)
        self.spatial_merge_size = spatial_merge_size

        self.pos_embed = nn.Embedding(num_position_embeddings, hidden_size)

    def forward(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Compute interpolated position embeddings for vision patches.

        For each image described by (temporal, height, width), maps the (h, w)
        patch positions onto the learned grid via bilinear interpolation, repeats
        along the temporal axis, and reorders into spatial-merge block order.

        Args:
            grid_thw: (num_images, 3) tensor of (temporal, height, width) per
                image. Height and width are in units of patches (after Conv3d
                patch embedding).

        Returns:
            Position embeddings of shape (total_patches, hidden_size), where
            total_patches = sum(t_i * h_i * w_i) across all images. Patches
            are ordered by spatial-merge blocks within each image.
        """
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        device = self.pos_embed.weight.device

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=device
        )
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        # Split per-image and reorder into spatial-merge block order
        patch_pos_embeds = patch_pos_embeds.split(
            [h * w for h, w in zip(grid_hs, grid_ws)]
        )

        merge_size = self.spatial_merge_size
        patch_pos_embeds_permute = []
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(
                    t, h // merge_size, merge_size, w // merge_size, merge_size, -1
                )
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds
