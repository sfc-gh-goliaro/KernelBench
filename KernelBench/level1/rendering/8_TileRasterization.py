import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Tile-Based Rasterization (3D/Neural Rendering)

    Used by: 3D Gaussian Splatting, Real-time Neural Rendering

    Divides screen into tiles and processes primitives per-tile for
    efficient parallel rendering. Key optimization for Gaussian splatting.

    Shapes:
        positions_2d: (num_primitives, 2) screen-space positions
        radii: (num_primitives,) bounding radii
        colors: (num_primitives, 3) RGB colors
        opacities: (num_primitives,) alpha values
        Output: (height, width, 3) rendered image
    """

    def __init__(self, image_height: int = 512, image_width: int = 512,
                 tile_size: int = 16):
        """
        Initialize tile-based rasterizer.

        Args:
            image_height: Output image height
            image_width: Output image width
            tile_size: Size of each tile in pixels
        """
        super(Model, self).__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.tile_size = tile_size

        self.tiles_x = (image_width + tile_size - 1) // tile_size
        self.tiles_y = (image_height + tile_size - 1) // tile_size

    def assign_to_tiles(self, positions_2d: torch.Tensor,
                       radii: torch.Tensor) -> tuple:
        """Assign primitives to overlapping tiles."""
        device = positions_2d.device
        num_primitives = positions_2d.shape[0]

        # Compute tile bounds for each primitive
        min_x = (positions_2d[:, 0] - radii) / self.tile_size
        max_x = (positions_2d[:, 0] + radii) / self.tile_size
        min_y = (positions_2d[:, 1] - radii) / self.tile_size
        max_y = (positions_2d[:, 1] + radii) / self.tile_size

        # Clamp to valid tile range
        min_x = min_x.floor().long().clamp(0, self.tiles_x - 1)
        max_x = max_x.ceil().long().clamp(0, self.tiles_x - 1)
        min_y = min_y.floor().long().clamp(0, self.tiles_y - 1)
        max_y = max_y.ceil().long().clamp(0, self.tiles_y - 1)

        return min_x, max_x, min_y, max_y

    def forward(self, positions_2d: torch.Tensor, radii: torch.Tensor,
                colors: torch.Tensor, opacities: torch.Tensor,
                depths: torch.Tensor) -> torch.Tensor:
        """
        Tile-based rasterization.

        Args:
            positions_2d: Screen positions (num_primitives, 2)
            radii: Bounding radii (num_primitives,)
            colors: RGB colors (num_primitives, 3)
            opacities: Alpha values (num_primitives,)
            depths: Depth values for sorting (num_primitives,)

        Returns:
            Rendered image (height, width, 3)
        """
        device = positions_2d.device
        num_primitives = positions_2d.shape[0]

        # Initialize output
        image = torch.zeros(self.image_height, self.image_width, 3, device=device)
        accumulated_alpha = torch.zeros(self.image_height, self.image_width, device=device)

        # Sort primitives by depth
        depth_order = torch.argsort(depths)

        # Assign primitives to tiles
        min_x, max_x, min_y, max_y = self.assign_to_tiles(positions_2d, radii)

        # Process each tile
        for ty in range(self.tiles_y):
            for tx in range(self.tiles_x):
                # Find primitives touching this tile
                in_tile = (min_x <= tx) & (max_x >= tx) & (min_y <= ty) & (max_y >= ty)
                tile_indices = torch.where(in_tile)[0]

                if len(tile_indices) == 0:
                    continue

                # Sort tile primitives by depth
                tile_depths = depths[tile_indices]
                tile_order = torch.argsort(tile_depths)
                tile_indices = tile_indices[tile_order]

                # Tile pixel coordinates
                tile_x_start = tx * self.tile_size
                tile_y_start = ty * self.tile_size
                tile_x_end = min(tile_x_start + self.tile_size, self.image_width)
                tile_y_end = min(tile_y_start + self.tile_size, self.image_height)

                # Create pixel grid for this tile
                py = torch.arange(tile_y_start, tile_y_end, device=device).float()
                px = torch.arange(tile_x_start, tile_x_end, device=device).float()
                yy, xx = torch.meshgrid(py, px, indexing='ij')

                # Process primitives in depth order
                tile_image = image[tile_y_start:tile_y_end, tile_x_start:tile_x_end]
                tile_alpha = accumulated_alpha[tile_y_start:tile_y_end, tile_x_start:tile_x_end]

                for idx in tile_indices[:100]:  # Limit for efficiency
                    cx, cy = positions_2d[idx]
                    r = radii[idx]
                    color = colors[idx]
                    opacity = opacities[idx]

                    # Compute Gaussian weight
                    dx = xx - cx
                    dy = yy - cy
                    dist_sq = dx**2 + dy**2
                    weight = torch.exp(-0.5 * dist_sq / (r**2 + 1e-6)) * opacity

                    # Alpha compositing
                    alpha = weight * (1 - tile_alpha)
                    tile_image += alpha.unsqueeze(-1) * color
                    tile_alpha += alpha

                image[tile_y_start:tile_y_end, tile_x_start:tile_x_end] = tile_image
                accumulated_alpha[tile_y_start:tile_y_end, tile_x_start:tile_x_end] = tile_alpha

        return image


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"num_primitives": 10000, "image_height": 512, "image_width": 512, "tile_size": 16},
    # 3DGS: Standard tile-based rendering
    {"num_primitives": 50000, "image_height": 800, "image_width": 800, "tile_size": 16},
    # 3DGS-HD: High-resolution rendering with more primitives
    {"num_primitives": 200000, "image_height": 1080, "image_width": 1920, "tile_size": 16},
    # 3DGS-Mobile: Smaller tiles for mobile GPU efficiency
    {"num_primitives": 25000, "image_height": 720, "image_width": 1280, "tile_size": 8},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("rendering", "8_TileRasterization")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    positions_2d = DISTRIBUTIONS[dist_name]((p["num_primitives"], 2), dtype=dtype, device=device)
    radii = DISTRIBUTIONS[dist_name]((p["num_primitives"]), dtype=dtype, device=device)
    depths = DISTRIBUTIONS[dist_name]((p["num_primitives"]), dtype=dtype, device=device)
    return [positions_2d, radii, colors, opacities, depths]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["image_height"], p["image_width"], p["tile_size"]]
