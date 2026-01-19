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
    Deformable Attention (Object Detection)

    Used by: Deformable DETR, DINO, Co-DETR

    Multi-scale deformable attention that attends to a sparse set of
    learnable sampling locations instead of all spatial positions.
    Much more efficient than full attention for high-resolution features.

    Shapes:
        query: (batch, num_queries, hidden_size)
        reference_points: (batch, num_queries, num_levels, 2) normalized coords
        value: (batch, total_spatial, hidden_size) multi-scale features
        spatial_shapes: (num_levels, 2) H, W for each level
        Output: (batch, num_queries, hidden_size)
    """

    def __init__(self, hidden_size: int = 256, num_heads: int = 8,
                 num_levels: int = 4, num_points: int = 4):
        """
        Initialize deformable attention.

        Args:
            hidden_size: Hidden dimension
            num_heads: Number of attention heads
            num_levels: Number of feature map levels
            num_points: Number of sampling points per head per level
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = hidden_size // num_heads

        # Sampling offsets prediction
        self.sampling_offsets = nn.Linear(hidden_size, num_heads * num_levels * num_points * 2)

        # Attention weights
        self.attention_weights = nn.Linear(hidden_size, num_heads * num_levels * num_points)

        # Value projection
        self.value_proj = nn.Linear(hidden_size, hidden_size)

        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.)
        nn.init.constant_(self.sampling_offsets.bias.data, 0.)
        nn.init.constant_(self.attention_weights.weight.data, 0.)
        nn.init.constant_(self.attention_weights.bias.data, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.)

    def forward(self, query: torch.Tensor, reference_points: torch.Tensor,
                value: torch.Tensor, spatial_shapes: torch.Tensor,
                level_start_index: torch.Tensor) -> torch.Tensor:
        """
        Deformable attention forward.

        Args:
            query: Query features (batch, num_queries, hidden_size)
            reference_points: Reference points (batch, num_queries, num_levels, 2)
            value: Multi-scale value features (batch, total_spatial, hidden_size)
            spatial_shapes: Shape of each level (num_levels, 2)
            level_start_index: Start index for each level (num_levels,)

        Returns:
            Output features (batch, num_queries, hidden_size)
        """
        batch_size, num_queries, _ = query.shape
        _, total_spatial, _ = value.shape

        # Project values
        value = self.value_proj(value)
        value = value.view(batch_size, total_spatial, self.num_heads, self.head_dim)

        # Predict sampling offsets
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            batch_size, num_queries, self.num_heads, self.num_levels, self.num_points, 2
        )

        # Predict attention weights
        attention_weights = self.attention_weights(query)
        attention_weights = attention_weights.view(
            batch_size, num_queries, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = attention_weights.view(
            batch_size, num_queries, self.num_heads, self.num_levels, self.num_points
        )

        # Compute sampling locations
        # reference_points: (batch, num_queries, num_levels, 2)
        # sampling_offsets: (batch, num_queries, num_heads, num_levels, num_points, 2)
        offset_normalizer = spatial_shapes.flip(-1).float()  # (num_levels, 2) as (W, H)
        sampling_locations = reference_points[:, :, None, :, None, :] + \
            sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        sampling_locations = sampling_locations.clamp(0, 1)

        # Sample values at computed locations (simplified - actual impl uses custom CUDA)
        # For each level, sample at the computed locations
        output = torch.zeros(batch_size, num_queries, self.num_heads, self.head_dim,
                           device=query.device, dtype=query.dtype)

        for level in range(self.num_levels):
            h, w = spatial_shapes[level]
            start_idx = level_start_index[level]
            end_idx = level_start_index[level + 1] if level < self.num_levels - 1 else total_spatial

            # Get values for this level
            level_value = value[:, start_idx:end_idx, :, :]  # (batch, h*w, heads, head_dim)
            level_value = level_value.view(batch_size, h, w, self.num_heads, self.head_dim)
            level_value = level_value.permute(0, 3, 4, 1, 2)  # (batch, heads, head_dim, h, w)

            # Get sampling locations for this level
            level_locs = sampling_locations[:, :, :, level, :, :]  # (batch, queries, heads, points, 2)

            # Grid sample (2D interpolation)
            for head in range(self.num_heads):
                for point in range(self.num_points):
                    # Normalize to [-1, 1] for grid_sample
                    grid = level_locs[:, :, head, point, :] * 2 - 1  # (batch, queries, 2)
                    grid = grid.view(batch_size, num_queries, 1, 2)

                    # Sample
                    sampled = F.grid_sample(
                        level_value[:, head, :, :, :],  # (batch, head_dim, h, w)
                        grid,
                        mode='bilinear',
                        padding_mode='zeros',
                        align_corners=False
                    )  # (batch, head_dim, queries, 1)

                    sampled = sampled.squeeze(-1).transpose(1, 2)  # (batch, queries, head_dim)

                    # Weight and accumulate
                    weight = attention_weights[:, :, head, level, point:point+1]  # (batch, queries, 1)
                    output[:, :, head, :] += sampled * weight

        # Reshape and project output
        output = output.view(batch_size, num_queries, self.hidden_size)
        output = self.output_proj(output)

        return output


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    # High-throughput: Deformable DETR batched inference (standard resolution)
    {"batch_size": 16, "num_queries": 300, "hidden_size": 256, "num_heads": 8, "num_levels": 4, "num_points": 4, "spatial_shapes": [(80, 80), (40, 40), (20, 20), (10, 10)]},
    # High-throughput: DINO high-volume detection (more queries for dense prediction)
    {"batch_size": 8, "num_queries": 900, "hidden_size": 256, "num_heads": 8, "num_levels": 4, "num_points": 4, "spatial_shapes": [(100, 100), (50, 50), (25, 25), (13, 13)]},
    # Low-latency: Co-DETR high-resolution single image (1280x1280 input)
    {"batch_size": 1, "num_queries": 900, "hidden_size": 256, "num_heads": 8, "num_levels": 4, "num_points": 4, "spatial_shapes": [(160, 160), (80, 80), (40, 40), (20, 20)]},
    # Low-latency: DINO-5scale with extra feature level (high-resolution)
    {"batch_size": 2, "num_queries": 900, "hidden_size": 256, "num_heads": 8, "num_levels": 5, "num_points": 4, "spatial_shapes": [(200, 200), (100, 100), (50, 50), (25, 25), (13, 13)]},
    # Balanced: Deformable DETR standard detection
    {"batch_size": 4, "num_queries": 300, "hidden_size": 256, "num_heads": 8, "num_levels": 4, "num_points": 4, "spatial_shapes": [(80, 80), (40, 40), (20, 20), (10, 10)]},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("detection", "9_DeformableAttention")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]

    query = DISTRIBUTIONS[dist_name]((p["batch_size"], p["num_queries"], p["hidden_size"]), dtype=dtype, device=device)

    # Reference points (normalized 0-1)
    reference_points = torch.rand(p["batch_size"], p["num_queries"], p["num_levels"], 2, dtype=dtype, device=device)

    # Compute total spatial size
    spatial_shapes_list = p["spatial_shapes"]
    total_spatial = sum(h * w for h, w in spatial_shapes_list)
    value = DISTRIBUTIONS[dist_name]((p["batch_size"], total_spatial, p["hidden_size"]), dtype=dtype, device=device)

    spatial_shapes = torch.tensor(spatial_shapes_list, device=device, dtype=torch.long)

    # Level start indices
    level_start_index = torch.zeros(p["num_levels"], dtype=torch.long, device=device)
    for i in range(1, p["num_levels"]):
        h, w = spatial_shapes_list[i-1]
        level_start_index[i] = level_start_index[i-1] + h * w

    return [query, reference_points, value, spatial_shapes, level_start_index]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["hidden_size"], p["num_heads"], p["num_levels"], p["num_points"]]
