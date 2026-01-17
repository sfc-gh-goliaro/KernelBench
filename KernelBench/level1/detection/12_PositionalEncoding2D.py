import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    2D Positional Encoding (Object Detection)

    Used by: DETR, Deformable DETR, ViT-Det

    Generates 2D sinusoidal positional embeddings for image feature maps.
    Encodes both x and y positions using sine and cosine functions.

    Shapes:
        feature_map: (batch, channels, height, width)
        Output: (batch, hidden_size, height, width) positional encoding
    """

    def __init__(self, hidden_size: int = 256, temperature: float = 10000.0,
                 normalize: bool = True, scale: float = 2 * math.pi):
        """
        Initialize 2D positional encoding.

        Args:
            hidden_size: Output embedding dimension (must be divisible by 2)
            temperature: Temperature for frequency scaling
            normalize: Whether to normalize positions to [0, 1]
            scale: Scale factor for normalized positions
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale

        assert hidden_size % 2 == 0, "hidden_size must be divisible by 2"

    def forward(self, feature_map: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Generate 2D positional encodings.

        Args:
            feature_map: Input feature map (batch, channels, height, width)
            mask: Optional mask (batch, height, width), True for padded positions

        Returns:
            Positional encoding (batch, hidden_size, height, width)
        """
        batch_size, _, height, width = feature_map.shape
        device = feature_map.device
        dtype = feature_map.dtype

        if mask is None:
            mask = torch.zeros(batch_size, height, width, dtype=torch.bool, device=device)

        # Create position grids
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=dtype)
        x_embed = not_mask.cumsum(2, dtype=dtype)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # Frequency bands
        dim_t = self.hidden_size // 2
        dim_indices = torch.arange(dim_t, dtype=dtype, device=device)
        dim_t_scaled = self.temperature ** (2 * (dim_indices // 2) / dim_t)

        # Compute positional encodings
        pos_x = x_embed[:, :, :, None] / dim_t_scaled  # (batch, h, w, dim_t)
        pos_y = y_embed[:, :, :, None] / dim_t_scaled

        # Apply sin to even indices, cos to odd indices
        pos_x = torch.stack([pos_x[:, :, :, 0::2].sin(),
                            pos_x[:, :, :, 1::2].cos()], dim=4).flatten(3)
        pos_y = torch.stack([pos_y[:, :, :, 0::2].sin(),
                            pos_y[:, :, :, 1::2].cos()], dim=4).flatten(3)

        # Concatenate x and y encodings
        pos = torch.cat([pos_y, pos_x], dim=3)  # (batch, h, w, hidden_size)
        pos = pos.permute(0, 3, 1, 2)  # (batch, hidden_size, h, w)

        return pos


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
channels = 256
height = 80
width = 80
hidden_size = 256

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    feature_map = torch.randn(batch_size, channels, height, width, device='cuda')
    return [feature_map]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [hidden_size]
