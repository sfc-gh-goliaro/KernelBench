import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Volume Rendering (3D/Neural Rendering)

    Used by: NeRF, 3D Gaussian Splatting, Neural Volumes

    Integrates colors and densities along rays using alpha compositing.
    Converts point samples to final pixel colors.

    Shapes:
        colors: (batch, num_samples, 3) RGB at each sample
        densities: (batch, num_samples) density at each sample
        deltas: (batch, num_samples) distance between samples
        Output: (batch, 3) rendered RGB color per ray
    """

    def __init__(self, white_background: bool = False):
        """
        Initialize volume renderer.

        Args:
            white_background: Whether to composite over white background
        """
        super(Model, self).__init__()
        self.white_background = white_background

    def forward(self, colors: torch.Tensor, densities: torch.Tensor,
                deltas: torch.Tensor) -> tuple:
        """
        Volume render colors along rays.

        Args:
            colors: RGB colors at samples (batch, num_samples, 3)
            densities: Densities at samples (batch, num_samples)
            deltas: Distances between samples (batch, num_samples)

        Returns:
            Tuple of:
                - rgb: Rendered colors (batch, 3)
                - depth: Expected depth (batch,)
                - weights: Sample weights (batch, num_samples)
        """
        # Compute alpha from density: alpha = 1 - exp(-sigma * delta)
        alpha = 1.0 - torch.exp(-densities * deltas)

        # Compute transmittance: T_i = prod_{j<i}(1 - alpha_j)
        # Using exclusive cumprod
        one_minus_alpha = 1.0 - alpha + 1e-10
        transmittance = torch.cumprod(one_minus_alpha, dim=-1)
        # Shift for exclusive product
        transmittance = torch.cat([
            torch.ones_like(transmittance[:, :1]),
            transmittance[:, :-1]
        ], dim=-1)

        # Compute weights: w_i = T_i * alpha_i
        weights = transmittance * alpha

        # Render color: C = sum(w_i * c_i)
        rgb = (weights.unsqueeze(-1) * colors).sum(dim=1)

        # Compute expected depth
        # Create sample positions (assuming uniform sampling from 0)
        num_samples = deltas.shape[1]
        t_vals = torch.cumsum(deltas, dim=-1) - deltas / 2  # Midpoints
        depth = (weights * t_vals).sum(dim=-1)

        # Background
        if self.white_background:
            acc = weights.sum(dim=-1, keepdim=True)
            rgb = rgb + (1 - acc) * 1.0  # White background

        return rgb, depth, weights


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 4096  # Number of rays
num_samples = 128  # Samples per ray

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    colors = torch.sigmoid(torch.randn(batch_size, num_samples, 3, device='cuda'))
    densities = F.softplus(torch.randn(batch_size, num_samples, device='cuda'))
    deltas = torch.rand(batch_size, num_samples, device='cuda') * 0.1

    return [colors, densities, deltas]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [True]  # white_background
