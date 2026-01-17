import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Ray Marching (3D/Neural Rendering)

    Used by: NeRF, Neural SDF, Volume Rendering

    Generates sample points along rays for neural rendering.
    Supports stratified, hierarchical, and importance sampling.

    Shapes:
        ray_origins: (batch, 3) ray start points
        ray_directions: (batch, 3) ray directions
        Output: (batch, num_samples, 3) sample positions along rays
    """

    def __init__(self, near: float = 0.1, far: float = 10.0,
                 num_samples: int = 64, perturb: bool = True):
        """
        Initialize ray marcher.

        Args:
            near: Near plane distance
            far: Far plane distance
            num_samples: Number of samples per ray
            perturb: Whether to add random perturbation (stratified sampling)
        """
        super(Model, self).__init__()
        self.near = near
        self.far = far
        self.num_samples = num_samples
        self.perturb = perturb

    def forward(self, ray_origins: torch.Tensor,
                ray_directions: torch.Tensor,
                weights: torch.Tensor = None) -> tuple:
        """
        Generate sample points along rays.

        Args:
            ray_origins: Ray origins (batch, 3)
            ray_directions: Ray directions (batch, 3)
            weights: Optional weights for importance sampling (batch, num_coarse)

        Returns:
            Tuple of:
                - points: Sample positions (batch, num_samples, 3)
                - t_vals: Sample distances (batch, num_samples)
                - deltas: Distances between samples (batch, num_samples)
        """
        batch_size = ray_origins.shape[0]
        device = ray_origins.device

        if weights is None:
            # Uniform/stratified sampling
            t_vals = torch.linspace(self.near, self.far, self.num_samples, device=device)
            t_vals = t_vals.unsqueeze(0).expand(batch_size, -1)

            if self.perturb and self.training:
                # Stratified sampling: add noise within each bin
                mids = 0.5 * (t_vals[:, 1:] + t_vals[:, :-1])
                upper = torch.cat([mids, t_vals[:, -1:]], dim=-1)
                lower = torch.cat([t_vals[:, :1], mids], dim=-1)
                t_rand = torch.rand_like(t_vals)
                t_vals = lower + (upper - lower) * t_rand
        else:
            # Importance sampling based on weights
            # Compute PDF from weights
            weights = weights + 1e-5  # Prevent division by zero
            pdf = weights / weights.sum(dim=-1, keepdim=True)

            # Compute CDF
            cdf = torch.cumsum(pdf, dim=-1)
            cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)

            # Sample uniformly in CDF space
            u = torch.rand(batch_size, self.num_samples, device=device)

            # Invert CDF
            inds = torch.searchsorted(cdf, u, right=True)
            below = (inds - 1).clamp(min=0)
            above = inds.clamp(max=cdf.shape[-1] - 1)

            # Linear interpolation
            cdf_below = torch.gather(cdf, 1, below)
            cdf_above = torch.gather(cdf, 1, above)

            # Original t_vals for the coarse samples
            t_coarse = torch.linspace(self.near, self.far, weights.shape[1] + 1, device=device)
            t_coarse = t_coarse.unsqueeze(0).expand(batch_size, -1)

            t_below = torch.gather(t_coarse, 1, below)
            t_above = torch.gather(t_coarse, 1, above)

            denom = cdf_above - cdf_below
            denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
            t = (u - cdf_below) / denom

            t_vals = t_below + t * (t_above - t_below)
            t_vals, _ = torch.sort(t_vals, dim=-1)

        # Compute 3D points
        # points = origins + t * directions
        points = ray_origins.unsqueeze(1) + t_vals.unsqueeze(-1) * ray_directions.unsqueeze(1)

        # Compute deltas (distances between samples)
        deltas = t_vals[:, 1:] - t_vals[:, :-1]
        # Pad last delta with large value (infinity)
        deltas = torch.cat([deltas, torch.full_like(deltas[:, :1], 1e10)], dim=-1)

        return points, t_vals, deltas


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 4096  # Number of rays
num_samples = 64

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    ray_origins = torch.randn(batch_size, 3, device='cuda')
    ray_directions = F.normalize(torch.randn(batch_size, 3, device='cuda'), dim=-1)
    return [ray_origins, ray_directions]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [0.1, 10.0, num_samples]
