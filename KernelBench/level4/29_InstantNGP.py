"""
Instant-NGP Neural Radiance Fields Model

Implements Instant-NGP for fast neural rendering:
- Multi-resolution hash encoding
- Compact MLP for density and color
- Occupancy grid acceleration

Variants from Table 5:
- Instant-NGP-Base: Standard Instant-NGP

This model uses level1 operators from KernelBench.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, Tuple, List

# Import level1 operators (used directly - no wrapping needed)
from ..level1.activations._1_ReLU import Model as ReLU
from ..level1.activations._3_Sigmoid import Model as Sigmoid
from ..level1.matmul._1_MatMul import Model as MatMul


# ============================================================================
# Model Variants - configs loaded from config_loader
# ============================================================================

VARIANTS: Dict[str, str] = {
    "Base": "instantngp-base",
    "Large": "instantngp-large",
}


# ============================================================================
# Component Modules (using level1 operators)
# ============================================================================

class HashEncoding(nn.Module):
    """Multi-resolution hash encoding from Instant-NGP."""
    
    def __init__(
        self,
        num_levels: int = 16,
        base_resolution: int = 16,
        log2_hashmap_size: int = 19,
        feature_dim: int = 2,
        growth_factor: float = 2.0,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.feature_dim = feature_dim
        self.log2_hashmap_size = log2_hashmap_size
        self.hashmap_size = 2 ** log2_hashmap_size
        
        # Compute resolutions for each level
        self.resolutions = []
        for i in range(num_levels):
            res = int(base_resolution * (growth_factor ** i))
            self.resolutions.append(res)
        
        # Hash table parameters for each level
        self.embeddings = nn.ModuleList([
            nn.Embedding(self.hashmap_size, feature_dim)
            for _ in range(num_levels)
        ])
        
        # Initialize with small random values
        for emb in self.embeddings:
            nn.init.uniform_(emb.weight, -1e-4, 1e-4)
        
        # Primes for spatial hashing
        self.primes = [1, 2654435761, 805459861]

    def hash(self, coords: torch.Tensor, level: int) -> torch.Tensor:
        """Spatial hash function."""
        # coords: (..., 3) integer coordinates
        h = torch.zeros(coords.shape[:-1], dtype=torch.long, device=coords.device)
        for i in range(3):
            h = h ^ (coords[..., i].long() * self.primes[i])
        return h % self.hashmap_size

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-resolution hash encoding.
        
        Args:
            positions: (..., 3) positions normalized to [0, 1]
            
        Returns:
            features: (..., num_levels * feature_dim)
        """
        features = []
        
        for level, resolution in enumerate(self.resolutions):
            # Scale to grid resolution
            scaled = positions * resolution
            
            # Get corner vertices
            floor = torch.floor(scaled).long()
            
            # Compute interpolation weights
            weights = scaled - floor.float()
            
            # Trilinear interpolation over 8 corners
            corner_features = torch.zeros(*positions.shape[:-1], self.feature_dim, device=positions.device)
            
            for dx in [0, 1]:
                for dy in [0, 1]:
                    for dz in [0, 1]:
                        corner = floor + torch.tensor([dx, dy, dz], device=positions.device)
                        corner = corner % resolution  # Wrap around
                        
                        idx = self.hash(corner, level)
                        feat = self.embeddings[level](idx)
                        
                        # Trilinear weight
                        wx = weights[..., 0] if dx == 1 else (1 - weights[..., 0])
                        wy = weights[..., 1] if dy == 1 else (1 - weights[..., 1])
                        wz = weights[..., 2] if dz == 1 else (1 - weights[..., 2])
                        w = (wx * wy * wz).unsqueeze(-1)
                        
                        corner_features = corner_features + w * feat
            
            features.append(corner_features)
        
        return torch.cat(features, dim=-1)


class NeRFMLP(nn.Module):
    """Compact MLP for density and color prediction using level1 operators."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        
        # Density network
        density_layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            density_layers.append(nn.Linear(in_dim, hidden_dim))
            density_layers.append(ReLU())
            in_dim = hidden_dim
        density_layers.append(nn.Linear(hidden_dim, 16))  # 1 density + 15 features
        self.density_net = nn.Sequential(*density_layers)
        
        # Color network (view-dependent)
        self.color_net = nn.Sequential(
            nn.Linear(16 + 3, hidden_dim),  # features + direction
            ReLU(),
            nn.Linear(hidden_dim, 3),
            Sigmoid(),
        )
        
        self.sigmoid = Sigmoid()

    def forward(
        self,
        encoded: torch.Tensor,
        directions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict density and color.
        
        Args:
            encoded: (..., input_dim) encoded positions
            directions: (..., 3) view directions
            
        Returns:
            density: (..., 1) volume density
            color: (..., 3) RGB color
        """
        x = self.density_net(encoded)
        density = F.softplus(x[..., :1])  # Density is non-negative
        features = x[..., 1:]
        
        if directions is not None:
            color_input = torch.cat([features, directions], dim=-1)
        else:
            color_input = torch.cat([features, torch.zeros_like(encoded[..., :3])], dim=-1)
        
        color = self.color_net(color_input)
        
        return density, color


class VolumeRenderer(nn.Module):
    """Differentiable volume renderer."""
    
    def __init__(self, near: float = 0.0, far: float = 1.0, num_samples: int = 64):
        super().__init__()
        self.near = near
        self.far = far
        self.num_samples = num_samples

    def forward(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        density_fn,
    ) -> torch.Tensor:
        """
        Render rays using volume rendering.
        
        Args:
            origins: (N, 3) ray origins
            directions: (N, 3) ray directions
            density_fn: Function that returns (density, color) for positions
            
        Returns:
            colors: (N, 3) rendered colors
        """
        N = origins.shape[0]
        device = origins.device
        
        # Sample points along rays
        t_vals = torch.linspace(self.near, self.far, self.num_samples, device=device)
        t_vals = t_vals.unsqueeze(0).expand(N, -1)  # (N, num_samples)
        
        # Add noise for anti-aliasing during training
        if self.training:
            mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
            upper = torch.cat([mids, t_vals[..., -1:]], dim=-1)
            lower = torch.cat([t_vals[..., :1], mids], dim=-1)
            t_rand = torch.rand(t_vals.shape, device=device)
            t_vals = lower + (upper - lower) * t_rand
        
        # Compute sample positions
        positions = origins.unsqueeze(1) + t_vals.unsqueeze(-1) * directions.unsqueeze(1)
        # positions: (N, num_samples, 3)
        
        # Query density and color
        flat_positions = positions.reshape(-1, 3)
        flat_directions = directions.unsqueeze(1).expand(-1, self.num_samples, -1).reshape(-1, 3)
        
        density, color = density_fn(flat_positions, flat_directions)
        density = density.reshape(N, self.num_samples)
        color = color.reshape(N, self.num_samples, 3)
        
        # Volume rendering
        dists = t_vals[..., 1:] - t_vals[..., :-1]
        dists = torch.cat([dists, torch.full((N, 1), 1e10, device=device)], dim=-1)
        
        alpha = 1 - torch.exp(-density * dists)
        transmittance = torch.cumprod(1 - alpha + 1e-10, dim=-1)
        transmittance = torch.cat([torch.ones(N, 1, device=device), transmittance[..., :-1]], dim=-1)
        
        weights = alpha * transmittance
        rendered_color = (weights.unsqueeze(-1) * color).sum(dim=1)
        
        return rendered_color


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    Instant-NGP for fast neural radiance fields.
    
    Uses level1 operators from KernelBench:
    - ReLU from level1/activations/1_ReLU
    - Sigmoid from level1/activations/3_Sigmoid
    - MatMul from level1/matmul/1_MatMul
    
    Supports variants: Base, Large (configs loaded from config_loader)
    """
    
    VARIANTS = VARIANTS
    
    @classmethod
    def from_pretrained(cls, variant: str = "Base", operator_level: Optional[OperatorLevel] = None, **kwargs):
        """Create model with config loaded from config_loader."""
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}. Available: {list(VARIANTS.keys())}")
        hf_config = load_hf_config(VARIANTS[variant])
        hf_config.update(kwargs)
        return cls(operator_level=operator_level, **hf_config)
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        operator_level: Optional[OperatorLevel] = None,
        **kwargs
    ):
        num_levels = kwargs.get('num_levels', 16)
        base_resolution = kwargs.get('base_resolution', 16)
        log2_hashmap_size = kwargs.get('log2_hashmap_size', 19)
        hash_table_size = kwargs.get('hash_table_size', 2**log2_hashmap_size)
        feature_dim = kwargs.get('feature_dim', kwargs.get('feature_per_level', 2))
        hidden_dim = kwargs.get('hidden_dim', kwargs.get('mlp_hidden', 64))
        num_layers = kwargs.get('num_layers', kwargs.get('mlp_layers', 2))
        num_samples = kwargs.get('num_samples', 64)
        
        if config is None:
            config = ModelConfig(
                hidden_size=hidden_dim,
            )
        
        super().__init__()
        
        # Hash encoding
        self.encoding = HashEncoding(
            num_levels, base_resolution, log2_hashmap_size, feature_dim
        )
        
        # MLP
        encoding_dim = num_levels * feature_dim
        self.mlp = NeRFMLP(encoding_dim, hidden_dim, num_layers)
        
        # Volume renderer
        self.renderer = VolumeRenderer(num_samples=num_samples)

    def density_color(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query density and color at given positions."""
        # Normalize positions to [0, 1]
        positions_norm = (positions + 1) / 2  # Assuming positions in [-1, 1]
        positions_norm = positions_norm.clamp(0, 1)
        
        # Encode
        encoded = self.encoding(positions_norm)
        
        # Predict
        return self.mlp(encoded, directions)

    def forward(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Render colors for given rays.
        
        Args:
            ray_origins: (N, 3) ray origin points
            ray_directions: (N, 3) ray direction vectors
            
        Returns:
            colors: (N, 3) rendered RGB colors
        """
        return self.renderer(ray_origins, ray_directions, self.density_color)
