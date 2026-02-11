"""
3D Gaussian Splatting Model

Implements 3D Gaussian Splatting for novel view synthesis:
- Gaussian primitive representation
- Differentiable rasterization
- Spherical harmonics for view-dependent color

Variants from Table 5:
- 3DGS-Base: Standard 3D Gaussian Splatting

This model uses level1 operators from KernelBench.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, Tuple

# Import level1 operators (used directly - no wrapping needed)
from ..level1.activations._3_Sigmoid import Model as Sigmoid
from ..level1.activations._5_Softmax import Model as Softmax
from ..level1.matmul._1_MatMul import Model as MatMul


# ============================================================================
# Model Variants - configs loaded from config_loader
# ============================================================================

VARIANTS: Dict[str, str] = {
    "1M": "3dgs-1m",
    "5M": "3dgs-5m",
}


# ============================================================================
# Component Modules (using level1 operators)
# ============================================================================

def build_rotation_from_quaternion(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to rotation matrix."""
    # q: (..., 4) quaternion [w, x, y, z]
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    
    R = torch.stack([
        1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w,
        2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w,
        2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y,
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    
    return R


def build_covariance_from_scaling_rotation(scaling: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Build 3D covariance matrix from scaling and rotation."""
    # scaling: (N, 3), rotation: (N, 4) quaternion
    R = build_rotation_from_quaternion(rotation)  # (N, 3, 3)
    S = torch.diag_embed(scaling)  # (N, 3, 3)
    
    # Covariance = R @ S @ S^T @ R^T
    RS = torch.bmm(R, S)
    cov = torch.bmm(RS, RS.transpose(-1, -2))
    
    return cov


class SphericalHarmonics(nn.Module):
    """Spherical harmonics for view-dependent color."""
    def __init__(self, degree: int = 3):
        super().__init__()
        self.degree = degree
        self.num_coeffs = (degree + 1) ** 2
    
    def forward(self, sh_coeffs: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        """
        Evaluate spherical harmonics.
        
        Args:
            sh_coeffs: (N, 3, num_coeffs) SH coefficients
            directions: (N, 3) view directions
            
        Returns:
            colors: (N, 3) RGB colors
        """
        # Simplified: only use first few degrees
        x, y, z = directions.unbind(-1)
        
        # Degree 0
        result = sh_coeffs[:, :, 0] * 0.28209479177387814
        
        if self.degree >= 1 and sh_coeffs.shape[-1] >= 4:
            # Degree 1
            result = result + sh_coeffs[:, :, 1] * 0.4886025119029199 * y
            result = result + sh_coeffs[:, :, 2] * 0.4886025119029199 * z
            result = result + sh_coeffs[:, :, 3] * 0.4886025119029199 * x
        
        return result


class GaussianRenderer(nn.Module):
    """Differentiable Gaussian splatting renderer using level1 operators."""
    def __init__(self, image_height: int, image_width: int):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.matmul = MatMul()
        self.sigmoid = Sigmoid()
        self.sh = SphericalHarmonics()

    def project_gaussians(
        self,
        means3d: torch.Tensor,
        covs3d: torch.Tensor,
        view_matrix: torch.Tensor,
        proj_matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project 3D Gaussians to 2D screen space."""
        N = means3d.shape[0]
        
        # Transform to camera space
        means_homo = F.pad(means3d, (0, 1), value=1.0)  # (N, 4)
        means_cam = self.matmul(means_homo, view_matrix.T)[:, :3]  # (N, 3)
        
        # Project to clip space
        means_clip = self.matmul(means_homo, proj_matrix.T)  # (N, 4)
        means_ndc = means_clip[:, :2] / (means_clip[:, 3:4] + 1e-6)  # (N, 2)
        
        # Convert to screen space
        means2d = torch.stack([
            (means_ndc[:, 0] + 1) * 0.5 * self.image_width,
            (means_ndc[:, 1] + 1) * 0.5 * self.image_height,
        ], dim=-1)
        
        # Approximate 2D covariance from 3D (simplified)
        # Full implementation would use Jacobian of projection
        depth = means_cam[:, 2:3].clamp(min=0.1)
        covs2d = covs3d[:, :2, :2] / (depth.unsqueeze(-1) ** 2)
        
        return means2d, covs2d, depth.squeeze(-1)

    def forward(
        self,
        means3d: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
        sh_coeffs: torch.Tensor,
        view_matrix: torch.Tensor,
        proj_matrix: torch.Tensor,
        camera_center: torch.Tensor,
    ) -> torch.Tensor:
        """
        Render image using Gaussian splatting.
        
        Returns:
            image: (H, W, 3) rendered image
        """
        N = means3d.shape[0]
        
        # Build 3D covariance
        covs3d = build_covariance_from_scaling_rotation(scales, rotations)
        
        # Project to 2D
        means2d, covs2d, depths = self.project_gaussians(
            means3d, covs3d, view_matrix, proj_matrix
        )
        
        # Compute view-dependent colors
        directions = F.normalize(means3d - camera_center, dim=-1)
        colors = self.sh(sh_coeffs, directions)
        colors = self.sigmoid(colors)
        
        # Apply opacity
        alphas = self.sigmoid(opacities)
        
        # Simplified rasterization (tile-based would be more efficient)
        image = torch.zeros(self.image_height, self.image_width, 3, device=means3d.device)
        
        # Sort by depth (back-to-front for alpha compositing)
        sorted_indices = torch.argsort(depths, descending=True)
        
        # Create pixel coordinates
        y_coords = torch.arange(self.image_height, device=means3d.device).float()
        x_coords = torch.arange(self.image_width, device=means3d.device).float()
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        pixels = torch.stack([xx, yy], dim=-1)  # (H, W, 2)
        
        # Simplified: render top-k nearest gaussians per pixel
        k = min(10, N)
        accumulated = torch.zeros(self.image_height, self.image_width, device=means3d.device)
        
        for idx in sorted_indices[:k]:
            mean = means2d[idx]  # (2,)
            cov = covs2d[idx]   # (2, 2)
            color = colors[idx]  # (3,)
            alpha = alphas[idx]
            
            # Compute Gaussian weight
            diff = pixels - mean  # (H, W, 2)
            cov_inv = torch.inverse(cov + 1e-4 * torch.eye(2, device=cov.device))
            
            # Mahalanobis distance
            mahal = torch.einsum('hwi,ij,hwj->hw', diff, cov_inv, diff)
            weight = torch.exp(-0.5 * mahal) * alpha
            
            # Alpha compositing
            contrib = weight.unsqueeze(-1) * color * (1 - accumulated.unsqueeze(-1))
            image = image + contrib
            accumulated = accumulated + weight * (1 - accumulated)
        
        return image.clamp(0, 1)


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    3D Gaussian Splatting for novel view synthesis.
    
    Uses level1 operators from KernelBench:
    - Sigmoid from level1/activations/3_Sigmoid
    - Softmax from level1/activations/5_Softmax
    - MatMul from level1/matmul/1_MatMul
    
    Supports variants: 1M, 5M (configs loaded from config_loader)
    """
    
    VARIANTS = VARIANTS
    
    @classmethod
    def from_pretrained(cls, variant: str = "1M", operator_level: Optional[OperatorLevel] = None, **kwargs):
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
        sh_degree = kwargs.get('sh_degree', 3)
        max_gaussians = kwargs.get('max_gaussians', kwargs.get('num_gaussians', 100000))
        feature_dim = kwargs.get('feature_dim', kwargs.get('features_dim', 32))
        image_height = kwargs.get('image_height', kwargs.get('image_size', 800))
        image_width = kwargs.get('image_width', kwargs.get('image_size', 800))
        
        if config is None:
            config = ModelConfig(
                hidden_size=feature_dim,
            )
        
        super().__init__()
        
        self.sh_degree = sh_degree
        self.num_sh_coeffs = (sh_degree + 1) ** 2
        
        # Learnable Gaussian parameters
        self.means = nn.Parameter(torch.randn(max_gaussians, 3))
        self.scales = nn.Parameter(torch.zeros(max_gaussians, 3))
        self.rotations = nn.Parameter(torch.zeros(max_gaussians, 4))
        self.rotations.data[:, 0] = 1.0  # Identity quaternion
        self.opacities = nn.Parameter(torch.zeros(max_gaussians))
        self.sh_coeffs = nn.Parameter(torch.zeros(max_gaussians, 3, self.num_sh_coeffs))
        
        self.renderer = GaussianRenderer(image_height, image_width)
        self.sigmoid = Sigmoid()

    def forward(
        self,
        view_matrix: torch.Tensor,
        proj_matrix: torch.Tensor,
        camera_center: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Render image from given camera pose.
        
        Args:
            view_matrix: (4, 4) world-to-camera transform
            proj_matrix: (4, 4) camera-to-clip projection
            camera_center: (3,) camera position in world coords
            active_mask: Optional (N,) mask for active Gaussians
        """
        if active_mask is not None:
            means = self.means[active_mask]
            scales = torch.exp(self.scales[active_mask])
            rotations = self.rotations[active_mask]
            opacities = self.opacities[active_mask]
            sh_coeffs = self.sh_coeffs[active_mask]
        else:
            means = self.means
            scales = torch.exp(self.scales)
            rotations = self.rotations
            opacities = self.opacities
            sh_coeffs = self.sh_coeffs
        
        return self.renderer(
            means, scales, rotations, opacities, sh_coeffs,
            view_matrix, proj_matrix, camera_center
        )
