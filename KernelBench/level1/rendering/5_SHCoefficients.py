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
    Spherical Harmonics Coefficients (3D/Neural Rendering)

    Used by: 3D Gaussian Splatting, Plenoxels, NeRF++

    Evaluates spherical harmonics for view-dependent color representation.
    Converts SH coefficients to RGB given view direction.

    Shapes:
        sh_coeffs: (batch, num_coeffs, 3) SH coefficients per color channel
        directions: (batch, 3) view directions
        Output: (batch, 3) RGB colors
    """

    def __init__(self, degree: int = 3):
        """
        Initialize spherical harmonics evaluator.

        Args:
            degree: Maximum SH degree (0-3 commonly used)
        """
        super(Model, self).__init__()
        self.degree = degree
        self.num_coeffs = (degree + 1) ** 2

        # Precompute constants
        self.register_buffer('C0', torch.tensor(0.28209479177387814))
        self.register_buffer('C1', torch.tensor(0.4886025119029199))
        self.register_buffer('C2', torch.tensor([
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396
        ]))
        self.register_buffer('C3', torch.tensor([
            -0.5900435899266435,
            2.890611442640554,
            -0.4570457994644658,
            0.3731763325901154,
            -0.4570457994644658,
            1.445305721320277,
            -0.5900435899266435
        ]))

    def forward(self, sh_coeffs: torch.Tensor,
                directions: torch.Tensor) -> torch.Tensor:
        """
        Evaluate SH coefficients for given directions.

        Args:
            sh_coeffs: SH coefficients (batch, num_coeffs, 3)
            directions: View directions (batch, 3), should be normalized

        Returns:
            RGB colors (batch, 3)
        """
        # Normalize directions
        directions = F.normalize(directions, dim=-1)
        x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]

        batch_size = directions.shape[0]
        device = directions.device

        # Evaluate SH basis functions
        result = self.C0 * sh_coeffs[:, 0]  # l=0

        if self.degree >= 1 and sh_coeffs.shape[1] >= 4:
            # l=1
            result = result + self.C1 * (
                -y.unsqueeze(-1) * sh_coeffs[:, 1] +
                z.unsqueeze(-1) * sh_coeffs[:, 2] +
                -x.unsqueeze(-1) * sh_coeffs[:, 3]
            )

        if self.degree >= 2 and sh_coeffs.shape[1] >= 9:
            # l=2
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z

            result = result + self.C2[0] * xy.unsqueeze(-1) * sh_coeffs[:, 4]
            result = result + self.C2[1] * yz.unsqueeze(-1) * sh_coeffs[:, 5]
            result = result + self.C2[2] * (2.0 * zz - xx - yy).unsqueeze(-1) * sh_coeffs[:, 6]
            result = result + self.C2[3] * xz.unsqueeze(-1) * sh_coeffs[:, 7]
            result = result + self.C2[4] * (xx - yy).unsqueeze(-1) * sh_coeffs[:, 8]

        if self.degree >= 3 and sh_coeffs.shape[1] >= 16:
            # l=3
            result = result + self.C3[0] * y * (3 * x * x - y * y).unsqueeze(-1) * sh_coeffs[:, 9]
            result = result + self.C3[1] * (x * y * z).unsqueeze(-1) * sh_coeffs[:, 10]
            result = result + self.C3[2] * y * (4 * z * z - x * x - y * y).unsqueeze(-1) * sh_coeffs[:, 11]
            result = result + self.C3[3] * z * (2 * z * z - 3 * x * x - 3 * y * y).unsqueeze(-1) * sh_coeffs[:, 12]
            result = result + self.C3[4] * x * (4 * z * z - x * x - y * y).unsqueeze(-1) * sh_coeffs[:, 13]
            result = result + self.C3[5] * z * (x * x - y * y).unsqueeze(-1) * sh_coeffs[:, 14]
            result = result + self.C3[6] * x * (x * x - 3 * y * y).unsqueeze(-1) * sh_coeffs[:, 15]

        # Clamp to valid color range
        result = torch.clamp(result + 0.5, 0.0, 1.0)

        return result


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 100000, "degree": 3, "num_coeffs": 16},
    # 3DGS: Standard spherical harmonics for view-dependent color
    {"batch_size": 200000, "degree": 3, "num_coeffs": 16},
    # Plenoxels: Lower degree SH for faster rendering
    {"batch_size": 500000, "degree": 2, "num_coeffs": 9},
    # 3DGS-HD: High-fidelity with degree-4 harmonics
    {"batch_size": 150000, "degree": 4, "num_coeffs": 25},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("rendering", "5_SHCoefficients")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    sh_coeffs = DISTRIBUTIONS[dist_name]((p["batch_size"], p["num_coeffs"], 3), dtype=dtype, device=device)
    return [sh_coeffs, directions]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["degree"]]
