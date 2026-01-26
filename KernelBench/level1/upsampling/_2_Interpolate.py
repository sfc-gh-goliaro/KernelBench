import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Interpolation Upsampling
    
    Used by: UNet, FPN, diffusion models, segmentation networks
    
    Upsamples spatial dimensions using various interpolation methods.
    Supports bilinear, bicubic, nearest, and trilinear modes.
    
    Shapes:
        Input: (batch_size, channels, height, width)
        Output: (batch_size, channels, new_height, new_width)
    """
    
    def __init__(self, scale_factor: float = 2.0, mode: str = 'bilinear',
                 align_corners: bool = False):
        """
        Initialize Interpolation.
        
        Args:
            scale_factor: Upsampling factor
            mode: Interpolation mode ('nearest', 'bilinear', 'bicubic', 'trilinear')
            align_corners: If True, align corners of input and output
        """
        super(Model, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners if mode != 'nearest' else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply interpolation upsampling.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            Upsampled tensor (batch_size, channels, new_height, new_width)
        """
        return F.interpolate(
            x, 
            scale_factor=self.scale_factor, 
            mode=self.mode,
            align_corners=self.align_corners
        )


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 16, "channels": 512, "height": 32, "width": 32, "scale_factor": 2.0, "mode": 'bilinear', "align_corners": False},
    # U-Net: decoder upsampling in segmentation
    {"batch_size": 8, "channels": 256, "height": 64, "width": 64, "scale_factor": 2.0, "mode": 'bilinear', "align_corners": True},
    # SDXL: diffusion model feature upsampling
    {"batch_size": 2, "channels": 1280, "height": 16, "width": 16, "scale_factor": 2.0, "mode": 'nearest', "align_corners": False},
    # FPN: feature pyramid network upsampling
    {"batch_size": 4, "channels": 256, "height": 32, "width": 32, "scale_factor": 2.0, "mode": 'bicubic', "align_corners": False},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("upsampling", "2_Interpolate")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["channels"], p["height"], p["width"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["scale_factor"], p["mode"], p["align_corners"]]
