import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Pixel Shuffle (Sub-pixel Convolution)
    
    Used by: ESPCN, super-resolution models, some diffusion decoders
    """
    
    def __init__(self, upscale_factor: int = 2):
        super(Model, self).__init__()
        self.upscale_factor = upscale_factor
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pixel_shuffle(x)


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 16, "channels": 64, "upscale_factor": 2, "height": 64, "width": 64},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("upsampling", "1_PixelShuffle")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    
    in_channels = p["channels"] * p["upscale_factor"] * p["upscale_factor"]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], in_channels, p["height"], p["width"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["upscale_factor"]]
