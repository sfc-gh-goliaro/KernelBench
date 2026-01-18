import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Classifier-Free Guidance (CFG) Scale
    
    Used by: All diffusion inference
    
    CFG interpolation: output = uncond + scale * (cond - uncond).
    Amplifies the difference between conditional and unconditional
    predictions.
    
    Shapes:
        cond_output: (batch, ...)
        uncond_output: (batch, ...)
        Output: (batch, ...)
    """
    
    def __init__(self, guidance_scale: float = 7.5):
        """
        Initialize CFG.
        
        Args:
            guidance_scale: CFG scale factor (typically 1-15)
        """
        super(Model, self).__init__()
        self.guidance_scale = guidance_scale
    
    def forward(self, cond_output: torch.Tensor, uncond_output: torch.Tensor) -> torch.Tensor:
        """
        Apply classifier-free guidance.
        
        Args:
            cond_output: Conditional model output
            uncond_output: Unconditional model output
            
        Returns:
            Guided output
        """
        return uncond_output + self.guidance_scale * (cond_output - uncond_output)


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 8, "channels": 4, "height": 64, "width": 64},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("diffusion", "4_CFG_Scale")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    cond_output = DISTRIBUTIONS[dist_name]((p["batch_size"], p["channels"], p["height"], p["width"]), dtype=dtype, device=device)
    uncond_output = DISTRIBUTIONS[dist_name]((p["batch_size"], p["channels"], p["height"], p["width"]), dtype=dtype, device=device)
    return [cond_output, uncond_output]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return [7.5]
