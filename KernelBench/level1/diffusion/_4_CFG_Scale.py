import os
import sys
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
