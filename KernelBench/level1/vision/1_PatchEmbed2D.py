import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    2D Patch Embedding
    
    Used by: ViT, CLIP, SigLIP, DINOv2, EVA
    """
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, 
                 embed_dim: int = 768):
        super(Model, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 32, "img_size": 224, "patch_size": 16, "in_channels": 3, "embed_dim": 768},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("vision", "1_PatchEmbed2D")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["in_channels"], p["img_size"], p["img_size"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["img_size"], p["patch_size"], p["in_channels"], p["embed_dim"]]
