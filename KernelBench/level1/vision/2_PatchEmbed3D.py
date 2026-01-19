import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    3D Patch Embedding (Video)
    
    Used by: Qwen2-VL, Video-LLaVA, CogVideo
    
    Video patch embedding via Conv3d for temporal-spatial patches.
    Creates a 3D grid of tokens from video frames.
    
    Shapes:
        Input: (batch, channels, time, height, width)
        Output: (batch, num_patches, embed_dim)
    """
    
    def __init__(self, img_size: int = 224, num_frames: int = 8, patch_size: int = 16,
                 temporal_patch_size: int = 2, in_channels: int = 3, embed_dim: int = 768):
        """
        Initialize 3D patch embedding.
        
        Args:
            img_size: Input image size (spatial)
            num_frames: Number of video frames
            patch_size: Spatial patch size
            temporal_patch_size: Temporal patch size
            in_channels: Number of input channels
            embed_dim: Embedding dimension
        """
        super(Model, self).__init__()
        self.img_size = img_size
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.embed_dim = embed_dim
        
        self.num_spatial_patches = (img_size // patch_size) ** 2
        self.num_temporal_patches = num_frames // temporal_patch_size
        self.num_patches = self.num_spatial_patches * self.num_temporal_patches
        
        # Conv3d with kernel (temporal_patch, spatial_patch, spatial_patch)
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            stride=(temporal_patch_size, patch_size, patch_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed video patches.
        
        Args:
            x: Input video (batch, channels, time, height, width)
            
        Returns:
            Patch embeddings (batch, num_patches, embed_dim)
        """
        # Project patches: (batch, embed_dim, T/temp_patch, H/patch, W/patch)
        x = self.proj(x)
        
        # Flatten all spatial-temporal dimensions
        x = x.flatten(2)  # (batch, embed_dim, num_patches)
        
        # Transpose to (batch, num_patches, embed_dim)
        x = x.transpose(1, 2)
        
        return x


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 8, "img_size": 224, "num_frames": 8, "patch_size": 16, "temporal_patch_size": 2, "in_channels": 3, "embed_dim": 768},
    # Qwen2-VL-2B: patch_size=14, temporal_patch_size=2, vision_hidden=1280
    {"batch_size": 8, "img_size": 224, "num_frames": 8, "patch_size": 14, "temporal_patch_size": 2, "in_channels": 3, "embed_dim": 1280},
    # Qwen2-VL-7B: patch_size=14, temporal_patch_size=2, vision_hidden=1280
    {"batch_size": 4, "img_size": 224, "num_frames": 16, "patch_size": 14, "temporal_patch_size": 2, "in_channels": 3, "embed_dim": 1280},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("vision", "2_PatchEmbed3D")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["in_channels"], p["num_frames"], p["img_size"], p["img_size"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["img_size"], p["num_frames"], p["patch_size"], p["temporal_patch_size"], p["in_channels"], p["embed_dim"]]
