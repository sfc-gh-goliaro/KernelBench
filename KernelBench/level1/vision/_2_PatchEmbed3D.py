import os
import sys
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    3D Patch Embedding (Video)
    
    Used by: Qwen2-VL, Video-LLaVA, CogVideo
    
    Video patch embedding via Conv3d for temporal-spatial patches.
    Creates a 3D grid of tokens from video frames.
    
    Supports two input modes:
    
    1. **Full video** (default):
       Input:  (batch, channels, time, height, width)
       Output: (batch, num_patches, embed_dim)
       The Conv3d strides over the full spatial-temporal volume.
    
    2. **Pre-chunked patches** (e.g. Qwen2-VL):
       Input:  (num_patches, channels, temporal_patch_size, patch_size, patch_size)
               Each element is already a single 3D patch.
       Output: (num_patches, embed_dim)
       The Conv3d kernel covers the entire input, producing one vector per patch.
       This mode is auto-detected when spatial dims equal the Conv3d kernel size.
    """
    
    def __init__(self, img_size: int = 224, num_frames: int = 8, patch_size: int = 16,
                 temporal_patch_size: int = 2, in_channels: int = 3, embed_dim: int = 768,
                 bias: bool = True):
        """
        Initialize 3D patch embedding.
        
        Args:
            img_size: Input image size (spatial)
            num_frames: Number of video frames
            patch_size: Spatial patch size
            temporal_patch_size: Temporal patch size
            in_channels: Number of input channels
            embed_dim: Embedding dimension
            bias: If True, adds a learnable bias to the Conv3d projection. Default: True
        """
        super(Model, self).__init__()
        self.img_size = img_size
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        self.num_spatial_patches = (img_size // patch_size) ** 2
        self.num_temporal_patches = num_frames // temporal_patch_size
        self.num_patches = self.num_spatial_patches * self.num_temporal_patches
        
        # Conv3d with kernel (temporal_patch, spatial_patch, spatial_patch)
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            stride=(temporal_patch_size, patch_size, patch_size),
            bias=bias,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed video patches.
        
        Accepts three input layouts:
        
        1. Full video ``(batch, channels, time, height, width)`` —
           returns ``(batch, num_patches, embed_dim)``.
        2. Pre-chunked 5-D patches
           ``(num_patches, channels, temporal_patch_size, patch_size, patch_size)``
           where each element is a single 3-D patch —
           returns ``(num_patches, embed_dim)``.
        3. Flattened patches ``(num_patches, channels * temporal_patch_size * patch_size * patch_size)``
           (used by Qwen2-VL which flattens each patch into a vector) —
           automatically reshaped to layout 2, returns ``(num_patches, embed_dim)``.
        """
        target_dtype = self.proj.weight.dtype
        x = x.to(dtype=target_dtype)

        # Layout 3: flat 2-D input -> reshape to 5-D pre-chunked patches
        if x.ndim == 2:
            x = x.view(-1, self.in_channels, self.temporal_patch_size,
                        self.patch_size, self.patch_size)

        # Detect pre-chunked mode: spatial dims match kernel size exactly,
        # so Conv3d produces a single (1,1,1) output per patch.
        _, _, t, h, w = x.shape
        pre_chunked = (t == self.temporal_patch_size and
                       h == self.patch_size and
                       w == self.patch_size)

        if pre_chunked:
            # (N, C, T_ps, H_ps, W_ps) -> Conv3d -> (N, embed_dim, 1, 1, 1)
            return self.proj(x).view(-1, self.embed_dim)
        else:
            # (batch, C, T, H, W) -> Conv3d -> (batch, embed_dim, T', H', W')
            x = self.proj(x)
            x = x.flatten(2)           # (batch, embed_dim, num_patches)
            x = x.transpose(1, 2)      # (batch, num_patches, embed_dim)
            return x
