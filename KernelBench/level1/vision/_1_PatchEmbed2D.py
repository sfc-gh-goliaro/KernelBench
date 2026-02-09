import os
import sys
import torch
import torch.nn as nn
from typing import Tuple, Union

class Model(nn.Module):
    """
    2D Patch Embedding
    
    Used by: ViT, CLIP, SigLIP, DINOv2, EVA, SwinV2
    
    Image patch embedding via Conv2d with kernel_size=stride=patch_size.
    Optionally flattens spatial patches into a sequence of embeddings.
    
    Shapes:
        Input: (batch, in_channels, height, width)
        Output (flatten=True):  (batch, num_patches, embed_dim)
        Output (flatten=False): (batch, embed_dim, H/patch, W/patch)
    
    When flatten=True (default), also returns spatial dimensions (H/patch, W/patch)
    as a second return value, which is needed by hierarchical models like SwinV2.
    """
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, 
                 embed_dim: int = 768, flatten: bool = True):
        """
        Initialize patch embedding.
        
        Args:
            img_size: Input image size
            patch_size: Size of each patch
            in_channels: Number of input channels
            embed_dim: Embedding dimension
            flatten: If True, flatten spatial dims and transpose to
                     (batch, num_patches, embed_dim). If False, return raw
                     Conv2d output (batch, embed_dim, H/patch, W/patch).
        """
        super(Model, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.flatten = flatten
        
        # Conv2d with kernel and stride equal to patch size
        # Named "projection" to align with HuggingFace weight naming conventions
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> Union[Tuple[torch.Tensor, Tuple[int, int]], torch.Tensor]:
        """
        Embed image patches.
        
        Args:
            x: Input image (batch, channels, height, width)
            
        Returns:
            If flatten=True:
                Tuple of (patch_embeddings, spatial_dims) where
                patch_embeddings is (batch, num_patches, embed_dim) and
                spatial_dims is (height // patch_size, width // patch_size).
            If flatten=False:
                Raw Conv2d output (batch, embed_dim, H/patch, W/patch).
        """
        # Project patches: (batch, embed_dim, H/patch, W/patch)
        x = self.projection(x)
        
        if not self.flatten:
            return x
        
        _, _, height, width = x.shape
        
        # Flatten spatial dimensions: (batch, embed_dim, num_patches)
        x = x.flatten(2)
        
        # Transpose to (batch, num_patches, embed_dim)
        x = x.transpose(1, 2)
        
        return x, (height, width)


# ============================================================================
# Benchmark Configuration
# ============================================================================
