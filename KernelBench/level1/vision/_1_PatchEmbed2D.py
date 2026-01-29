import os
import sys
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    2D Patch Embedding
    
    Used by: ViT, CLIP, SigLIP, DINOv2, EVA
    
    Image patch embedding via Conv2d with kernel_size=stride=patch_size.
    Flattens spatial patches into a sequence of embeddings.
    
    Shapes:
        Input: (batch, 3, height, width)
        Output: (batch, num_patches, embed_dim)
    """
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, 
                 embed_dim: int = 768):
        """
        Initialize patch embedding.
        
        Args:
            img_size: Input image size
            patch_size: Size of each patch
            in_channels: Number of input channels
            embed_dim: Embedding dimension
        """
        super(Model, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Conv2d with kernel and stride equal to patch size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed image patches.
        
        Args:
            x: Input image (batch, channels, height, width)
            
        Returns:
            Patch embeddings (batch, num_patches, embed_dim)
        """
        # Project patches: (batch, embed_dim, H/patch, W/patch)
        x = self.proj(x)
        
        # Flatten spatial dimensions: (batch, embed_dim, num_patches)
        x = x.flatten(2)
        
        # Transpose to (batch, num_patches, embed_dim)
        x = x.transpose(1, 2)
        
        return x


# ============================================================================
# Benchmark Configuration
# ============================================================================
