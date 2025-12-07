import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Flexible Patch Embedding for Vision Transformers.
    
    Supports multiple patch embedding strategies including:
    - Standard convolution-based
    - Overlapping patches
    - Hierarchical patch merging
    - Dynamic resolution
    
    Based on: "An Image is Worth 16x16 Words" and "Swin Transformer"
    """
    def __init__(self, image_size, patch_size, in_channels, embed_dim, 
                 overlap=0, use_conv_stem=False, flatten=True):
        """
        :param image_size: Input image size (assumed square)
        :param patch_size: Size of each patch
        :param in_channels: Number of input channels
        :param embed_dim: Embedding dimension
        :param overlap: Overlap between patches (0 for non-overlapping)
        :param use_conv_stem: Use convolutional stem instead of single projection
        :param flatten: Whether to flatten spatial dimensions
        """
        super(Model, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.overlap = overlap
        self.flatten = flatten
        
        # Compute stride and padding
        self.stride = patch_size - overlap
        self.num_patches_per_side = (image_size - patch_size) // self.stride + 1
        self.num_patches = self.num_patches_per_side ** 2
        
        if use_conv_stem:
            # Multi-stage convolutional stem (like in ConvNeXt)
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, embed_dim // 4, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.BatchNorm2d(embed_dim // 4),
                nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.BatchNorm2d(embed_dim // 2),
                nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.BatchNorm2d(embed_dim),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
            )
            # Recalculate num patches for conv stem
            self.num_patches_per_side = image_size // 16
            self.num_patches = self.num_patches_per_side ** 2
        else:
            # Standard patch embedding with optional overlap
            self.proj = nn.Conv2d(
                in_channels, embed_dim, 
                kernel_size=patch_size, 
                stride=self.stride,
                padding=overlap // 2
            )
        
        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim) * 0.02
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # CLS token (optional)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
    def interpolate_pos_embed(self, x, h, w):
        """Interpolate positional embeddings for different resolutions."""
        num_patches = h * w
        
        if num_patches == self.num_patches:
            return self.pos_embed
        
        # Reshape to 2D
        pos_embed = self.pos_embed.reshape(
            1, self.num_patches_per_side, self.num_patches_per_side, -1
        ).permute(0, 3, 1, 2)
        
        # Interpolate
        pos_embed = F.interpolate(
            pos_embed, size=(h, w), mode='bicubic', align_corners=False
        )
        
        return pos_embed.permute(0, 2, 3, 1).reshape(1, -1, self.embed_dim)
    
    def forward(self, x, return_all_tokens=False):
        """
        Forward pass for flexible patch embedding.
        
        :param x: Input images (batch, channels, height, width)
        :param return_all_tokens: Whether to return CLS + patch tokens or just patches
        :return: Patch embeddings (batch, num_patches, embed_dim) or 
                 (batch, 1 + num_patches, embed_dim) if return_all_tokens
        """
        batch_size, _, height, width = x.shape
        
        # Apply projection
        x = self.proj(x)  # (batch, embed_dim, h_patches, w_patches)
        
        h_patches, w_patches = x.shape[2], x.shape[3]
        
        if self.flatten:
            # Flatten spatial dimensions
            x = x.flatten(2).transpose(1, 2)  # (batch, num_patches, embed_dim)
            
            # Add positional embedding (interpolate if needed)
            pos_embed = self.interpolate_pos_embed(x, h_patches, w_patches)
            x = x + pos_embed
            
            # Apply layer norm
            x = self.norm(x)
            
            if return_all_tokens:
                # Prepend CLS token
                cls_tokens = self.cls_token.expand(batch_size, -1, -1)
                x = torch.cat([cls_tokens, x], dim=1)
        else:
            # Keep spatial structure
            x = x.permute(0, 2, 3, 1)  # (batch, h, w, embed_dim)
            x = self.norm(x)
        
        return x


# Test parameters
batch_size = 16
image_size = 224
patch_size = 16
in_channels = 3
embed_dim = 768
overlap = 0

def get_inputs():
    return [torch.randn(batch_size, in_channels, image_size, image_size)]

def get_init_inputs():
    return [image_size, patch_size, in_channels, embed_dim, overlap]

