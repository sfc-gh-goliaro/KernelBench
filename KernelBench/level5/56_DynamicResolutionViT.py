import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Dynamic Resolution Vision Transformer Encoder.
    
    Handles arbitrary image resolutions through dynamic patch
    embedding and position interpolation. Used in Qwen-VL and NaViT.
    
    Based on: "Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution"
    """
    def __init__(self, patch_size=14, embed_dim=1024, num_heads=16, 
                 num_layers=24, mlp_ratio=4.0):
        """
        :param patch_size: Size of image patches
        :param embed_dim: Embedding dimension
        :param num_heads: Number of attention heads
        :param num_layers: Number of transformer layers
        :param mlp_ratio: MLP hidden dim ratio
        """
        super(Model, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Patch embedding (convolution-based)
        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
        # Learnable position embedding for base resolution
        # Will be interpolated for other resolutions
        self.base_resolution = (224, 224)
        num_patches = (224 // patch_size) ** 2
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, embed_dim) * 0.02
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=0.0,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Final norm
        self.norm = nn.LayerNorm(embed_dim)
    
    def interpolate_pos_encoding(self, height, width):
        """
        Interpolate position encoding for arbitrary resolution.
        
        :param height: Image height in patches
        :param width: Image width in patches
        :return: Interpolated position encoding
        """
        num_patches = height * width
        
        # Get base resolution patches
        base_h = self.base_resolution[0] // self.patch_size
        base_w = self.base_resolution[1] // self.patch_size
        
        if height == base_h and width == base_w:
            return self.pos_embed
        
        # Reshape to 2D for interpolation
        pos_2d = self.pos_embed.reshape(1, base_h, base_w, self.embed_dim)
        pos_2d = pos_2d.permute(0, 3, 1, 2)  # (1, dim, h, w)
        
        # Interpolate
        pos_2d = F.interpolate(
            pos_2d, size=(height, width), 
            mode='bicubic', align_corners=False
        )
        
        # Reshape back
        pos_2d = pos_2d.permute(0, 2, 3, 1)  # (1, h, w, dim)
        return pos_2d.reshape(1, num_patches, self.embed_dim)
    
    def forward(self, images, return_all=False):
        """
        Forward pass with dynamic resolution support.
        
        :param images: Input images (batch, 3, height, width)
        :param return_all: Return all tokens or just CLS
        :return: Image features
        """
        batch_size, _, height, width = images.shape
        
        # Patch embedding
        x = self.patch_embed(images)  # (batch, embed_dim, h_patches, w_patches)
        h_patches, w_patches = x.shape[2], x.shape[3]
        
        # Flatten spatial dimensions
        x = x.flatten(2).transpose(1, 2)  # (batch, num_patches, embed_dim)
        
        # Add interpolated position encoding
        pos_embed = self.interpolate_pos_encoding(h_patches, w_patches)
        x = x + pos_embed
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Final norm
        x = self.norm(x)
        
        if return_all:
            return x
        return x[:, 0]  # CLS token


# Test parameters
batch_size = 4
height = 448  # Non-standard resolution
width = 336
patch_size = 14
embed_dim = 1024
num_heads = 16
num_layers = 24

def get_inputs():
    images = torch.randn(batch_size, 3, height, width)
    return [images]

def get_init_inputs():
    return [patch_size, embed_dim, num_heads, num_layers]

