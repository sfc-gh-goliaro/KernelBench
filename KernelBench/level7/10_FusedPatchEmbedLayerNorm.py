import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Fused Vision Encoder Patch Embedding + LayerNorm.
    
    Combines:
    1. Patch extraction (Conv2d with stride=patch_size)
    2. Flatten and transpose
    3. LayerNorm on patch embeddings
    
    This fusion eliminates intermediate memory traffic in ViT-style
    vision encoders used in multimodal models.
    
    Reference: ViT, SigLIP, InternVL patch embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768,
                 norm_eps=1e-6, bias=True):
        """
        :param img_size: Input image size
        :param patch_size: Patch size (P x P)
        :param in_channels: Number of input channels
        :param embed_dim: Embedding dimension
        :param norm_eps: LayerNorm epsilon
        :param bias: Use bias in projection
        """
        super(Model, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.norm_eps = norm_eps
        
        # Patch embedding as convolution
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=bias
        )
        
        # LayerNorm
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)
    
    def forward(self, x):
        """
        Fused patch embedding + LayerNorm.
        
        :param x: Input images (batch, channels, height, width)
        :return: Patch embeddings (batch, num_patches, embed_dim)
        """
        batch_size = x.shape[0]
        
        # === FUSED KERNEL START ===
        # Step 1: Patch projection (Conv2d)
        x = self.proj(x)  # (batch, embed_dim, H/P, W/P)
        
        # Step 2: Flatten spatial dimensions
        x = x.flatten(2)  # (batch, embed_dim, num_patches)
        
        # Step 3: Transpose to (batch, num_patches, embed_dim)
        x = x.transpose(1, 2)
        
        # Step 4: LayerNorm
        x = self.norm(x)
        # === FUSED KERNEL END ===
        
        return x


# Overlapping patch embedding variant
class FusedOverlappingPatchEmbed(nn.Module):
    """
    Fused overlapping patch embedding + LayerNorm.
    
    Uses overlapping patches for better local feature preservation.
    """
    def __init__(self, img_size=224, patch_size=16, stride=12, in_channels=3, 
                 embed_dim=768, norm_eps=1e-6):
        super(FusedOverlappingPatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Calculate output size
        self.num_patches_h = (img_size - patch_size) // stride + 1
        self.num_patches_w = (img_size - patch_size) // stride + 1
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # Overlapping convolution
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=stride
        )
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)
    
    def forward(self, x):
        """Overlapping patch embedding."""
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


# Hybrid CNN-ViT patch embedding
class FusedHybridPatchEmbed(nn.Module):
    """
    Fused Hybrid Patch Embedding (CNN stem + projection + LayerNorm).
    
    Uses a small CNN stem before projection for better features.
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, 
                 embed_dim=768, stem_channels=64):
        super(FusedHybridPatchEmbed, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # CNN stem (reduces resolution by 4x)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(stem_channels),
            nn.GELU(),
            nn.Conv2d(stem_channels, stem_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(stem_channels),
            nn.GELU(),
        )
        
        # Adjust patch size for stem output
        stem_patch_size = patch_size // 4
        self.proj = nn.Conv2d(
            stem_channels, embed_dim,
            kernel_size=stem_patch_size, stride=stem_patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        """Hybrid patch embedding."""
        # === FUSED KERNEL START ===
        x = self.stem(x)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        # === FUSED KERNEL END ===
        return x


# Dynamic resolution patch embedding
class FusedDynamicPatchEmbed(nn.Module):
    """
    Fused Dynamic Resolution Patch Embedding + LayerNorm.
    
    Handles variable input sizes with 2D interpolated position embeddings.
    """
    def __init__(self, patch_size=14, in_channels=3, embed_dim=1024,
                 max_img_size=1344):
        super(FusedDynamicPatchEmbed, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.max_img_size = max_img_size
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)
        
        # Learnable position embedding for max size
        max_patches = (max_img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches, embed_dim) * 0.02)
    
    def _interpolate_pos_embed(self, num_patches_h, num_patches_w):
        """Interpolate position embeddings for current resolution."""
        max_patches_side = int(self.pos_embed.shape[1] ** 0.5)
        pos_embed = self.pos_embed.view(1, max_patches_side, max_patches_side, self.embed_dim)
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # (1, dim, H, W)
        
        pos_embed = F.interpolate(
            pos_embed, size=(num_patches_h, num_patches_w),
            mode='bicubic', align_corners=False
        )
        
        return pos_embed.permute(0, 2, 3, 1).view(1, -1, self.embed_dim)
    
    def forward(self, x):
        """Dynamic resolution patch embedding."""
        batch, _, H, W = x.shape
        
        # Patch projection
        x = self.proj(x)  # (batch, embed_dim, H/P, W/P)
        num_patches_h, num_patches_w = x.shape[2], x.shape[3]
        
        # Flatten and transpose
        x = x.flatten(2).transpose(1, 2)  # (batch, num_patches, embed_dim)
        
        # LayerNorm
        x = self.norm(x)
        
        # Add interpolated position embeddings
        pos_embed = self._interpolate_pos_embed(num_patches_h, num_patches_w)
        x = x + pos_embed
        
        return x


# Test parameters
batch_size = 8
img_size = 224
patch_size = 16
in_channels = 3
embed_dim = 768

def get_inputs():
    return [torch.randn(batch_size, in_channels, img_size, img_size)]

def get_init_inputs():
    return [img_size, patch_size, in_channels, embed_dim]

