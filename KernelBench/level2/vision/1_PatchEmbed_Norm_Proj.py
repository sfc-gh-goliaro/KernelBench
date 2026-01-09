import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Fused Patch Embedding + LayerNorm + Linear Projection
    
    Used by: Vision Transformers (ViT, DeiT, BEiT, CLIP, SigLIP)
    
    Fuses the image patchification, embedding, normalization, and
    optional projection into a single efficient operation.
    
    Shapes:
        Input: (batch_size, channels, height, width)
        Output: (batch_size, num_patches, embed_dim)
    """
    
    def __init__(self, img_size: int = 224, patch_size: int = 14, 
                 in_channels: int = 3, embed_dim: int = 768,
                 proj_dim: int = None, use_norm: bool = True):
        """
        Initialize fused patch embedding.
        
        Args:
            img_size: Input image size
            patch_size: Patch size
            in_channels: Number of input channels
            embed_dim: Embedding dimension
            proj_dim: Optional projection dimension (for CLIP-style)
            use_norm: Whether to apply LayerNorm
        """
        super(Model, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        
        # Patch embedding via conv
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        # Optional norm
        self.norm = nn.LayerNorm(embed_dim) if use_norm else nn.Identity()
        
        # Optional projection (for multi-modal like CLIP)
        self.linear_proj = nn.Linear(embed_dim, proj_dim) if proj_dim else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fused patch embedding + norm + projection.
        
        Args:
            x: Input images (batch_size, channels, height, width)
            
        Returns:
            Patch embeddings (batch_size, num_patches, embed_dim/proj_dim)
        """
        # Patch embedding
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        
        # Flatten spatial dims
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Fused norm + projection
        x = self.norm(x)
        x = self.linear_proj(x)
        
        return x


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
img_size = 224
patch_size = 14
in_channels = 3
embed_dim = 768

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    images = torch.randn(batch_size, in_channels, img_size, img_size, device='cuda')
    return [images]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [img_size, patch_size, in_channels, embed_dim]

