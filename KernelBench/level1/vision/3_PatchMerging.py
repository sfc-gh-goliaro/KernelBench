import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Patch Merging
    
    Used by: Swin Transformer (between stages)
    
    Spatial downsampling by concatenating 2x2 neighboring patches
    followed by linear projection. Reduces spatial resolution by 2x.
    
    Shapes:
        Input: (batch, height, width, channels)
        Output: (batch, height/2, width/2, 2*channels)
    """
    
    def __init__(self, dim: int):
        """
        Initialize patch merging.
        
        Args:
            dim: Input channel dimension
        """
        super(Model, self).__init__()
        self.dim = dim
        
        # Linear projection from 4*dim to 2*dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge patches to reduce spatial resolution.
        
        Args:
            x: Input tensor (batch, height, width, channels)
            
        Returns:
            Merged tensor (batch, height/2, width/2, 2*channels)
        """
        B, H, W, C = x.shape
        
        # Ensure H and W are even
        assert H % 2 == 0 and W % 2 == 0, f"H ({H}) and W ({W}) must be even"
        
        # Extract 2x2 patches
        x0 = x[:, 0::2, 0::2, :]  # Top-left
        x1 = x[:, 1::2, 0::2, :]  # Bottom-left
        x2 = x[:, 0::2, 1::2, :]  # Top-right
        x3 = x[:, 1::2, 1::2, :]  # Bottom-right
        
        # Concatenate along channel dimension
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4*C)
        
        # Layer norm and linear projection
        x = self.norm(x)
        x = self.reduction(x)  # (B, H/2, W/2, 2*C)
        
        return x


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
height = 56
width = 56
channels = 96

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    x = torch.randn(batch_size, height, width, channels, device='cuda')
    return [x]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [channels]

