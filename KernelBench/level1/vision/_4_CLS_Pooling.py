import os
import sys
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    CLS Token Pooling
    
    Used by: ViT, CLIP, SigLIP
    
    Extract CLS token (index 0) or compute mean over spatial tokens
    for image-level features.
    
    Shapes:
        Input: (batch, num_patches + 1, embed_dim) with CLS token
        Output: (batch, embed_dim)
    """
    
    def __init__(self, embed_dim: int, pool_type: str = 'cls'):
        """
        Initialize CLS pooling.
        
        Args:
            embed_dim: Embedding dimension
            pool_type: 'cls' for CLS token, 'mean' for mean pooling
        """
        super(Model, self).__init__()
        self.embed_dim = embed_dim
        self.pool_type = pool_type
        
        # Optional layer norm after pooling
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool sequence to single vector.
        
        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            
        Returns:
            Pooled tensor (batch, embed_dim)
        """
        if self.pool_type == 'cls':
            # Extract CLS token (first position)
            pooled = x[:, 0]
        elif self.pool_type == 'mean':
            # Mean over all tokens (or exclude CLS for patch tokens only)
            pooled = x[:, 1:].mean(dim=1)
        elif self.pool_type == 'mean_all':
            # Mean over all tokens including CLS
            pooled = x.mean(dim=1)
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")
        
        return self.norm(pooled)
