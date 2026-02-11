import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from ..normalization._6_LayerNorm import Model as LayerNormOp
from ..matmul._10_Linear import Model as Linear


class Model(nn.Module):
    """
    Patch Merging
    
    Used by: Swin Transformer v1 and v2 (between stages)
    
    Spatial downsampling by concatenating 2x2 neighboring patches
    followed by layer norm and linear projection. Reduces spatial
    resolution by 2x and doubles the channel dimension.
    
    Supports two norm orderings via ``norm_before_reduction``:
      - True  (Swin v1): norm(4*dim) -> reduction(4*dim, 2*dim)
      - False (Swin v2): reduction(4*dim, 2*dim) -> norm(2*dim)
    
    Accepts either 4D spatial input (B, H, W, C) or 3D sequential
    input (B, H*W, C) with ``input_dimensions`` provided.
    Handles odd spatial dimensions by zero-padding before merging.
    
    Shapes:
        Input:  (batch, H*W, C) with input_dimensions=(H, W)
                or (batch, H, W, C)
        Output: (batch, (H//2)*(W//2), 2*C)
    
    Uses level1 operators:
    - Linear from level1/matmul/10_Linear
    - LayerNorm from level1/normalization/6_LayerNorm
    """
    
    def __init__(self, dim: int, norm_before_reduction: bool = True):
        """
        Initialize patch merging.
        
        Args:
            dim: Input channel dimension
            norm_before_reduction: If True (Swin v1), apply LayerNorm on
                concatenated 4*dim features before linear reduction. If False
                (Swin v2), apply LayerNorm on the reduced 2*dim features after
                the linear reduction.
        """
        super(Model, self).__init__()
        self.dim = dim
        self.norm_before_reduction = norm_before_reduction
        
        # Linear projection from 4*dim to 2*dim (no bias, matches HF)
        self.reduction = Linear(4 * dim, 2 * dim, bias=False)
        
        # LayerNorm dimension depends on ordering
        norm_dim = 4 * dim if norm_before_reduction else 2 * dim
        self.norm = LayerNormOp(norm_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        input_dimensions: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Merge patches to reduce spatial resolution.
        
        Args:
            x: Input tensor, either:
               - (batch, height, width, channels) — 4D spatial layout
               - (batch, seq_len, channels) — 3D sequential layout
                 (requires ``input_dimensions``)
            input_dimensions: (height, width) required when x is 3D.
            
        Returns:
            Merged tensor (batch, (H_out)*(W_out), 2*channels) where
            H_out = ceil(H/2), W_out = ceil(W/2).
        """
        if x.dim() == 3:
            assert input_dimensions is not None, \
                "input_dimensions required for 3D input"
            height, width = input_dimensions
            batch_size = x.shape[0]
            num_channels = x.shape[2]
            x = x.view(batch_size, height, width, num_channels)
        else:
            batch_size, height, width, num_channels = x.shape
        
        # Pad if height or width is odd
        if height % 2 == 1 or width % 2 == 1:
            x = F.pad(x, (0, 0, 0, width % 2, 0, height % 2))
        
        # Extract 2x2 patches
        x0 = x[:, 0::2, 0::2, :]  # Top-left
        x1 = x[:, 1::2, 0::2, :]  # Bottom-left
        x2 = x[:, 0::2, 1::2, :]  # Top-right
        x3 = x[:, 1::2, 1::2, :]  # Bottom-right
        
        # Concatenate along channel dimension: (B, H/2, W/2, 4*C)
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        
        # Flatten spatial dims back to sequence: (B, H/2 * W/2, 4*C)
        x = x.view(batch_size, -1, 4 * num_channels)
        
        if self.norm_before_reduction:
            # Swin v1 order: norm then reduce
            x = self.norm(x)
            x = self.reduction(x)
        else:
            # Swin v2 order: reduce then norm
            x = self.reduction(x)
            x = self.norm(x)
        
        return x
