import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
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


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 32, "num_patches": 196, "embed_dim": 768},
    # ViT-Large: 14x14 patches, 1024-dim
    {"batch_size": 16, "num_patches": 256, "embed_dim": 1024},
    # CLIP ViT-H/14: 16x16 patches, 1280-dim
    {"batch_size": 8, "num_patches": 256, "embed_dim": 1280},
    # SigLIP-Large: 24x24 patches from 384px image, 1024-dim
    {"batch_size": 16, "num_patches": 576, "embed_dim": 1024},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("vision", "4_CLS_Pooling")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    x = DISTRIBUTIONS[dist_name]((p["batch_size"], p["num_patches"] + 1, p["embed_dim"]), dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["embed_dim"], 'cls']
