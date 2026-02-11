import os
import sys
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Adaptive Layer Normalization (AdaLN)
    
    Used by: DiT, SD-3, FLUX, PixArt
    
    Adaptive LayerNorm where scale and shift are predicted from
    timestep/conditioning embedding. Core building block for diffusion
    transformers.
    
    Shapes:
        x: (batch, seq_len, hidden_size)
        conditioning: (batch, cond_dim)
        Output: (batch, seq_len, hidden_size)
    """
    
    def __init__(self, hidden_size: int, cond_dim: int):
        """
        Initialize AdaLN.
        
        Args:
            hidden_size: Hidden dimension
            cond_dim: Conditioning embedding dimension
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        
        # Project conditioning to scale and shift
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_size)
        )
    
    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive layer normalization.
        
        Args:
            x: Input tensor (batch, seq_len, hidden_size)
            conditioning: Conditioning embedding (batch, cond_dim)
            
        Returns:
            Normalized and modulated tensor
        """
        # Get scale and shift from conditioning
        shift, scale = self.adaLN_modulation(conditioning).chunk(2, dim=-1)
        
        # Apply layer norm
        x = self.norm(x)
        
        # Apply adaptive scale and shift
        # shift, scale: (batch, hidden_size) -> (batch, 1, hidden_size)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        return x
