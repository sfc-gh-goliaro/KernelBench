import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Adaptive Layer Normalization (AdaLN / AdaLN-Zero).
    
    Modulates layer normalization parameters based on conditioning
    input (e.g., timestep, class). Used in DiT (Diffusion Transformer)
    and FLUX models.
    
    Based on: "Scalable Diffusion Models with Transformers" (DiT)
    """
    def __init__(self, dim, cond_dim, use_zero_init=True):
        """
        :param dim: Feature dimension to normalize
        :param cond_dim: Conditioning dimension (e.g., timestep embedding)
        :param use_zero_init: Initialize scale to 0 for residual learning
        """
        super(Model, self).__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.use_zero_init = use_zero_init
        
        # Layer normalization without learnable parameters
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        
        # Conditioning projection: produces shift and scale
        # For AdaLN-Zero, also produces gate
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * dim if use_zero_init else 2 * dim)
        )
        
        # Initialize to zero for residual learning
        if use_zero_init:
            nn.init.zeros_(self.adaLN_modulation[-1].weight)
            nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(self, x, cond):
        """
        Apply adaptive layer normalization.
        
        :param x: Input tensor (batch, seq_len, dim)
        :param cond: Conditioning tensor (batch, cond_dim)
        :return: Normalized and modulated tensor
        """
        # Get modulation parameters from conditioning
        modulation = self.adaLN_modulation(cond)  # (batch, 6*dim or 2*dim)
        
        if self.use_zero_init:
            # AdaLN-Zero: shift, scale, gate for both attention and FFN
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                modulation.chunk(6, dim=-1)
            
            # Normalize
            x_norm = self.norm(x)
            
            # Modulate
            x_modulated = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
            
            return x_modulated, gate_msa.unsqueeze(1), (shift_mlp, scale_mlp, gate_mlp)
        else:
            # Standard AdaLN
            shift, scale = modulation.chunk(2, dim=-1)
            
            # Normalize
            x_norm = self.norm(x)
            
            # Modulate: x * (1 + scale) + shift
            return x_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# Test parameters
batch_size = 16
seq_len = 256  # e.g., image patches
dim = 1152     # DiT-XL dimension
cond_dim = 1152

def get_inputs():
    x = torch.randn(batch_size, seq_len, dim)
    cond = torch.randn(batch_size, cond_dim)  # e.g., timestep embedding
    return [x, cond]

def get_init_inputs():
    return [dim, cond_dim, True]

