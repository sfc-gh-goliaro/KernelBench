import torch
import torch.nn as nn
import torch.nn.functional as F
import math


TASK_CONFIG = {
    'comparison_mode': 'default',
    'atol': 1e-4,
    'rtol': 1e-4,
}

class Model(nn.Module):
    """
    Perceiver-style Cross-Attention with Learned Latents.
    
    Uses a fixed set of learned latent vectors to compress
    variable-length input sequences via cross-attention.
    
    Based on: "Perceiver: General Perception with Iterative Attention"
    """
    def __init__(self, input_dim, latent_dim, num_latents, num_heads, 
                 num_cross_attn=1, num_self_attn=6):
        """
        :param input_dim: Dimension of input features
        :param latent_dim: Dimension of latent vectors
        :param num_latents: Number of latent vectors
        :param num_heads: Number of attention heads
        :param num_cross_attn: Number of cross-attention layers
        :param num_self_attn: Number of self-attention layers per cross-attn
        """
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Learned latent vectors
        self.latents = nn.Parameter(torch.randn(1, num_latents, latent_dim) * 0.02)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, latent_dim)
        
        # Cross-attention blocks
        self.cross_attn_blocks = nn.ModuleList()
        for _ in range(num_cross_attn):
            self.cross_attn_blocks.append(nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(latent_dim, num_heads, batch_first=True),
                'cross_norm1': nn.LayerNorm(latent_dim),
                'cross_norm2': nn.LayerNorm(latent_dim),
                'cross_ffn': nn.Sequential(
                    nn.Linear(latent_dim, latent_dim * 4),
                    nn.GELU(),
                    nn.Linear(latent_dim * 4, latent_dim)
                ),
                # Self-attention layers
                'self_attn_layers': nn.ModuleList([
                    nn.ModuleDict({
                        'self_attn': nn.MultiheadAttention(latent_dim, num_heads, batch_first=True),
                        'norm1': nn.LayerNorm(latent_dim),
                        'norm2': nn.LayerNorm(latent_dim),
                        'ffn': nn.Sequential(
                            nn.Linear(latent_dim, latent_dim * 4),
                            nn.GELU(),
                            nn.Linear(latent_dim * 4, latent_dim)
                        )
                    })
                    for _ in range(num_self_attn)
                ])
            }))
        
    def forward(self, x, mask=None):
        """
        Forward pass for Perceiver cross-attention.
        
        :param x: Input features (batch, seq_len, input_dim)
        :param mask: Optional attention mask (batch, seq_len)
        :return: Latent representations (batch, num_latents, latent_dim)
        """
        batch_size = x.shape[0]
        
        # Project input
        x = self.input_proj(x)  # (batch, seq, latent_dim)
        
        # Expand latents for batch
        latents = self.latents.expand(batch_size, -1, -1)
        
        # Create key padding mask if provided
        key_padding_mask = ~mask if mask is not None else None
        
        # Process through cross-attention blocks
        for block in self.cross_attn_blocks:
            # Cross-attention: latents attend to input
            latents_norm = block['cross_norm1'](latents)
            
            cross_out, _ = block['cross_attn'](
                query=latents_norm,
                key=x,
                value=x,
                key_padding_mask=key_padding_mask
            )
            latents = latents + cross_out
            
            # FFN
            latents = latents + block['cross_ffn'](block['cross_norm2'](latents))
            
            # Self-attention layers
            for self_attn_layer in block['self_attn_layers']:
                # Self-attention
                latents_norm = self_attn_layer['norm1'](latents)
                self_out, _ = self_attn_layer['self_attn'](
                    query=latents_norm,
                    key=latents_norm,
                    value=latents_norm
                )
                latents = latents + self_out
                
                # FFN
                latents = latents + self_attn_layer['ffn'](self_attn_layer['norm2'](latents))
        
        return latents


# Test parameters
batch_size = 8
seq_len = 2048  # Variable-length input
input_dim = 768
latent_dim = 512
num_latents = 64
num_heads = 8

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    return [torch.randn(batch_size, seq_len, input_dim)]

def get_init_inputs():
    return [input_dim, latent_dim, num_latents, num_heads]

