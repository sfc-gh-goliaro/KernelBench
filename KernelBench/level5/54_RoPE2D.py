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
    2D Rotary Position Embedding for Vision Transformers.
    
    Extends RoPE to 2D spatial positions for image patches.
    Used in Qwen-VL, FLUX, and vision-language models.
    
    Based on: RoFormer extended to 2D for vision
    """
    def __init__(self, dim, max_height=64, max_width=64, base=10000):
        """
        :param dim: Dimension of embeddings (must be divisible by 4)
        :param max_height: Maximum image height in patches
        :param max_width: Maximum image width in patches
        :param base: Base for frequency computation
        """
        super(Model, self).__init__()
        assert dim % 4 == 0, "Dimension must be divisible by 4 for 2D RoPE"
        
        self.dim = dim
        self.max_height = max_height
        self.max_width = max_width
        
        # Dimension split: half for height, half for width
        self.dim_per_axis = dim // 2
        
        # Precompute frequencies
        half = self.dim_per_axis // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, half, 2).float() / half))
        self.register_buffer('inv_freq', inv_freq)
        
        # Build 2D position cache
        self._build_cache(max_height, max_width)
    
    def _build_cache(self, height, width):
        """Build sin/cos cache for 2D positions."""
        # Height positions
        h_pos = torch.arange(height).float()
        h_freqs = torch.outer(h_pos, self.inv_freq)
        h_emb = torch.cat([h_freqs.sin(), h_freqs.cos()], dim=-1)
        
        # Width positions
        w_pos = torch.arange(width).float()
        w_freqs = torch.outer(w_pos, self.inv_freq)
        w_emb = torch.cat([w_freqs.sin(), w_freqs.cos()], dim=-1)
        
        # Create 2D grid
        # Shape: (height, width, dim)
        h_emb = h_emb.unsqueeze(1).expand(-1, width, -1)
        w_emb = w_emb.unsqueeze(0).expand(height, -1, -1)
        
        emb_2d = torch.cat([h_emb, w_emb], dim=-1)
        
        # Flatten to (height * width, dim)
        emb_flat = emb_2d.reshape(-1, self.dim)
        
        # Split into cos and sin
        cos_cache = torch.cos(emb_flat)
        sin_cache = torch.sin(emb_flat)
        
        self.register_buffer('cos_cache', cos_cache, persistent=False)
        self.register_buffer('sin_cache', sin_cache, persistent=False)
        self.register_buffer('emb_2d', emb_2d, persistent=False)
    
    def _rotate_half(self, x):
        """Rotate half the hidden dims."""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, q, k, height, width):
        """
        Apply 2D rotary embeddings.
        
        :param q: Query tensor (batch, num_heads, seq_len, head_dim)
        :param k: Key tensor (batch, num_heads, seq_len, head_dim)
        :param height: Image height in patches
        :param width: Image width in patches
        :return: Tuple of (rotated_q, rotated_k)
        """
        seq_len = q.shape[2]
        
        # Rebuild cache if needed
        if height > self.max_height or width > self.max_width:
            self._build_cache(height, width)
        
        # Get position embeddings for current resolution
        num_patches = height * width
        
        if num_patches != seq_len:
            # Handle CLS token or different seq length
            # Assume first token is CLS, rest are patches
            cos = self.cos_cache[:num_patches].unsqueeze(0).unsqueeze(0)
            sin = self.sin_cache[:num_patches].unsqueeze(0).unsqueeze(0)
            
            if seq_len > num_patches:
                # Pad for CLS token with zeros (no rotation)
                pad_len = seq_len - num_patches
                cos = F.pad(cos, (0, 0, pad_len, 0), value=1.0)
                sin = F.pad(sin, (0, 0, pad_len, 0), value=0.0)
        else:
            cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)
            sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        
        # Apply rotation
        q_embed = q * cos + self._rotate_half(q) * sin
        k_embed = k * cos + self._rotate_half(k) * sin
        
        return q_embed, k_embed


# Test parameters
batch_size = 8
num_heads = 16
height = 14  # 14x14 = 196 patches (typical ViT)
width = 14
seq_len = height * width
head_dim = 64

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    return [q, k, height, width]

def get_init_inputs():
    return [head_dim, height * 2, width * 2]

