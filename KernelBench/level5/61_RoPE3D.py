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
    3D Rotary Position Embedding for Video/Spatiotemporal Understanding.
    
    Extends RoPE to three dimensions (height, width, time) for
    video transformers and document understanding models.
    
    Based on: PaddleOCR-VL and video transformer architectures
    """
    def __init__(self, dim, max_height=64, max_width=64, max_time=32, base=10000):
        """
        :param dim: Dimension of embeddings (must be divisible by 6)
        :param max_height: Maximum height in patches
        :param max_width: Maximum width in patches
        :param max_time: Maximum temporal length
        :param base: Base for frequency computation
        """
        super(Model, self).__init__()
        assert dim % 6 == 0, "Dimension must be divisible by 6 for 3D RoPE"
        
        self.dim = dim
        self.max_height = max_height
        self.max_width = max_width
        self.max_time = max_time
        
        # Dimension split: 1/3 for each axis
        self.dim_per_axis = dim // 3
        
        # Precompute frequencies for each axis
        half = self.dim_per_axis // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, half, 2).float() / half))
        self.register_buffer('inv_freq', inv_freq)
        
        # Build 3D cache
        self._build_cache(max_height, max_width, max_time)
    
    def _build_cache(self, height, width, time):
        """Build sin/cos cache for 3D positions."""
        # Height embeddings
        h_pos = torch.arange(height).float()
        h_freqs = torch.outer(h_pos, self.inv_freq)
        h_emb = torch.cat([h_freqs.sin(), h_freqs.cos()], dim=-1)
        
        # Width embeddings
        w_pos = torch.arange(width).float()
        w_freqs = torch.outer(w_pos, self.inv_freq)
        w_emb = torch.cat([w_freqs.sin(), w_freqs.cos()], dim=-1)
        
        # Time embeddings
        t_pos = torch.arange(time).float()
        t_freqs = torch.outer(t_pos, self.inv_freq)
        t_emb = torch.cat([t_freqs.sin(), t_freqs.cos()], dim=-1)
        
        # Create 3D grid (time, height, width, dim)
        # Expand each embedding to the full 3D grid
        t_emb = t_emb.view(time, 1, 1, -1).expand(-1, height, width, -1)
        h_emb = h_emb.view(1, height, 1, -1).expand(time, -1, width, -1)
        w_emb = w_emb.view(1, 1, width, -1).expand(time, height, -1, -1)
        
        # Concatenate along dim
        emb_3d = torch.cat([t_emb, h_emb, w_emb], dim=-1)
        
        # Flatten to (time * height * width, dim)
        emb_flat = emb_3d.reshape(-1, self.dim)
        
        self.register_buffer('cos_cache', torch.cos(emb_flat), persistent=False)
        self.register_buffer('sin_cache', torch.sin(emb_flat), persistent=False)
    
    def _rotate_half(self, x):
        """Rotate half the hidden dims."""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, q, k, time_len, height, width):
        """
        Apply 3D rotary embeddings.
        
        :param q: Query tensor (batch, num_heads, seq_len, head_dim)
        :param k: Key tensor (batch, num_heads, seq_len, head_dim)
        :param time_len: Number of frames
        :param height: Height in patches
        :param width: Width in patches
        :return: Tuple of (rotated_q, rotated_k)
        """
        seq_len = q.shape[2]
        expected_len = time_len * height * width
        
        # Rebuild cache if needed
        if (time_len > self.max_time or height > self.max_height or 
            width > self.max_width):
            self._build_cache(max(height, self.max_height), 
                            max(width, self.max_width),
                            max(time_len, self.max_time))
        
        # Get embeddings
        cos = self.cos_cache[:expected_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[:expected_len].unsqueeze(0).unsqueeze(0)
        
        # Handle sequence length mismatch (e.g., with CLS token)
        if seq_len != expected_len:
            if seq_len > expected_len:
                # Pad for extra tokens (CLS, etc.)
                pad_len = seq_len - expected_len
                cos = F.pad(cos, (0, 0, pad_len, 0), value=1.0)
                sin = F.pad(sin, (0, 0, pad_len, 0), value=0.0)
            else:
                cos = cos[:, :, :seq_len]
                sin = sin[:, :, :seq_len]
        
        # Apply rotation
        q_embed = q * cos + self._rotate_half(q) * sin
        k_embed = k * cos + self._rotate_half(k) * sin
        
        return q_embed, k_embed


# Test parameters
batch_size = 4
num_heads = 12
time_len = 8    # Video frames
height = 14     # Patches per frame
width = 14
seq_len = time_len * height * width
head_dim = 96   # Must be divisible by 6

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
    return [q, k, time_len, height, width]

def get_init_inputs():
    return [head_dim, height, width, time_len]

