import torch
import torch.nn as nn
import math

class Model(nn.Module):
    """
    3D Rotary Position Embedding
    
    Used by: Qwen2-VL (M-RoPE)
    
    3D rotary position embedding for spatial (height, width) and
    temporal dimensions. Used for video and multi-image understanding.
    
    Shapes:
        Input q, k: (batch, seq_len, num_heads, head_dim)
        Output: (batch, seq_len, num_heads, head_dim)
    """
    
    def __init__(self, head_dim: int, max_temporal: int = 64, max_height: int = 64, 
                 max_width: int = 64, base: float = 10000.0):
        """
        Initialize 3D RoPE.
        
        Args:
            head_dim: Dimension of each attention head
            max_temporal: Maximum temporal positions
            max_height: Maximum height positions
            max_width: Maximum width positions
            base: Base for frequency computation
        """
        super(Model, self).__init__()
        self.head_dim = head_dim
        self.max_temporal = max_temporal
        self.max_height = max_height
        self.max_width = max_width
        
        # Split head_dim into 3 parts for T, H, W
        assert head_dim % 3 == 0, "head_dim must be divisible by 3 for 3D RoPE"
        self.dim_t = self.dim_h = self.dim_w = head_dim // 3
        
        # Precompute inverse frequencies for each dimension
        inv_freq_t = 1.0 / (base ** (torch.arange(0, self.dim_t, 2).float() / self.dim_t))
        inv_freq_h = 1.0 / (base ** (torch.arange(0, self.dim_h, 2).float() / self.dim_h))
        inv_freq_w = 1.0 / (base ** (torch.arange(0, self.dim_w, 2).float() / self.dim_w))
        
        self.register_buffer('inv_freq_t', inv_freq_t)
        self.register_buffer('inv_freq_h', inv_freq_h)
        self.register_buffer('inv_freq_w', inv_freq_w)
    
    def _compute_rope(self, positions: torch.Tensor, inv_freq: torch.Tensor) -> tuple:
        """Compute cos and sin for given positions."""
        freqs = torch.outer(positions.float(), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()
    
    def _apply_rotary_1d(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding to a slice of x."""
        dim = x.shape[-1]
        x1 = x[..., :dim // 2]
        x2 = x[..., dim // 2:]
        rotated = torch.cat((-x2, x1), dim=-1)
        return x * cos + rotated * sin
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, 
                position_ids_t: torch.Tensor, position_ids_h: torch.Tensor, 
                position_ids_w: torch.Tensor) -> tuple:
        """
        Apply 3D RoPE to q and k.
        
        Args:
            q: Query tensor (batch, seq, num_heads, head_dim)
            k: Key tensor (batch, seq, num_heads, head_dim)
            position_ids_t: Temporal positions (batch, seq)
            position_ids_h: Height positions (batch, seq)
            position_ids_w: Width positions (batch, seq)
            
        Returns:
            Tuple of rotated (q, k)
        """
        batch_size, seq_len = q.shape[:2]
        
        # Split q and k into temporal, height, width parts
        q_t = q[..., :self.dim_t]
        q_h = q[..., self.dim_t:self.dim_t + self.dim_h]
        q_w = q[..., self.dim_t + self.dim_h:]
        
        k_t = k[..., :self.dim_t]
        k_h = k[..., self.dim_t:self.dim_t + self.dim_h]
        k_w = k[..., self.dim_t + self.dim_h:]
        
        # Compute and apply RoPE for each dimension
        # Temporal
        cos_t, sin_t = self._compute_rope(position_ids_t.view(-1), self.inv_freq_t)
        cos_t = cos_t.view(batch_size, seq_len, 1, -1)
        sin_t = sin_t.view(batch_size, seq_len, 1, -1)
        q_t = self._apply_rotary_1d(q_t, cos_t, sin_t)
        k_t = self._apply_rotary_1d(k_t, cos_t, sin_t)
        
        # Height
        cos_h, sin_h = self._compute_rope(position_ids_h.view(-1), self.inv_freq_h)
        cos_h = cos_h.view(batch_size, seq_len, 1, -1)
        sin_h = sin_h.view(batch_size, seq_len, 1, -1)
        q_h = self._apply_rotary_1d(q_h, cos_h, sin_h)
        k_h = self._apply_rotary_1d(k_h, cos_h, sin_h)
        
        # Width
        cos_w, sin_w = self._compute_rope(position_ids_w.view(-1), self.inv_freq_w)
        cos_w = cos_w.view(batch_size, seq_len, 1, -1)
        sin_w = sin_w.view(batch_size, seq_len, 1, -1)
        q_w = self._apply_rotary_1d(q_w, cos_w, sin_w)
        k_w = self._apply_rotary_1d(k_w, cos_w, sin_w)
        
        # Concatenate back
        q = torch.cat([q_t, q_h, q_w], dim=-1)
        k = torch.cat([k_t, k_h, k_w], dim=-1)
        
        return q, k


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 576  # 24x24 patches
num_heads = 32
head_dim = 96  # Must be divisible by 3

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    q = torch.randn(batch_size, seq_length, num_heads, head_dim, device='cuda')
    k = torch.randn(batch_size, seq_length, num_heads, head_dim, device='cuda')
    
    # Generate 3D position IDs
    h = w = int(math.sqrt(seq_length))
    position_ids_t = torch.zeros(batch_size, seq_length, dtype=torch.long, device='cuda')
    position_ids_h = torch.arange(h, device='cuda').repeat_interleave(w).unsqueeze(0).expand(batch_size, -1)
    position_ids_w = torch.arange(w, device='cuda').repeat(h).unsqueeze(0).expand(batch_size, -1)
    
    return [q, k, position_ids_t, position_ids_h, position_ids_w]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [head_dim]

