import torch
import torch.nn as nn
import math

class Model(nn.Module):
    """
    DeepSeek RoPE
    
    Used by: DeepSeek-V2, DeepSeek-V3
    
    DeepSeek-specific RoPE scaling with custom frequency adjustments
    and support for their MLA architecture.
    
    Shapes:
        Input: (batch_size, seq_len, num_heads, head_dim) for q and k
        Output: (batch_size, seq_len, num_heads, head_dim) for q and k
    """
    
    def __init__(self, head_dim: int, max_seq_len: int = 8192, base: float = 10000.0,
                 scaling_factor: float = 1.0, beta: float = 32.0,
                 mscale: float = 1.0, mscale_all_dim: float = 0.0):
        """
        Initialize DeepSeek RoPE.
        
        Args:
            head_dim: Dimension of each attention head
            max_seq_len: Maximum sequence length
            base: Base for frequency computation
            scaling_factor: Context length scaling factor
            beta: Scaling smoothness parameter
            mscale: Magnitude scaling factor
            mscale_all_dim: Magnitude scaling for all dimensions
        """
        super(Model, self).__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_factor = scaling_factor
        self.beta = beta
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        
        # Compute DeepSeek-scaled inverse frequencies
        inv_freq = self._compute_inv_freq()
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos and sin
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Apply magnitude scaling
        mag_scale = self._compute_magnitude_scale()
        self.register_buffer('cos_cached', emb.cos() * mag_scale)
        self.register_buffer('sin_cached', emb.sin() * mag_scale)
    
    def _compute_inv_freq(self) -> torch.Tensor:
        """Compute DeepSeek-specific inverse frequencies."""
        dim = self.head_dim
        
        # Base inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        
        if self.scaling_factor == 1.0:
            return inv_freq
        
        # DeepSeek uses smooth interpolation based on beta
        # Higher frequency dimensions are scaled less
        dim_idx = torch.arange(0, dim // 2).float()
        smooth_factor = 1.0 / (1.0 + (dim_idx / (self.beta * (dim / 2))).pow(2))
        
        # Interpolate between no scaling and full scaling
        scale = 1.0 + (self.scaling_factor - 1.0) * smooth_factor
        
        return inv_freq / scale
    
    def _compute_magnitude_scale(self) -> float:
        """Compute attention magnitude scaling."""
        if self.mscale_all_dim > 0:
            return self.mscale * (self.mscale_all_dim ** 0.5)
        if self.scaling_factor <= 1.0:
            return 1.0
        return 0.1 * self.mscale * math.log(self.scaling_factor) + 1.0
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor = None) -> tuple:
        """
        Apply DeepSeek RoPE to q and k.
        
        Args:
            q: Query tensor (batch, seq, num_heads, head_dim)
            k: Key tensor (batch, seq, num_heads, head_dim)
            position_ids: Optional position indices
            
        Returns:
            Tuple of rotated (q, k)
        """
        seq_len = q.shape[1]
        
        if position_ids is None:
            cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(2)
            sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(2)
        else:
            cos = self.cos_cached[position_ids].unsqueeze(2)
            sin = self.sin_cached[position_ids].unsqueeze(2)
        
        q_rotated = self._apply_rotary(q, cos, sin)
        k_rotated = self._apply_rotary(k, cos, sin)
        
        return q_rotated, k_rotated
    
    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding."""
        x1 = x[..., :self.head_dim // 2]
        x2 = x[..., self.head_dim // 2:]
        rotated = torch.cat((-x2, x1), dim=-1)
        return x * cos + rotated * sin


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 8192
num_heads = 32
head_dim = 128

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    q = torch.randn(batch_size, seq_length, num_heads, head_dim, device='cuda')
    k = torch.randn(batch_size, seq_length, num_heads, head_dim, device='cuda')
    return [q, k]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [head_dim, 8192, 10000.0, 2.0]

