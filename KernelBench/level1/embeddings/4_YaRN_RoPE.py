import torch
import torch.nn as nn
import math

class Model(nn.Module):
    """
    YaRN (Yet another RoPE extensioN) RoPE
    
    Used by: Llama with extended context, Yarn models
    
    YaRN scaling for extended context lengths with attention scaling.
    Combines NTK interpolation with attention temperature adjustment.
    
    Shapes:
        Input: (batch_size, seq_len, num_heads, head_dim) for q and k
        Output: (batch_size, seq_len, num_heads, head_dim) for q and k
    """
    
    def __init__(self, head_dim: int, max_seq_len: int = 8192, base: float = 10000.0,
                 scale: float = 1.0, original_max_seq_len: int = 4096,
                 beta_fast: float = 32.0, beta_slow: float = 1.0):
        """
        Initialize YaRN RoPE.
        
        Args:
            head_dim: Dimension of each attention head
            max_seq_len: Maximum (extended) sequence length
            base: Base for frequency computation
            scale: Context length scaling factor
            original_max_seq_len: Original model's max sequence length
            beta_fast: Fast dimension interpolation threshold
            beta_slow: Slow dimension interpolation threshold
        """
        super(Model, self).__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scale = scale
        self.original_max_seq_len = original_max_seq_len
        
        # YaRN parameters
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        
        # Compute scaling factor for attention
        self.mscale = self._compute_mscale(scale)
        
        # Precompute scaled frequencies
        self._precompute_freqs()
    
    def _compute_mscale(self, scale: float) -> float:
        """Compute attention temperature scaling."""
        if scale <= 1:
            return 1.0
        return 0.1 * math.log(scale) + 1.0
    
    def _yarn_find_correction_dim(self, num_rotations: int, dim: int, base: float, max_seq_len: int) -> float:
        """Find correction dimension for YaRN."""
        return (dim * math.log(max_seq_len / (num_rotations * 2 * math.pi))) / (2 * math.log(base))
    
    def _yarn_find_correction_range(self, low_rot: float, high_rot: float, dim: int, 
                                     base: float, max_seq_len: int) -> tuple:
        """Find correction range for interpolation."""
        low = math.floor(self._yarn_find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(self._yarn_find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)
    
    def _yarn_linear_ramp_mask(self, low: int, high: int, dim: int) -> torch.Tensor:
        """Create linear ramp mask for interpolation."""
        if low == high:
            high += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - low) / (high - low)
        return torch.clamp(linear_func, 0, 1)
    
    def _precompute_freqs(self):
        """Precompute YaRN-scaled frequencies."""
        dim = self.head_dim
        
        # Base inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        
        # Apply YaRN scaling
        low, high = self._yarn_find_correction_range(
            self.beta_fast, self.beta_slow, dim, self.base, self.original_max_seq_len
        )
        
        inv_freq_mask = 1 - self._yarn_linear_ramp_mask(low, high, dim // 2)
        inv_freq = inv_freq / self.scale * (1 - inv_freq_mask) + inv_freq * inv_freq_mask
        
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos and sin
        t = torch.arange(self.max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos() * self.mscale)
        self.register_buffer('sin_cached', emb.sin() * self.mscale)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor = None) -> tuple:
        """
        Apply YaRN RoPE to q and k.
        
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
    return [head_dim, 8192, 10000.0, 2.0, 4096]  # 2x context extension

