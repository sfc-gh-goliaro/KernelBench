import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Linear Scaling RoPE
    
    Used by: Position interpolation models
    
    Simple linear interpolation of position indices for longer sequences.
    Divides position indices by the scaling factor.
    
    Shapes:
        Input: (batch_size, seq_len, num_heads, head_dim) for q and k
        Output: (batch_size, seq_len, num_heads, head_dim) for q and k
    """
    
    def __init__(self, head_dim: int, max_seq_len: int = 8192, base: float = 10000.0,
                 scaling_factor: float = 1.0):
        """
        Initialize Linear Scaling RoPE.
        
        Args:
            head_dim: Dimension of each attention head
            max_seq_len: Maximum sequence length
            base: Base for frequency computation
            scaling_factor: Factor to divide positions by
        """
        super(Model, self).__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_factor = scaling_factor
        
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute with scaled positions
        t = torch.arange(max_seq_len).float() / scaling_factor
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor = None) -> tuple:
        """
        Apply linear-scaled RoPE to q and k.
        
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

