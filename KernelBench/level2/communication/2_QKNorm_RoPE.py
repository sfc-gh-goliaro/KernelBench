import torch
import torch.nn as nn
import math

class Model(nn.Module):
    """
    Fused QK Normalization + RoPE
    
    Used by: DeepSeek-V2/V3, models with QK-norm
    
    Fuses QK normalization with rotary position embedding application.
    This combines two sequential operations on Q and K tensors into
    a single kernel pass.
    
    Found in: TensorRT-LLM (fusedQKNormRopeKernel), vLLM
    
    Shapes:
        Input Q: (batch_size, num_heads, seq_len, head_dim)
        Input K: (batch_size, num_kv_heads, seq_len, head_dim)
        Output Q: (batch_size, num_heads, seq_len, head_dim)
        Output K: (batch_size, num_kv_heads, seq_len, head_dim)
    """
    
    def __init__(self, head_dim: int, num_heads: int, num_kv_heads: int, 
                 max_seq_len: int = 8192, base: float = 10000.0, eps: float = 1e-6):
        """
        Initialize fused QKNorm + RoPE.
        
        Args:
            head_dim: Dimension per head
            num_heads: Number of query heads
            num_kv_heads: Number of key/value heads
            max_seq_len: Maximum sequence length
            base: RoPE base frequency
            eps: Epsilon for normalization
        """
        super(Model, self).__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.eps = eps
        
        # QK norm weights
        self.q_norm_weight = nn.Parameter(torch.ones(head_dim))
        self.k_norm_weight = nn.Parameter(torch.ones(head_dim))
        
        # Precompute RoPE frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos/sin cache
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())
    
    def _rms_norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization."""
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * weight
    
    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embedding."""
        x1, x2 = x[..., ::2], x[..., 1::2]
        
        # Reshape cos/sin for broadcasting
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, head_dim/2)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        # Apply rotation
        rotated = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1).flatten(-2)
        
        return rotated
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, 
                position_ids: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fused QK normalization + RoPE.
        
        Args:
            q: Query tensor (batch, num_heads, seq_len, head_dim)
            k: Key tensor (batch, num_kv_heads, seq_len, head_dim)
            position_ids: Optional position IDs (batch, seq_len)
            
        Returns:
            Tuple of (normalized+rotated Q, normalized+rotated K)
        """
        seq_len = q.shape[2]
        
        # Get position embeddings
        if position_ids is None:
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]
        else:
            cos = self.cos_cached[position_ids]
            sin = self.sin_cached[position_ids]
        
        # Fused: QKNorm then RoPE
        q_normed = self._rms_norm(q, self.q_norm_weight)
        k_normed = self._rms_norm(k, self.k_norm_weight)
        
        q_out = self._apply_rope(q_normed, cos, sin)
        k_out = self._apply_rope(k_normed, cos, sin)
        
        return q_out, k_out


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 2048
num_heads = 32
num_kv_heads = 8
head_dim = 128

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    q = torch.randn(batch_size, num_heads, seq_length, head_dim, device='cuda')
    k = torch.randn(batch_size, num_kv_heads, seq_length, head_dim, device='cuda')
    return [q, k]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [head_dim, num_heads, num_kv_heads]

