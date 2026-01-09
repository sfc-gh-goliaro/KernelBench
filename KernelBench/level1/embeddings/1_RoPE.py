import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    
    Used by: Llama, Qwen, Mistral, Gemma, Yi, DeepSeek, Phi
    
    Applies rotary position embeddings to query and key tensors by rotating
    pairs of dimensions using precomputed cos/sin values based on position.
    
    Shapes:
        Input: (batch_size, seq_len, num_heads, head_dim) for q and k
        Output: (batch_size, seq_len, num_heads, head_dim) for q and k (rotated)
    """
    
    def __init__(self, head_dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        """
        Initialize RoPE.
        
        Args:
            head_dim: Dimension of each attention head (must be even)
            max_seq_len: Maximum sequence length for precomputed embeddings
            base: Base for the frequency computation
        """
        super(Model, self).__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos and sin for all positions
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor = None) -> tuple:
        """
        Apply rotary embeddings to q and k.
        
        Args:
            q: Query tensor of shape (batch_size, seq_len, num_heads, head_dim)
            k: Key tensor of shape (batch_size, seq_len, num_heads, head_dim)
            position_ids: Optional position indices of shape (batch_size, seq_len)
            
        Returns:
            Tuple of (rotated_q, rotated_k) with same shapes as inputs
        """
        seq_len = q.shape[1]
        
        if position_ids is None:
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]
        else:
            cos = self.cos_cached[position_ids]
            sin = self.sin_cached[position_ids]
        
        # Reshape for broadcasting: (seq_len, head_dim) -> (1, seq_len, 1, head_dim)
        if position_ids is None:
            cos = cos.unsqueeze(0).unsqueeze(2)
            sin = sin.unsqueeze(0).unsqueeze(2)
        else:
            cos = cos.unsqueeze(2)
            sin = sin.unsqueeze(2)
        
        q_rotated = self._apply_rotary(q, cos, sin)
        k_rotated = self._apply_rotary(k, cos, sin)
        
        return q_rotated, k_rotated
    
    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding to a single tensor."""
        # Split into two halves
        x1 = x[..., :self.head_dim // 2]
        x2 = x[..., self.head_dim // 2:]
        
        # Rotate
        rotated = torch.cat((-x2, x1), dim=-1)
        
        return x * cos + rotated * sin


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 2048
num_heads = 32
head_dim = 128

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    q = torch.randn(batch_size, seq_length, num_heads, head_dim, device='cuda')
    k = torch.randn(batch_size, seq_length, num_heads, head_dim, device='cuda')
    return [q, k]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [head_dim]

