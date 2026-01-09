import torch
import torch.nn as nn

class Model(nn.Module):
    """
    QK Normalization
    
    Used by: Llama 4, some Gemma variants
    
    Applies RMS normalization to query and key tensors before attention.
    Helps stabilize attention scores, especially with large head dimensions.
    
    Shapes:
        Input Q: (batch_size, seq_length, num_heads, head_dim)
        Input K: (batch_size, seq_length, num_kv_heads, head_dim)
        Output Q: (batch_size, seq_length, num_heads, head_dim)
        Output K: (batch_size, seq_length, num_kv_heads, head_dim)
    """
    
    def __init__(self, head_dim: int = 128, eps: float = 1e-6):
        """
        Initialize QK Norm.
        
        Args:
            head_dim: Dimension per attention head
            eps: Small constant for numerical stability
        """
        super(Model, self).__init__()
        self.head_dim = head_dim
        self.eps = eps
        
        # Separate learnable scales for Q and K
        self.q_scale = nn.Parameter(torch.ones(head_dim))
        self.k_scale = nn.Parameter(torch.ones(head_dim))
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple:
        """
        Apply QK normalization.
        
        Args:
            q: Query tensor (batch_size, seq_length, num_heads, head_dim)
            k: Key tensor (batch_size, seq_length, num_kv_heads, head_dim)
            
        Returns:
            Tuple of (normalized_q, normalized_k)
        """
        # RMS normalize Q
        q_variance = q.pow(2).mean(-1, keepdim=True)
        q_normed = q * torch.rsqrt(q_variance + self.eps) * self.q_scale
        
        # RMS normalize K
        k_variance = k.pow(2).mean(-1, keepdim=True)
        k_normed = k * torch.rsqrt(k_variance + self.eps) * self.k_scale
        
        return q_normed, k_normed


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
    q = torch.randn(batch_size, seq_length, num_heads, head_dim, device='cuda')
    k = torch.randn(batch_size, seq_length, num_kv_heads, head_dim, device='cuda')
    return [q, k]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [head_dim]

