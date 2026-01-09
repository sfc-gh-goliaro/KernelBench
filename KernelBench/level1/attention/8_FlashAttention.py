import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Flash Attention (Reference Implementation)
    
    Used by: All transformers (training & inference)
    
    Memory-efficient attention that avoids materializing the full attention
    matrix. This is a reference implementation; production uses optimized
    CUDA kernels (flash_attn library).
    
    Shapes:
        Input Q, K, V: (batch_size, num_heads, seq_len, head_dim)
        Output: (batch_size, num_heads, seq_len, head_dim)
    """
    
    def __init__(self, head_dim: int, dropout: float = 0.0, causal: bool = True):
        """
        Initialize Flash Attention.
        
        Args:
            head_dim: Dimension of each attention head
            dropout: Attention dropout probability
            causal: Whether to use causal masking
        """
        super(Model, self).__init__()
        self.head_dim = head_dim
        self.dropout = dropout
        self.causal = causal
        self.scale = 1.0 / math.sqrt(head_dim)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Flash attention forward pass.
        
        Uses PyTorch's scaled_dot_product_attention which dispatches to
        flash attention when available.
        
        Args:
            q: Query tensor (batch, num_heads, seq_len, head_dim)
            k: Key tensor (batch, num_heads, seq_len, head_dim)
            v: Value tensor (batch, num_heads, seq_len, head_dim)
            
        Returns:
            Attention output (batch, num_heads, seq_len, head_dim)
        """
        # Use PyTorch's native flash attention implementation
        dropout_p = self.dropout if self.training else 0.0
        
        output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=dropout_p,
            is_causal=self.causal,
            scale=self.scale
        )
        
        return output


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
num_heads = 32
seq_length = 2048
head_dim = 128

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    q = torch.randn(batch_size, num_heads, seq_length, head_dim, device='cuda')
    k = torch.randn(batch_size, num_heads, seq_length, head_dim, device='cuda')
    v = torch.randn(batch_size, num_heads, seq_length, head_dim, device='cuda')
    return [q, k, v]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [head_dim]

