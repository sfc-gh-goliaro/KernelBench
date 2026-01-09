import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Fused QKV Parallel Linear Projection
    
    Used by: vLLM, TensorRT-LLM, most inference frameworks
    
    Fused projection for Query, Key, Value in a single GEMM operation.
    Supports tensor parallelism by computing QKV for local heads only.
    
    Shapes:
        Input: (batch_size, seq_length, hidden_size)
        Output: (batch_size, seq_length, num_heads * head_dim * 3) for MHA
                or separate Q, K, V tensors
    """
    
    def __init__(self, hidden_size: int = 4096, num_heads: int = 32, 
                 num_kv_heads: int = 8, head_dim: int = 128):
        """
        Initialize QKV Parallel Linear.
        
        Args:
            hidden_size: Input hidden dimension
            num_heads: Number of query heads
            num_kv_heads: Number of key/value heads (for GQA)
            head_dim: Dimension per head
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        
        # Output sizes for Q, K, V
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        total_size = self.q_size + 2 * self.kv_size
        
        # Fused QKV projection
        self.qkv_proj = nn.Linear(hidden_size, total_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Compute fused QKV projection.
        
        Args:
            x: Input tensor (batch_size, seq_length, hidden_size)
            
        Returns:
            Tuple of (Q, K, V) tensors
        """
        batch_size, seq_length, _ = x.shape
        
        # Fused projection
        qkv = self.qkv_proj(x)
        
        # Split into Q, K, V
        q = qkv[:, :, :self.q_size]
        k = qkv[:, :, self.q_size:self.q_size + self.kv_size]
        v = qkv[:, :, self.q_size + self.kv_size:]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_length, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_length, self.num_kv_heads, self.head_dim)
        
        return q, k, v


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 2048
hidden_size = 4096
num_heads = 32
num_kv_heads = 8
head_dim = 128

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    x = torch.randn(batch_size, seq_length, hidden_size, device='cuda')
    return [x]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [hidden_size, num_heads, num_kv_heads, head_dim]

