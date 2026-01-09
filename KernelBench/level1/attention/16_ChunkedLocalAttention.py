import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Chunked Local Attention
    
    Used by: Llama 4, Longformer-style models
    
    Attention that operates on local chunks/windows for efficiency.
    Reduces memory from O(n^2) to O(n * chunk_size) for long sequences.
    
    Shapes:
        Input Q, K, V: (batch_size, seq_length, num_heads, head_dim)
        Output: (batch_size, seq_length, num_heads, head_dim)
    """
    
    def __init__(self, num_heads: int = 32, head_dim: int = 128, 
                 chunk_size: int = 512):
        """
        Initialize Chunked Local Attention.
        
        Args:
            num_heads: Number of attention heads
            head_dim: Dimension per head
            chunk_size: Size of each local attention chunk
        """
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.scale = 1.0 / math.sqrt(head_dim)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute chunked local attention.
        
        Args:
            q: Query tensor (batch_size, seq_length, num_heads, head_dim)
            k: Key tensor (batch_size, seq_length, num_heads, head_dim)
            v: Value tensor (batch_size, seq_length, num_heads, head_dim)
            
        Returns:
            Attention output (batch_size, seq_length, num_heads, head_dim)
        """
        batch_size, seq_length, num_heads, head_dim = q.shape
        
        # Pad sequence to be divisible by chunk_size
        pad_len = (self.chunk_size - seq_length % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            q = F.pad(q, (0, 0, 0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, 0, 0, pad_len))
        
        padded_seq_length = q.shape[1]
        num_chunks = padded_seq_length // self.chunk_size
        
        # Reshape to chunks: (batch, num_chunks, chunk_size, heads, head_dim)
        q = q.view(batch_size, num_chunks, self.chunk_size, num_heads, head_dim)
        k = k.view(batch_size, num_chunks, self.chunk_size, num_heads, head_dim)
        v = v.view(batch_size, num_chunks, self.chunk_size, num_heads, head_dim)
        
        # Transpose for attention: (batch, num_chunks, heads, chunk_size, head_dim)
        q = q.permute(0, 1, 3, 2, 4)
        k = k.permute(0, 1, 3, 2, 4)
        v = v.permute(0, 1, 3, 2, 4)
        
        # Local attention within each chunk
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask within each chunk
        chunk_mask = torch.triu(
            torch.ones(self.chunk_size, self.chunk_size, device=q.device),
            diagonal=1
        ).bool()
        attn_weights.masked_fill_(chunk_mask, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        # Reshape back: (batch, num_chunks, heads, chunk_size, head_dim)
        output = output.permute(0, 1, 3, 2, 4)  # (batch, num_chunks, chunk_size, heads, head_dim)
        output = output.contiguous().view(batch_size, padded_seq_length, num_heads, head_dim)
        
        # Remove padding
        if pad_len > 0:
            output = output[:, :seq_length, :, :]
        
        return output


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 4
seq_length = 4096
num_heads = 32
head_dim = 128
chunk_size = 512

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    q = torch.randn(batch_size, seq_length, num_heads, head_dim, device='cuda')
    k = torch.randn(batch_size, seq_length, num_heads, head_dim, device='cuda')
    v = torch.randn(batch_size, seq_length, num_heads, head_dim, device='cuda')
    return [q, k, v]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [num_heads, head_dim, chunk_size]

