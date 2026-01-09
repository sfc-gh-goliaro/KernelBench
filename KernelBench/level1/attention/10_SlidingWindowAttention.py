import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Sliding Window Attention
    
    Used by: Mistral, Gemma-2, Longformer
    
    Local attention where each token only attends to a fixed window of
    nearby tokens. This reduces memory and compute from O(n²) to O(n*w)
    where w is the window size.
    
    Shapes:
        Input: (batch_size, seq_len, hidden_size)
        Output: (batch_size, seq_len, hidden_size)
    """
    
    def __init__(self, hidden_size: int, num_heads: int, window_size: int = 4096, dropout: float = 0.0):
        """
        Initialize sliding window attention.
        
        Args:
            hidden_size: Model hidden dimension
            num_heads: Number of attention heads
            window_size: Size of attention window
            dropout: Attention dropout probability
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
        self.dropout = dropout
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def _create_sliding_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create sliding window causal mask."""
        # Start with causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        
        # Add sliding window constraint: can only attend to positions within window
        # Position i can attend to positions max(0, i - window_size + 1) to i
        for i in range(seq_len):
            start = max(0, i - self.window_size + 1)
            if start > 0:
                mask[i, :start] = 1
        
        return mask.bool()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with sliding window attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to (batch, num_heads, seq, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply sliding window causal mask
        mask = self._create_sliding_window_mask(seq_len, x.device)
        scores = scores.masked_fill(mask, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        if self.dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        return self.o_proj(attn_output)


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 4096
hidden_size = 4096
num_heads = 32
window_size = 4096

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    x = torch.randn(batch_size, seq_length, hidden_size, device='cuda')
    return [x]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [hidden_size, num_heads, window_size]

