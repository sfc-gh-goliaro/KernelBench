import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Multi-Head Latent Attention (MLA)
    
    Used by: DeepSeek-V2, DeepSeek-V3
    
    MLA compresses KV into a low-rank latent space before attention,
    reducing KV cache memory while maintaining model quality.
    The compressed KV is projected back up during attention computation.
    
    Shapes:
        Input: (batch_size, seq_len, hidden_size)
        Output: (batch_size, seq_len, hidden_size)
    """
    
    def __init__(self, hidden_size: int, num_heads: int, kv_lora_rank: int = 512, 
                 q_lora_rank: int = None, dropout: float = 0.0):
        """
        Initialize MLA.
        
        Args:
            hidden_size: Model hidden dimension
            num_heads: Number of attention heads
            kv_lora_rank: Rank for KV compression
            q_lora_rank: Rank for Q compression (optional)
            dropout: Attention dropout probability
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.dropout = dropout
        
        # Q projection (optionally compressed)
        if q_lora_rank is not None:
            self.q_down_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
            self.q_up_proj = nn.Linear(q_lora_rank, num_heads * self.head_dim, bias=False)
        else:
            self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        
        # KV compression: project to low-rank latent, then back up
        self.kv_down_proj = nn.Linear(hidden_size, kv_lora_rank, bias=False)
        self.k_up_proj = nn.Linear(kv_lora_rank, num_heads * self.head_dim, bias=False)
        self.v_up_proj = nn.Linear(kv_lora_rank, num_heads * self.head_dim, bias=False)
        
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-head latent attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q
        if self.q_lora_rank is not None:
            q = self.q_up_proj(self.q_down_proj(x))
        else:
            q = self.q_proj(x)
        
        # Compress KV to latent space
        kv_compressed = self.kv_down_proj(x)
        
        # Project back up to K and V
        k = self.k_up_proj(kv_compressed)
        v = self.v_up_proj(kv_compressed)
        
        # Reshape to (batch, num_heads, seq, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
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
seq_length = 2048
hidden_size = 4096
num_heads = 32
kv_lora_rank = 512

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    x = torch.randn(batch_size, seq_length, hidden_size, device='cuda')
    return [x]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [hidden_size, num_heads, kv_lora_rank]

