import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Cross-Attention
    
    Used by: T5, BART, Whisper decoder, encoder-decoder models
    
    Cross-attention where queries come from decoder and keys/values
    come from encoder output.
    
    Shapes:
        decoder_hidden: (batch_size, tgt_len, hidden_size)
        encoder_hidden: (batch_size, src_len, hidden_size)
        Output: (batch_size, tgt_len, hidden_size)
    """
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        """
        Initialize cross-attention.
        
        Args:
            hidden_size: Model hidden dimension
            num_heads: Number of attention heads
            dropout: Attention dropout probability
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, decoder_hidden: torch.Tensor, encoder_hidden: torch.Tensor,
                encoder_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Cross-attention forward pass.
        
        Args:
            decoder_hidden: Decoder states (batch, tgt_len, hidden)
            encoder_hidden: Encoder states (batch, src_len, hidden)
            encoder_mask: Optional mask for encoder padding (batch, src_len)
            
        Returns:
            Output tensor (batch, tgt_len, hidden)
        """
        batch_size, tgt_len, _ = decoder_hidden.shape
        src_len = encoder_hidden.shape[1]
        
        # Q from decoder, K/V from encoder
        q = self.q_proj(decoder_hidden)
        k = self.k_proj(encoder_hidden)
        v = self.v_proj(encoder_hidden)
        
        # Reshape to (batch, num_heads, seq, head_dim)
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply encoder mask if provided
        if encoder_mask is not None:
            # encoder_mask: (batch, src_len) -> (batch, 1, 1, src_len)
            mask = encoder_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        if self.dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.hidden_size)
        
        return self.o_proj(attn_output)


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
tgt_len = 512
src_len = 1024
hidden_size = 4096
num_heads = 32

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    decoder_hidden = torch.randn(batch_size, tgt_len, hidden_size, device='cuda')
    encoder_hidden = torch.randn(batch_size, src_len, hidden_size, device='cuda')
    return [decoder_hidden, encoder_hidden]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [hidden_size, num_heads]

