import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BERTSelfAttention(nn.Module):
    """BERT-style bidirectional multi-head self-attention."""
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn = attn + attention_mask
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return out


class BERTSelfOutput(nn.Module):
    """BERT attention output projection with residual and LayerNorm."""
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BERTIntermediate(nn.Module):
    """BERT FFN first layer with GELU."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.dense(x))


class BERTOutput(nn.Module):
    """BERT FFN output with residual and LayerNorm."""
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Model(nn.Module):
    """
    BERT Encoder Layer
    
    The core repeated block in BERT-style encoder-only transformers.
    Used by: BERT, RoBERTa, ALBERT, DistilBERT, DeBERTa
    
    Architecture (Post-LN):
        x -> Self-Attention -> + residual -> LayerNorm
          -> FFN (GELU) -> + residual -> LayerNorm
    
    Key features:
    - Bidirectional attention (no causal mask)
    - Post-LayerNorm (add then norm)
    - GELU activation
    """
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.attention = BERTSelfAttention(hidden_size, num_heads, dropout)
        self.attention_output = BERTSelfOutput(hidden_size, dropout)
        self.intermediate = BERTIntermediate(hidden_size, intermediate_size)
        self.output = BERTOutput(hidden_size, intermediate_size, dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention with post-norm
        attention_output = self.attention(x, attention_mask)
        attention_output = self.attention_output(attention_output, x)
        
        # FFN with post-norm
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        
        return layer_output


# Benchmark configuration (BERT-large dimensions)
batch_size = 32
seq_len = 512
hidden_size = 1024
num_heads = 16
intermediate_size = 4096
dropout = 0.0  # Set to 0 for deterministic benchmarking

def get_inputs():
    return [torch.randn(batch_size, seq_len, hidden_size)]

def get_init_inputs():
    return [hidden_size, num_heads, intermediate_size, dropout]

