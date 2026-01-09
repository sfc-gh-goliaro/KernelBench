import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class T5LayerNorm(nn.Module):
    """T5-style RMSNorm (no bias, no mean subtraction)."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class T5Attention(nn.Module):
    """T5 attention with relative position bias."""
    def __init__(self, hidden_size: int, num_heads: int, is_decoder: bool = False,
                 has_relative_attention_bias: bool = True, relative_attention_num_buckets: int = 32,
                 relative_attention_max_distance: int = 128, dropout: float = 0.1):
        super().__init__()
        self.is_decoder = is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        if has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(relative_attention_num_buckets, num_heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional, num_buckets, max_distance):
        """Compute relative position bucket."""
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).long() * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).long()
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length: int, key_length: int, device):
        """Compute relative position bias."""
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=not self.is_decoder,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance
        )
        
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values

    def forward(self, x: torch.Tensor, encoder_hidden_states: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if encoder_hidden_states is not None:
            # Cross-attention
            k = self.k(encoder_hidden_states).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v(encoder_hidden_states).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            key_length = encoder_hidden_states.shape[1]
        else:
            # Self-attention
            k = self.k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_length = seq_len
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Add relative position bias (only for self-attention in first layer)
        if self.has_relative_attention_bias and encoder_hidden_states is None:
            position_bias = self.compute_bias(seq_len, key_length, x.device)
            scores = scores + position_bias
        
        # Causal mask for decoder
        if self.is_decoder and encoder_hidden_states is None:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o(out)


class T5DenseGatedActDense(nn.Module):
    """T5 v1.1 gated feed-forward (GeGLU-like)."""
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.wi_0 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.wi_1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.wo = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_gelu = F.gelu(self.wi_0(x), approximate='tanh')
        hidden_linear = self.wi_1(x)
        x = hidden_gelu * hidden_linear
        x = self.dropout(x)
        return self.wo(x)


class Model(nn.Module):
    """
    T5 Encoder/Decoder Block
    
    The core repeated block in T5-style encoder-decoder transformers.
    Used by: T5, Flan-T5, mT5, UL2
    
    Architecture (Pre-LN):
        x -> LayerNorm -> Self-Attention -> + residual
          -> LayerNorm -> [Cross-Attention -> + residual] (decoder only)
          -> LayerNorm -> GeGLU FFN -> + residual
    
    Key features:
    - Pre-LayerNorm (norm before sublayer)
    - Relative position bias (no absolute positional embeddings)
    - Gated activation (T5 v1.1+)
    """
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int,
                 is_decoder: bool = False, dropout: float = 0.1):
        super().__init__()
        self.is_decoder = is_decoder
        
        # Self-attention
        self.layer_norm = T5LayerNorm(hidden_size)
        self.self_attn = T5Attention(hidden_size, num_heads, is_decoder=is_decoder, dropout=dropout)
        
        # Cross-attention (decoder only)
        if is_decoder:
            self.cross_attn_layer_norm = T5LayerNorm(hidden_size)
            self.cross_attn = T5Attention(
                hidden_size, num_heads, is_decoder=True,
                has_relative_attention_bias=False, dropout=dropout
            )
        
        # FFN
        self.final_layer_norm = T5LayerNorm(hidden_size)
        self.mlp = T5DenseGatedActDense(hidden_size, intermediate_size, dropout)

    def forward(self, x: torch.Tensor, encoder_hidden_states: torch.Tensor = None) -> torch.Tensor:
        # Self-attention
        residual = x
        x = self.layer_norm(x)
        x = self.self_attn(x)
        x = residual + x
        
        # Cross-attention (decoder only)
        if self.is_decoder and encoder_hidden_states is not None:
            residual = x
            x = self.cross_attn_layer_norm(x)
            x = self.cross_attn(x, encoder_hidden_states)
            x = residual + x
        
        # FFN
        residual = x
        x = self.final_layer_norm(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


# Benchmark configuration (T5-large dimensions)
batch_size = 16
seq_len = 512
hidden_size = 1024
num_heads = 16
intermediate_size = 2816
is_decoder = False
dropout = 0.0

def get_inputs():
    return [torch.randn(batch_size, seq_len, hidden_size)]

def get_init_inputs():
    return [hidden_size, num_heads, intermediate_size, is_decoder, dropout]

