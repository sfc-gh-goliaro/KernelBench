import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class WhisperSelfAttention(nn.Module):
    """Whisper decoder causal self-attention."""
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(causal_mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(out)


class WhisperCrossAttention(nn.Module):
    """Whisper decoder cross-attention to encoder."""
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        encoder_len = encoder_hidden_states.shape[1]
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(encoder_hidden_states).view(batch_size, encoder_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(encoder_hidden_states).view(batch_size, encoder_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(out)


class WhisperMLP(nn.Module):
    """Whisper decoder MLP."""
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class Model(nn.Module):
    """
    Whisper Decoder Layer
    
    The core repeated block in Whisper audio decoder.
    Used by: Whisper (all sizes), encoder-decoder ASR models
    
    Architecture:
        x -> LayerNorm -> Causal Self-Attention -> + residual
          -> LayerNorm -> Cross-Attention (to encoder) -> + residual
          -> LayerNorm -> FFN (GELU) -> + residual
    
    Note: Cross-attention attends to audio encoder hidden states.
    """
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        # Self-attention
        self.self_attn_layer_norm = nn.LayerNorm(hidden_size)
        self.self_attn = WhisperSelfAttention(hidden_size, num_heads, dropout)
        
        # Cross-attention
        self.encoder_attn_layer_norm = nn.LayerNorm(hidden_size)
        self.encoder_attn = WhisperCrossAttention(hidden_size, num_heads, dropout)
        
        # FFN
        self.final_layer_norm = nn.LayerNorm(hidden_size)
        self.mlp = WhisperMLP(hidden_size, intermediate_size, dropout)

    def forward(self, x: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        # Causal self-attention
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x)
        x = residual + x
        
        # Cross-attention to encoder
        residual = x
        x = self.encoder_attn_layer_norm(x)
        x = self.encoder_attn(x, encoder_hidden_states)
        x = residual + x
        
        # FFN
        residual = x
        x = self.final_layer_norm(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


# Benchmark configuration (Whisper-large-v3 dimensions)
batch_size = 8
decoder_seq_len = 448  # Max tokens
encoder_seq_len = 1500  # Audio features
hidden_size = 1280
num_heads = 20
intermediate_size = 5120
dropout = 0.0

def get_inputs():
    decoder_input = torch.randn(batch_size, decoder_seq_len, hidden_size)
    encoder_hidden = torch.randn(batch_size, encoder_seq_len, hidden_size)
    return [decoder_input, encoder_hidden]

def get_init_inputs():
    return [hidden_size, num_heads, intermediate_size, dropout]

