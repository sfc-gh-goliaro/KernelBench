import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class WhisperAttention(nn.Module):
    """Whisper encoder self-attention (bidirectional)."""
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
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(out)


class WhisperMLP(nn.Module):
    """Whisper encoder MLP with GELU."""
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class Model(nn.Module):
    """
    Whisper Encoder Layer
    
    The core repeated block in Whisper audio encoder.
    Used by: Whisper (all sizes), Whisper-style audio encoders
    
    Architecture:
        x -> LayerNorm -> Self-Attention -> + residual
          -> LayerNorm -> FFN (GELU) -> + residual
    
    Note: Whisper encoder processes mel-spectrogram features
    that have been passed through 2 Conv1d layers.
    """
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hidden_size)
        self.self_attn = WhisperAttention(hidden_size, num_heads, dropout)
        self.final_layer_norm = nn.LayerNorm(hidden_size)
        self.mlp = WhisperMLP(hidden_size, intermediate_size, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with pre-norm
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x)
        x = residual + x
        
        # FFN with pre-norm
        residual = x
        x = self.final_layer_norm(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


# Benchmark configuration (Whisper-large-v3 dimensions)
batch_size = 8
seq_len = 1500  # 30s of audio at 50 Hz
hidden_size = 1280
num_heads = 20
intermediate_size = 5120
dropout = 0.0

def get_inputs():
    return [torch.randn(batch_size, seq_len, hidden_size)]

def get_init_inputs():
    return [hidden_size, num_heads, intermediate_size, dropout]

