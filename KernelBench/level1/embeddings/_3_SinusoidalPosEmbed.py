import os
import sys
import torch
import torch.nn as nn
import math
from typing import Literal

class Model(nn.Module):
    """
    Sinusoidal Positional Embedding
    
    Used by: Original Transformer, BERT, Whisper encoder, T5, BART,
             Qwen3-Omni (audio encoder)
    
    Fixed (non-learned) positional embeddings using sine and cosine functions
    at different frequencies. Allows generalization to longer sequences.
    
    Supports two output layouts:
    - "interleaved" (default): [sin_0, cos_0, sin_1, cos_1, ...]
        PE(pos, 2i) = sin(pos / base^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / base^(2i/d_model))
    - "concatenated": [sin_0, sin_1, ..., cos_0, cos_1, ...]
        PE(pos, :d/2) = sin(pos / base^(2i/d_model))
        PE(pos, d/2:) = cos(pos / base^(2i/d_model))
      Used by Whisper-style audio encoders (e.g. Qwen3-Omni).
    
    Shapes:
        Input: seq_length (int)
        Output: (seq_length, hidden_size)
    """
    
    def __init__(self, hidden_size: int = 768, max_seq_length: int = 8192,
                 base: float = 10000.0,
                 mode: Literal["interleaved", "concatenated"] = "interleaved"):
        """
        Initialize Sinusoidal Positional Embedding.
        
        Args:
            hidden_size: Embedding dimension (must be even)
            max_seq_length: Maximum sequence length to precompute
            base: Base for the frequency computation (default: 10000.0)
            mode: Output layout.
                "interleaved" - [sin_0, cos_0, sin_1, cos_1, ...]
                "concatenated" - [sin_all, cos_all]
        """
        super(Model, self).__init__()
        if hidden_size % 2 != 0:
            raise ValueError("SinusoidalPosEmbed requires even hidden_size")
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.mode = mode
        
        # Precompute positional embeddings
        half_dim = hidden_size // 2
        log_timescale_increment = math.log(base) / (half_dim - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(half_dim, dtype=torch.float))
        scaled_time = torch.arange(max_seq_length, dtype=torch.float).unsqueeze(1) * inv_timescales.unsqueeze(0)
        
        if mode == "concatenated":
            pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        else:
            pe = torch.zeros(max_seq_length, hidden_size)
            pe[:, 0::2] = torch.sin(scaled_time)
            pe[:, 1::2] = torch.cos(scaled_time)
        
        self.register_buffer('pe', pe, persistent=False)
    
    def forward(self, seq_length: int) -> torch.Tensor:
        """
        Get positional embeddings for given sequence length.
        
        Args:
            seq_length: Length of sequence
            
        Returns:
            Positional embeddings (seq_length, hidden_size)
        """
        return self.pe[:seq_length]
