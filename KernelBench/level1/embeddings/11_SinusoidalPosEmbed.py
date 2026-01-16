import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn
import math

class Model(nn.Module):
    """
    Sinusoidal Positional Embedding
    
    Used by: Original Transformer, BERT, Whisper encoder, T5, BART
    
    Fixed (non-learned) positional embeddings using sine and cosine functions
    at different frequencies. Allows generalization to longer sequences.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Shapes:
        Input: (batch_size, seq_length) or seq_length positions
        Output: (seq_length, hidden_size) or (batch_size, seq_length, hidden_size)
    """
    
    def __init__(self, hidden_size: int = 768, max_seq_length: int = 8192):
        """
        Initialize Sinusoidal Positional Embedding.
        
        Args:
            hidden_size: Embedding dimension (must be even)
            max_seq_length: Maximum sequence length to precompute
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        
        # Precompute positional embeddings
        pe = torch.zeros(max_seq_length, hidden_size)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, seq_length: int) -> torch.Tensor:
        """
        Get positional embeddings for given sequence length.
        
        Args:
            seq_length: Length of sequence
            
        Returns:
            Positional embeddings (seq_length, hidden_size)
        """
        return self.pe[:seq_length]


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 32, "seq_length": 512, "hidden_size": 768, "max_seq_length": 8192},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("embeddings", "11_SinusoidalPosEmbed")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    # This model takes a scalar seq_length, not a tensor
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    return [p["seq_length"]]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["hidden_size"], p["max_seq_length"]]
