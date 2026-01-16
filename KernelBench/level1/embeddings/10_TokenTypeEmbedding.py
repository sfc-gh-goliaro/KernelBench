import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Token Type Embedding (Segment Embedding)
    
    Used by: BERT, RoBERTa, ALBERT (for sentence pair tasks)
    
    Embeds token type IDs (e.g., 0 for sentence A, 1 for sentence B).
    Used in encoder models to distinguish between segments.
    
    Shapes:
        Input: (batch_size, seq_length) - token type IDs
        Output: (batch_size, seq_length, hidden_size)
    """
    
    def __init__(self, num_token_types: int = 2, hidden_size: int = 768):
        """
        Initialize Token Type Embedding.
        
        Args:
            num_token_types: Number of token types (default 2 for BERT)
            hidden_size: Embedding dimension
        """
        super(Model, self).__init__()
        self.token_type_embeddings = nn.Embedding(num_token_types, hidden_size)
    
    def forward(self, token_type_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up token type embeddings.
        
        Args:
            token_type_ids: Token type IDs (batch_size, seq_length)
            
        Returns:
            Token type embeddings (batch_size, seq_length, hidden_size)
        """
        return self.token_type_embeddings(token_type_ids)


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 32, "seq_length": 512, "num_token_types": 2, "hidden_size": 768},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("embeddings", "10_TokenTypeEmbedding")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    # Token type IDs are indices
    token_type_ids = DISTRIBUTIONS["indices"]((p["batch_size"], p["seq_length"]), p["num_token_types"], dtype=torch.int64, device=device)
    return [token_type_ids]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["num_token_types"], p["hidden_size"]]
