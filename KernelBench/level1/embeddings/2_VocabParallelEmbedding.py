import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Vocabulary Parallel Embedding
    
    Used by: All LLMs (fundamental operator)
    
    Token embedding lookup with vocabulary sharding across tensor parallel
    ranks. Each rank holds a shard of the embedding table.
    
    Shapes:
        Input: (batch_size, seq_len) token indices
        Output: (batch_size, seq_len, hidden_size)
    """
    
    def __init__(self, vocab_size: int, hidden_size: int, padding_idx: int = None):
        """
        Initialize vocabulary parallel embedding.
        
        Args:
            vocab_size: Size of vocabulary (or shard size for TP)
            hidden_size: Embedding dimension
            padding_idx: Index for padding token
        """
        super(Model, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up embeddings for input tokens.
        
        Args:
            input_ids: Token indices of shape (batch_size, seq_len)
            
        Returns:
            Embeddings of shape (batch_size, seq_len, hidden_size)
        """
        return self.embedding(input_ids)


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "vocab_size": 32000, "hidden_size": 4096},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("embeddings", "2_VocabParallelEmbedding")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    # Embedding requires indices, not float tensors
    input_ids = DISTRIBUTIONS["indices"]((p["batch_size"], p["seq_length"]), p["vocab_size"], dtype=torch.int64, device=device)
    return [input_ids]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["vocab_size"], p["hidden_size"]]
