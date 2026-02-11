import os
import sys
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
