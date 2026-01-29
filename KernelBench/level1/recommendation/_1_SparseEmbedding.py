import os
import sys
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Sparse Embedding Lookup
    
    Used by: DeepFM, WDL, AutoInt
    
    Sparse embedding lookup for categorical features with huge vocab.
    
    Shapes:
        Input: (batch, num_features) categorical indices
        Output: (batch, num_features, embed_dim)
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, num_features: int):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.num_features = num_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)
