import os
import sys
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Hash Embedding
    
    Used by: Large-scale RecSys
    
    Hashing trick for very large categorical feature spaces.
    
    Shapes:
        Input: (batch, num_features) IDs (can be very large)
        Output: (batch, num_features, embed_dim)
    """
    
    def __init__(self, num_buckets: int, embed_dim: int):
        super(Model, self).__init__()
        self.num_buckets = num_buckets
        self.embedding = nn.Embedding(num_buckets, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Hash to bucket
        hashed = x % self.num_buckets
        return self.embedding(hashed)
