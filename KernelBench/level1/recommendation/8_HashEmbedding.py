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


batch_size = 4096
num_features = 26
num_buckets = 100000
embed_dim = 16

def get_inputs():
    x = torch.randint(0, 10000000, (batch_size, num_features), device='cuda')
    return [x]

def get_init_inputs():
    return [num_buckets, embed_dim]

