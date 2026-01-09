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


batch_size = 4096
num_features = 26
vocab_size = 1000000
embed_dim = 16

def get_inputs():
    x = torch.randint(0, vocab_size, (batch_size, num_features), device='cuda')
    return [x]

def get_init_inputs():
    return [vocab_size, embed_dim, num_features]

