import torch
import torch.nn as nn

class Model(nn.Module):
    """Fused Embedding + Factorization Machine (DeepFM)."""
    
    def __init__(self, vocab_size: int, embed_dim: int, num_features: int):
        super(Model, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(num_features * embed_dim, 1)
    
    def forward(self, x):
        e = self.embed(x)  # (B, F, D)
        sum_sq = e.sum(1).pow(2).sum(1, keepdim=True)
        sq_sum = e.pow(2).sum(1).sum(1, keepdim=True)
        fm = 0.5 * (sum_sq - sq_sum)
        return self.linear(e.view(e.size(0), -1)) + fm

batch_size, num_features, vocab_size, embed_dim = 4096, 26, 1000000, 16
def get_inputs(): return [torch.randint(0, vocab_size, (batch_size, num_features), device='cuda')]
def get_init_inputs(): return [vocab_size, embed_dim, num_features]

